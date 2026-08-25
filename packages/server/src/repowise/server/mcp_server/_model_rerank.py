"""Optional cross-encoder reranking for semantic retrieval candidates.

The model changes ranks, not score geometry.  Existing fused scores are
reassigned in descending order to the model's ordering so every downstream
heuristic keeps operating on the scale it was calibrated against.
"""

from __future__ import annotations

import contextvars
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import httpx

_log = logging.getLogger("repowise.mcp.rerank")

_MODEL_ENV = "REPOWISE_RERANK_MODEL"
_BASE_URL_ENV = "REPOWISE_RERANK_BASE_URL"
_API_KEY_ENV = "REPOWISE_RERANK_API_KEY"
_TIMEOUT_ENV = "REPOWISE_RERANK_TIMEOUT_S"
_CANDIDATES_ENV = "REPOWISE_RERANK_CANDIDATES"

_DEFAULT_TIMEOUT_S = 30.0
_MAX_TIMEOUT_S = 120.0
_DEFAULT_CANDIDATES = 15
_MAX_CANDIDATES = 50
_MAX_DOCUMENT_CHARS = 6_000

_RERANK_RECORD: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "repowise_rerank_record", default=None
)


@dataclass(frozen=True)
class RerankResult:
    hits: list[dict]
    meta: dict[str, Any] | None


def begin_rerank_record() -> None:
    """Clear request-local reranker telemetry."""
    _RERANK_RECORD.set(None)


def rerank_record() -> dict[str, Any] | None:
    """Return request-local reranker telemetry, if the stage was enabled."""
    record = _RERANK_RECORD.get()
    return dict(record) if record is not None else None


def reranker_cache_identity() -> str | None:
    """Trust-relevant reranker config that makes answer-cache entries distinct."""
    model = os.environ.get(_MODEL_ENV, "").strip()
    base_url = _base_url()
    if not model or not base_url:
        return None
    return f"{model}|{base_url}|{_candidate_limit()}"


def _bounded_number(name: str, default: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        _log.warning("Ignoring unusable %s=%r", name, raw)
        return default
    if not math.isfinite(parsed) or parsed <= 0:
        _log.warning("Ignoring unusable %s=%r", name, raw)
        return default
    return min(parsed, maximum)


def _candidate_limit() -> int:
    return int(_bounded_number(_CANDIDATES_ENV, _DEFAULT_CANDIDATES, _MAX_CANDIDATES))


def _base_url() -> str:
    explicit = os.environ.get(_BASE_URL_ENV, "").strip()
    if explicit:
        return explicit.rstrip("/")
    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _document(hit: dict) -> str:
    parts = [
        hit.get("title") or "",
        hit.get("target_path") or hit.get("file") or "",
        hit.get("summary") or hit.get("snippet") or hit.get("excerpt") or "",
    ]
    return "\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())[
        :_MAX_DOCUMENT_CHARS
    ]


def _meta(status: str, model: str, candidates: int) -> dict[str, Any]:
    previous = _RERANK_RECORD.get()
    severity = {"ok": 0, "timeout": 1, "error": 2}
    if previous is not None:
        status = max((previous["status"], status), key=severity.__getitem__)
        candidates += int(previous["candidates"])
    value = {"status": status, "model": model, "candidates": candidates}
    _RERANK_RECORD.set(value)
    return value


def _parse_order(payload: Any, count: int) -> list[int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("rerank response has no results list")
    scored: list[tuple[float, int]] = []
    seen: set[int] = set()
    for result in payload["results"]:
        if not isinstance(result, dict):
            raise ValueError("rerank result is not an object")
        index = result.get("index")
        score = result.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            raise ValueError("rerank result index is invalid")
        if index in seen:
            raise ValueError("rerank response contains a duplicate index")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("rerank result score is invalid")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("rerank result score is not finite")
        seen.add(index)
        scored.append((numeric_score, index))
    if len(seen) != count:
        raise ValueError("rerank response omitted candidates")
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [index for _, index in scored]


async def rerank_candidates(
    query: str,
    hits: list[dict],
    *,
    score_key: str,
    client: httpx.AsyncClient | None = None,
) -> RerankResult:
    """Rerank a bounded leading window, failing open to the original order."""
    model = os.environ.get(_MODEL_ENV, "").strip()
    base_url = _base_url()
    if not model or not base_url or len(hits) < 2:
        return RerankResult(hits, None)

    count = min(len(hits), _candidate_limit())
    payload = {
        "model": model,
        "query": query,
        "documents": [_document(hit) for hit in hits[:count]],
        "top_n": count,
        "return_documents": False,
    }
    headers: dict[str, str] = {}
    api_key = (
        os.environ.get(_API_KEY_ENV, "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=_bounded_number(_TIMEOUT_ENV, _DEFAULT_TIMEOUT_S, _MAX_TIMEOUT_S)
        )
    try:
        response = await client.post(f"{base_url}/rerank", json=payload, headers=headers)
        response.raise_for_status()
        order = _parse_order(response.json(), count)
    except httpx.TimeoutException:
        _log.warning("Reranker exceeded its request budget; preserving fused ranking")
        return RerankResult(hits, _meta("timeout", model, count))
    except Exception:
        _log.warning("Reranker failed; preserving fused ranking", exc_info=True)
        return RerankResult(hits, _meta("error", model, count))
    finally:
        if owns_client:
            await client.aclose()

    score_geometry = sorted(
        (float(hit.get(score_key) or 0.0) for hit in hits[:count]), reverse=True
    )
    reranked: list[dict] = []
    for rank, index in enumerate(order):
        hit = dict(hits[index])
        hit[score_key] = score_geometry[rank]
        reranked.append(hit)
    reranked.extend(hits[count:])
    return RerankResult(reranked, _meta("ok", model, count))
