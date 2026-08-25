from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import respx


def _hits() -> list[dict]:
    return [
        {
            "page_id": "file_page:src/first.py",
            "title": "First",
            "target_path": "src/first.py",
            "snippet": "weak match",
            "relevance_score": 4.8,
            "sources": ["fts", "vector"],
        },
        {
            "page_id": "file_page:src/second.py",
            "title": "Second",
            "target_path": "src/second.py",
            "snippet": "strong match",
            "relevance_score": 4.2,
            "sources": ["vector"],
        },
        {
            "page_id": "file_page:src/third.py",
            "title": "Third",
            "target_path": "src/third.py",
            "snippet": "tail",
            "relevance_score": 3.9,
            "sources": ["fts"],
        },
    ]


@pytest.mark.asyncio
async def test_rank_only_rerank_preserves_score_geometry_and_provenance(monkeypatch):
    from repowise.server.mcp_server._model_rerank import rerank_candidates

    monkeypatch.setenv("REPOWISE_RERANK_MODEL", "rerank-nemotron")
    monkeypatch.setenv("REPOWISE_RERANK_BASE_URL", "http://reranker.test")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://reranker.test/rerank"
        payload = __import__("json").loads(request.content)
        assert payload["model"] == "rerank-nemotron"
        assert payload["query"] == "where is the strong match"
        assert len(payload["documents"]) == 3
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.93},
                    {"index": 0, "relevance_score": 0.31},
                    {"index": 2, "relevance_score": 0.10},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await rerank_candidates(
            "where is the strong match", _hits(), score_key="relevance_score", client=client
        )
    finally:
        await client.aclose()

    assert [hit["page_id"] for hit in result.hits] == [
        "file_page:src/second.py",
        "file_page:src/first.py",
        "file_page:src/third.py",
    ]
    assert [hit["relevance_score"] for hit in result.hits] == [4.8, 4.2, 3.9]
    assert [hit["sources"] for hit in result.hits] == [
        ["vector"],
        ["fts", "vector"],
        ["fts"],
    ]
    assert result.meta == {
        "status": "ok",
        "model": "rerank-nemotron",
        "candidates": 3,
    }


@pytest.mark.asyncio
async def test_reranker_failure_is_explicit_and_fail_open(monkeypatch):
    from repowise.server.mcp_server._model_rerank import rerank_candidates

    monkeypatch.setenv("REPOWISE_RERANK_MODEL", "rerank-nemotron")
    monkeypatch.setenv("REPOWISE_RERANK_BASE_URL", "http://reranker.test")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    original = _hits()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await rerank_candidates(
            "query", original, score_key="relevance_score", client=client
        )
    finally:
        await client.aclose()

    assert result.hits == original
    assert result.meta == {
        "status": "error",
        "model": "rerank-nemotron",
        "candidates": 3,
    }


@pytest.mark.asyncio
async def test_reranker_is_disabled_without_a_model(monkeypatch):
    from repowise.server.mcp_server._model_rerank import rerank_candidates

    monkeypatch.delenv("REPOWISE_RERANK_MODEL", raising=False)
    original = _hits()

    result = await rerank_candidates("query", original, score_key="relevance_score")

    assert result.hits == original
    assert result.meta is None


@pytest.mark.asyncio
@respx.mock
async def test_semantic_search_uses_reranker_and_reports_it(monkeypatch):
    import repowise.server.mcp_server.tool_search as search_mod
    from repowise.server.mcp_server import search_codebase

    monkeypatch.setenv("REPOWISE_RERANK_MODEL", "rerank-nemotron")
    monkeypatch.setenv("REPOWISE_RERANK_BASE_URL", "http://reranker.test")

    fused = [
        {
            "page_id": "file_page:src/auth/service.py",
            "title": "Auth Service",
            "page_type": "file_page",
            "snippet": "authentication service",
            "relevance_score": 5.0,
            "sources": ["fts", "vector"],
        },
        {
            "page_id": "file_page:src/db/models.py",
            "title": "DB Models",
            "page_type": "file_page",
            "snippet": "database models",
            "relevance_score": 4.0,
            "sources": ["fts", "vector"],
        },
    ]

    async def fake_context(_repo):
        return SimpleNamespace(
            session_factory=object(), vector_store_ready=None, path="/tmp/test-repo"
        )

    class FakeSession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return None

    async def fake_repo(_session):
        return SimpleNamespace()

    async def fake_fused(_ctx, _query, _limit, _page_type):
        return [dict(hit) for hit in fused]

    async def fake_page_info(_session, _output, *, with_git=False):
        return (
            {
                "file_page:src/auth/service.py": "src/auth/service.py",
                "file_page:src/db/models.py": "src/db/models.py",
            },
            set(),
            {},
        )

    async def ready(_ctx):
        return None

    monkeypatch.setattr(search_mod, "_resolve_repo_context", fake_context)
    monkeypatch.setattr(search_mod, "get_session", lambda _factory: FakeSession())
    monkeypatch.setattr(search_mod, "_get_repo", fake_repo)
    monkeypatch.setattr(search_mod, "_fused_retrieve", fake_fused)
    monkeypatch.setattr(search_mod, "_load_page_info", fake_page_info)
    monkeypatch.setattr(search_mod, "_wait_for_vector_store", ready)
    monkeypatch.setattr(search_mod, "_get_exclude_spec", lambda _path: None)
    respx.post("http://reranker.test/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )
    )

    response = await search_codebase("authentication database architecture", mode="concept")

    assert response["results"][0]["target_path"] == "src/db/models.py"
    assert response["_meta"]["reranker"] == {
        "status": "ok",
        "model": "rerank-nemotron",
        "candidates": 2,
    }
    assert response["results"][0]["sources"] == ["fts", "vector"]


@pytest.mark.asyncio
async def test_exact_path_search_never_calls_reranker(monkeypatch):
    import repowise.server.mcp_server.tool_search as search_mod
    from repowise.server.mcp_server import search_codebase

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("exact/path modes must not invoke the reranker")

    async def fake_structured(*_args, **_kwargs):
        return {"results": [], "mode": "path", "_meta": {}}

    monkeypatch.setattr(search_mod, "rerank_candidates", forbidden)
    monkeypatch.setattr(search_mod, "_structured_search", fake_structured)

    response = await search_codebase("src/auth/service.py", mode="path")

    assert response["mode"] == "path"
    assert "reranker" not in response["_meta"]


def test_answer_cache_identity_includes_reranker_model(monkeypatch):
    from repowise.server.mcp_server.tool_answer.synthesis import _hash_answer_identity

    monkeypatch.delenv("REPOWISE_RERANK_MODEL", raising=False)
    without_reranking = _hash_answer_identity("how does auth work", None)
    monkeypatch.setenv("REPOWISE_RERANK_MODEL", "rerank-nemotron")
    monkeypatch.setenv("REPOWISE_RERANK_BASE_URL", "http://reranker.test")

    assert _hash_answer_identity("how does auth work", None) != without_reranking


@pytest.mark.asyncio
async def test_get_answer_pipeline_reranks_before_calibrated_policies(monkeypatch):
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server._answer_pipeline import begin_leg_record, retrieval_legs
    from repowise.server.mcp_server._model_rerank import RerankResult

    initial = [
        {"page_id": "first", "title": "First", "score": 5.0, "snippet": "same words"},
        {"page_id": "second", "title": "Second", "score": 4.0, "snippet": "same words"},
    ]
    policy_input: list[str] = []

    async def retrieve(_question, _ctx):
        begin_leg_record()
        return [dict(hit) for hit in initial]

    async def hydrate(hits, _ctx, *, scope):
        for hit in hits:
            hit["target_path"] = f"src/{hit['page_id']}.py"
            hit["page_type"] = "file_page"
        return hits

    async def rerank(_question, hits, *, score_key):
        assert score_key == "score"
        return RerankResult(
            [dict(hits[1], score=5.0), dict(hits[0], score=4.0)],
            {"status": "ok", "model": "rerank-nemotron", "candidates": 2},
        )

    def coverage(hits, _question):
        policy_input.extend(hit["page_id"] for hit in hits)
        return hits

    async def keep_hits(hits, *_args, **_kwargs):
        return hits

    async def keep_hits_and_paths(_session, _repo_id, hits, *_args, **_kwargs):
        return hits, []

    async def keep_third(_session, _repo_id, hits, *_args, **_kwargs):
        return hits

    class FakeSession:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(answer_mod, "_hybrid_retrieve", retrieve)
    monkeypatch.setattr(answer_mod, "_hydrate_hits", hydrate)
    monkeypatch.setattr(answer_mod, "rerank_candidates", rerank)
    monkeypatch.setattr(answer_mod, "_rerank_by_coverage", coverage)
    monkeypatch.setattr(answer_mod, "_apply_domain_penalty", lambda *_args: None)
    monkeypatch.setattr(answer_mod, "_intersection_boost", keep_hits)
    monkeypatch.setattr(answer_mod, "_apply_pagerank_bias", keep_hits)
    monkeypatch.setattr(answer_mod, "_expand_via_graph", keep_hits)
    monkeypatch.setattr(answer_mod, "_expand_via_flow_path", keep_hits_and_paths)
    monkeypatch.setattr(answer_mod, "_expand_via_neighbor_rerank", keep_third)
    monkeypatch.setattr(answer_mod, "_expand_via_parent_page", keep_hits)
    monkeypatch.setattr(answer_mod, "_demote_noise_hits", lambda hits, *_args, **_kwargs: hits)
    monkeypatch.setattr(answer_mod, "get_session", lambda _factory: FakeSession())
    monkeypatch.setattr(answer_mod, "_hydrate_symbols_for_hits", keep_hits)
    monkeypatch.setattr(answer_mod, "_hydrate_candidate_defines", keep_hits)

    result = await answer_mod._run_retrieval_pipeline(
        "where does auth live",
        SimpleNamespace(session_factory=object(), path="/tmp/test-repo"),
        scope=None,
        exclude_spec=None,
        repo_id="repo",
    )

    assert policy_input == ["second", "first"]
    assert [hit["page_id"] for hit in result.hits] == ["second", "first"]
    assert retrieval_legs()["rerank"] == "ok"
