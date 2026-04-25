"""Feature search ranking + cache + fallback behaviour."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from persona.config import SAEConfig
from persona.feature_search import (
    FeatureMatch,
    _keyword_overlap,
    _tokenize,
    search_features,
)


def _stub_search_response():
    """A canned Neuronpedia-shaped response."""
    return {
        "request": {"query": "fast food"},
        "results": [
            {
                "modelId": "llama3.1-8b",
                "layer": "15-llamascope-res-32k",
                "index": 12345,
                "description": "completely unrelated descriptor about prime numbers",
                "neuron": {"activations": []},
            },
            {
                "modelId": "llama3.1-8b",
                "layer": "15-llamascope-res-32k",
                "index": 67890,
                "description": "fast food restaurants and burger chains like McDonald's",
                "neuron": {"activations": [{"tokens": ["McDonald", "burger", "fries"], "values": [3.0, 2.5, 2.0]}]},
            },
            {
                "modelId": "llama3.1-8b",
                "layer": "15-llamascope-res-32k",
                "index": 11111,
                "description": "food preparation and cooking",
                "neuron": {"activations": []},
            },
        ],
        "resultsCount": 3,
        "hasMore": False,
    }


def test_tokenize_lowercases_and_drops_punct():
    assert _tokenize("Acme Bank's customer service!") == ["acme", "bank's", "customer", "service"]
    assert _tokenize("") == []


def test_keyword_overlap_zero_for_no_query():
    assert _keyword_overlap([], "anything") == 0.0


def test_keyword_overlap_fraction():
    q = ["fast", "food", "restaurants"]
    assert _keyword_overlap(q, "fast food places") == pytest.approx(2 / 3)
    assert _keyword_overlap(q, "completely unrelated") == 0.0


def test_search_filters_to_configured_source(tmp_path, monkeypatch):
    """The HTTP body must include the right model and source."""
    monkeypatch.chdir(tmp_path)
    captured = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, path, json, timeout):
            captured["path"] = path
            captured["body"] = json
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = _stub_search_response()
            return r

    with patch("persona.feature_search.httpx.Client", return_value=FakeClient()):
        # Disable embedding rerank to keep test deterministic.
        with patch("persona.feature_search._try_load_st_model", return_value=None):
            results = search_features(
                query="fast food",
                sae_config=SAEConfig(),
                top_k=10,
                use_cache=False,
            )

    assert captured["path"] == "/api/explanation/search"
    assert captured["body"]["modelId"] == "llama3.1-8b"
    assert captured["body"]["layers"] == ["15-llamascope-res-32k"]
    assert captured["body"]["query"] == "fast food"
    assert len(results) == 3


def test_keyword_rerank_promotes_relevant(tmp_path, monkeypatch):
    """Without an embedding model, keyword overlap should put fast-food first."""
    monkeypatch.chdir(tmp_path)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, path, json, timeout):
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = _stub_search_response()
            return r

    with patch("persona.feature_search.httpx.Client", return_value=FakeClient()):
        with patch("persona.feature_search._try_load_st_model", return_value=None):
            results = search_features(
                query="fast food",
                sae_config=SAEConfig(),
                top_k=10,
                use_cache=False,
            )
    # The fast-food explanation should now be ranked first by keyword overlap.
    assert results[0].feature_id == 67890
    assert results[0].rerank_score > 0
    # The unrelated 'prime numbers' descriptor should be last.
    assert results[-1].feature_id == 12345


def test_cache_round_trip(tmp_path, monkeypatch):
    """A cached search returns the same matches without making a second HTTP call."""
    monkeypatch.chdir(tmp_path)

    class FakeClient:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, path, json, timeout):
            FakeClient.calls += 1
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = _stub_search_response()
            return r

    with patch("persona.feature_search.httpx.Client", return_value=FakeClient()):
        with patch("persona.feature_search._try_load_st_model", return_value=None):
            r1 = search_features("fast food", SAEConfig(), top_k=10, use_cache=True)
            r2 = search_features("fast food", SAEConfig(), top_k=10, use_cache=True)
    assert FakeClient.calls == 1
    assert [m.feature_id for m in r1] == [m.feature_id for m in r2]


def test_short_query_rejected():
    with pytest.raises(ValueError):
        search_features("ab", SAEConfig(), top_k=5)


def test_empty_results_returns_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, path, json, timeout):
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"results": []}
            return r

    with patch("persona.feature_search.httpx.Client", return_value=FakeClient()):
        with patch("persona.feature_search._try_load_st_model", return_value=None):
            results = search_features("zorglub xyzzy", SAEConfig(), top_k=10, use_cache=False)
    assert results == []


def test_short_snippet_truncates():
    m = FeatureMatch(feature_id=1, explanation="x" * 200, layer="l", model_id="m")
    assert len(m.short_snippet(50)) == 50
    assert m.short_snippet(50).endswith("\u2026")
