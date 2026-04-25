"""Persona description -> SAE feature candidates via Neuronpedia.

Neuronpedia's `/api/explanation/search` endpoint already does semantic search
(it embeds the query server-side via OpenAI and ANN-searches feature
explanations). We pass the persona description verbatim, filter to the SAE
release we use, and present top-K results to the user.

Substring + lowercase keyword overlap is used as a tie-breaker re-rank because
the server-side ranker is sometimes too charitable (returns vaguely-related
explanations near the bottom). Optional cosine over
`sentence-transformers/all-MiniLM-L6-v2` is used if installed.

All fallbacks are printed verbatim per spec.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import SAEConfig

NEURONPEDIA_BASE = "https://www.neuronpedia.org"
SEARCH_ENDPOINT = "/api/explanation/search"
FEATURE_ENDPOINT_TPL = "/api/feature/{model}/{layer}/{index}"
CACHE_DIR = Path(".cache/persona/search")
DEFAULT_TIMEOUT = 30.0
MIN_QUERY_LEN = 3


@dataclass
class FeatureMatch:
    """One candidate feature returned by search + rerank."""

    feature_id: int
    explanation: str
    layer: str
    model_id: str
    rerank_score: float = 0.0
    server_rank: int = -1
    activations: list[dict[str, Any]] = field(default_factory=list)

    def top_tokens(self, max_tokens: int = 6) -> str:
        """Return a comma-separated preview of top-activating tokens, if any."""
        if not self.activations:
            return ""
        toks: list[str] = []
        for act in self.activations[:1]:
            tokens = act.get("tokens") or []
            values = act.get("values") or []
            if not tokens or not values:
                continue
            pairs = sorted(
                zip(tokens, values), key=lambda p: -float(p[1] or 0.0)
            )[:max_tokens]
            toks.extend(repr(t.strip()) for t, _ in pairs if t and str(t).strip())
        return ", ".join(toks)

    def short_snippet(self, max_chars: int = 70) -> str:
        return (
            self.explanation[: max_chars - 1] + "\u2026"
            if len(self.explanation) > max_chars
            else self.explanation
        )


def _cache_key(query: str, sae_config: SAEConfig, top_k: int) -> str:
    raw = f"{query}|{sae_config.neuronpedia_model}|{sae_config.neuronpedia_source}|{top_k}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str, ttl_seconds: int = 7 * 86400) -> list[FeatureMatch] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        with path.open() as f:
            raw = json.load(f)
        return [FeatureMatch(**r) for r in raw]
    except Exception:
        return None


def _write_cache(key: str, matches: list[FeatureMatch]) -> None:
    path = _cache_path(key)
    with path.open("w") as f:
        json.dump([asdict(m) for m in matches], f, indent=2)


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _keyword_overlap(query_tokens: list[str], explanation: str) -> float:
    if not query_tokens:
        return 0.0
    expl_tokens = set(_tokenize(explanation))
    if not expl_tokens:
        return 0.0
    hits = sum(1 for q in query_tokens if q in expl_tokens)
    return hits / max(len(query_tokens), 1)


def _try_load_st_model() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        return None
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"sentence-transformers found but model load failed ({e}); falling back to keyword overlap.")
        return None


def _embed_rerank(query: str, candidates: list[FeatureMatch]) -> None:
    """Mutates candidates in place: sets rerank_score from cosine similarity.

    Called only if sentence-transformers is installed AND embedding succeeds.
    """
    model = _try_load_st_model()
    if model is None:
        return
    print("Reranking with sentence-transformers/all-MiniLM-L6-v2 cosine.")
    import numpy as np  # local
    expl_texts = [c.explanation or "" for c in candidates]
    embs = model.encode([query] + expl_texts, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = embs[0]
    for cand, emb in zip(candidates, embs[1:]):
        cand.rerank_score = float(np.dot(q_emb, emb))


def _http_post(client: httpx.Client, path: str, body: dict) -> dict | None:
    try:
        r = client.post(path, json=body, timeout=DEFAULT_TIMEOUT)
    except httpx.HTTPError as e:
        print(f"Neuronpedia request failed: {e}. Try again in a few seconds; the API is rate-limited (200/hr).")
        return None
    if r.status_code == 429:
        print("Neuronpedia rate limit hit (429). Wait a minute and rerun.")
        return None
    if r.status_code >= 400:
        print(f"Neuronpedia returned {r.status_code}: {r.text[:200]}")
        return None
    try:
        return r.json()
    except Exception as e:
        print(f"Neuronpedia returned non-JSON response: {e}")
        return None


def _http_get(client: httpx.Client, path: str) -> dict | None:
    try:
        r = client.get(path, timeout=DEFAULT_TIMEOUT)
    except httpx.HTTPError as e:
        print(f"Neuronpedia GET failed: {e}")
        return None
    if r.status_code >= 400:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _build_client() -> httpx.Client:
    headers = {"Content-Type": "application/json", "User-Agent": "persona-steering/0.1"}
    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return httpx.Client(base_url=NEURONPEDIA_BASE, headers=headers)


def search_features(
    query: str,
    sae_config: SAEConfig,
    top_k: int = 10,
    use_cache: bool = True,
    rerank: bool = True,
) -> list[FeatureMatch]:
    """Search Neuronpedia for features matching `query`. Return top_k after rerank.

    Hits a local disk cache keyed by (query, source, top_k) with a 7-day TTL.
    """
    if not query or len(query.strip()) < MIN_QUERY_LEN:
        raise ValueError(
            f"Query must be at least {MIN_QUERY_LEN} characters. Got: {query!r}"
        )

    key = _cache_key(query, sae_config, top_k)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            print(f"(cached) {len(cached)} feature candidates for {query!r}")
            return cached[:top_k]

    body = {
        "modelId": sae_config.neuronpedia_model,
        "layers": [sae_config.neuronpedia_source],
        "query": query,
        "offset": 0,
    }

    print(f"Searching Neuronpedia: model={body['modelId']} source={body['layers'][0]} q={query!r}")
    with _build_client() as client:
        payload = _http_post(client, SEARCH_ENDPOINT, body)
        if payload is None or "results" not in payload:
            print(
                f"No response or empty results from Neuronpedia. "
                f"Check {NEURONPEDIA_BASE}/{sae_config.neuronpedia_model}/{sae_config.neuronpedia_source} "
                f"in a browser to confirm the source ID is valid."
            )
            return []
        raw_results = payload["results"] or []
        print(f"  server returned {len(raw_results)} results")

        matches: list[FeatureMatch] = []
        for rank, r in enumerate(raw_results):
            try:
                feature_id = int(r.get("index"))
            except (TypeError, ValueError):
                continue
            explanation = (r.get("description") or "").strip()
            layer = r.get("layer", sae_config.neuronpedia_source)
            model_id = r.get("modelId", sae_config.neuronpedia_model)
            neuron = r.get("neuron") or {}
            activations = neuron.get("activations") or []
            matches.append(
                FeatureMatch(
                    feature_id=feature_id,
                    explanation=explanation,
                    layer=layer,
                    model_id=model_id,
                    server_rank=rank,
                    activations=activations,
                )
            )

    if not matches:
        print(
            f"No matching features for {query!r}. Try a more specific persona description "
            f"(e.g. 'fast food chains and burger restaurants' instead of 'McDonald's')."
        )
        return []

    if rerank:
        _rerank(query, matches)

    matches = matches[:top_k]
    if use_cache:
        _write_cache(key, matches)
    return matches


def _rerank(query: str, matches: list[FeatureMatch]) -> None:
    """Rerank in place. Tries embedding first; falls back to keyword overlap."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore  # noqa: F401
        has_st = True
    except ImportError:
        has_st = False

    if has_st:
        _embed_rerank(query, matches)
    else:
        print("sentence-transformers not installed; using keyword overlap rerank "
              "(install with `pip install -e .[persona-rerank]` for better quality).")
        q_tokens = _tokenize(query)
        for m in matches:
            m.rerank_score = _keyword_overlap(q_tokens, m.explanation)

    # Stable sort: rerank_score desc, then server_rank asc as tie-break.
    matches.sort(key=lambda m: (-m.rerank_score, m.server_rank))


def fetch_max_activations(
    feature_id: int, sae_config: SAEConfig, n_acts: int = 3
) -> list[dict[str, Any]]:
    """Lazy fetch top activations for one feature. Used when the search response
    didn't include them."""
    path = FEATURE_ENDPOINT_TPL.format(
        model=sae_config.neuronpedia_model,
        layer=sae_config.neuronpedia_source,
        index=feature_id,
    )
    with _build_client() as client:
        data = _http_get(client, path)
    if not data:
        return []
    return (data.get("activations") or [])[:n_acts]
