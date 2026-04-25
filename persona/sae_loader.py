"""Load decoder vectors from a Llama-Scope SAE on Neuronpedia via sae_lens.

Returns dicts shaped exactly as `src/steering.py`'s `generate_steered_answer`
already consumes, so we can drop them into the upstream nnsight loop unchanged.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any

import torch

from .config import SAEConfig

_sae_cache: dict[tuple[str, str], Any] = {}


def _load_sae(release: str, sae_id: str, device: str) -> Any:
    """Load and memoize the SAE object for the lifetime of the process."""
    key = (release, sae_id)
    if key in _sae_cache:
        return _sae_cache[key]

    try:
        from sae_lens import SAE  # heavy import; defer
    except ImportError as e:
        print(
            "ERROR: sae_lens is not installed. Run `pip install sae-lens>=6.12.3`.",
            file=sys.stderr,
        )
        raise e

    print(f"Loading SAE {release}/{sae_id} (one-time per process; cached on disk)...")
    result = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    # sae_lens versions vary: sometimes returns SAE, sometimes (SAE, cfg, sparsity).
    if isinstance(result, tuple):
        sae = result[0]
    else:
        sae = result
    _sae_cache[key] = sae
    print(f"  loaded: d_sae={sae.W_dec.shape[0]} d_model={sae.W_dec.shape[1]}")
    return sae


def load_feature_vectors(
    sae_config: SAEConfig,
    feature_ids: list[int],
    device: str,
    coefficients: list[float] | None = None,
) -> list[dict]:
    """Return a list of steering_components dicts for `src/steering.py`.

    Each dict has keys: layer, feature, strength, vector. The vector is the
    L2-normalised decoder column for that feature, matching upstream convention.
    """
    if not feature_ids:
        return []

    if coefficients is None:
        coefficients = [0.0] * len(feature_ids)
    if len(coefficients) != len(feature_ids):
        raise ValueError(
            f"coefficients length {len(coefficients)} != feature_ids length {len(feature_ids)}"
        )

    sae = _load_sae(sae_config.release, sae_config.sae_id, device)
    n_features = sae.W_dec.shape[0]

    components: list[dict] = []
    for fid, coeff in zip(feature_ids, coefficients):
        if fid < 0 or fid >= n_features:
            raise ValueError(
                f"feature_id {fid} out of range for SAE "
                f"{sae_config.release}/{sae_config.sae_id} (size {n_features}). "
                f"Run `persona find-features` to discover valid feature ids."
            )
        vec = sae.W_dec[fid].detach().to(device, non_blocking=True).clone()
        norm = vec.norm()
        if norm.item() == 0.0:
            print(
                f"WARNING: decoder vector for feature {fid} is all zeros. "
                f"Steering with this feature will have no effect."
            )
        else:
            vec = vec / norm
        components.append(
            {
                "layer": sae_config.layer,
                "feature": fid,
                "strength": float(coeff),
                "vector": vec,
            }
        )
    return components


def update_strengths(components: list[dict], coefficients: list[float]) -> list[dict]:
    """Return new component dicts with strengths overridden. Vectors shared."""
    if len(components) != len(coefficients):
        raise ValueError(
            f"coefficients length {len(coefficients)} != components length {len(components)}"
        )
    return [
        {**c, "strength": float(s)} for c, s in zip(components, coefficients)
    ]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
