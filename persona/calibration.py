"""Compute the mean residual-stream norm at a layer for use as the sweep anchor.

Adapted from upstream's `scripts/compute_mean_activations.py`. The anchor is
the typical magnitude of activations on the layer we're steering; the upstream
blog finds that the steering sweet spot lives at roughly half of that.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path(".cache/persona/calibration")
DEFAULT_CHAT_TEMPLATE = True


def _cache_path(layer: int, llm_name: str, n_prompts: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = llm_name.replace("/", "__")
    return CACHE_DIR / f"{safe}_layer{layer}_n{n_prompts}.json"


def mean_residual_norm(
    llm,
    layer: int,
    prompts: list[str],
    use_cache: bool = True,
    llm_name: str | None = None,
) -> float:
    """Mean L2 norm of the residual stream at `layer` over all tokens of all prompts.

    Uses the existing nnsight LanguageModel `llm` from upstream. Caches the
    result to disk by (model_name, layer, n_prompts).
    """
    n = len(prompts)
    name = llm_name or getattr(llm, "_model_name", None) or getattr(llm, "model_name", "unknown")

    if use_cache:
        cache = _cache_path(layer, name, n)
        if cache.exists():
            data = json.loads(cache.read_text())
            print(f"(cached) mean_residual_norm layer={layer} n={n}: {data['mean_norm']:.3f}")
            return float(data["mean_norm"])

    print(f"Calibrating mean residual norm at layer {layer} over {n} prompts...")
    start = time.time()
    norms_per_prompt: list[float] = []

    apply_chat = bool(getattr(getattr(llm, "tokenizer", None), "chat_template", None))

    for i, prompt in enumerate(prompts):
        if apply_chat:
            ids = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            ids = llm.tokenizer.encode(prompt, return_tensors="pt")

        with llm.trace(ids):
            residual = llm.model.layers[layer].output
            norms = torch.norm(residual, dim=-1).save()

        n_arr = norms.cpu().detach().numpy().squeeze()
        # Drop the BOS token's norm (always huge, distorts the mean) per upstream pattern.
        if n_arr.ndim == 1 and n_arr.shape[0] > 1:
            n_arr = n_arr[1:]
        norms_per_prompt.append(float(n_arr.mean()))

        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - start
            print(f"  {i + 1}/{n} prompts, running mean norm = {np.mean(norms_per_prompt):.3f} "
                  f"({elapsed:.1f}s)")

    mean_norm = float(np.mean(norms_per_prompt))
    print(f"Mean residual norm at layer {layer}: {mean_norm:.3f}")

    if use_cache:
        cache = _cache_path(layer, name, n)
        cache.write_text(json.dumps({
            "mean_norm": mean_norm,
            "layer": layer,
            "n_prompts": n,
            "model": name,
        }, indent=2))

    return mean_norm
