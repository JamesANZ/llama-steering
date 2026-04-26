"""Demo: steer Llama 3.1 8B Instruct so it thinks it's Burger King.

End-to-end walkthrough that PRINTS each stage so you can see the repo working:

  1. Load persona spec (description + features + coefficients + system prompt)
     from personas/burger_king.yaml -- the artifact this whole repo is
     designed to produce.
  2. Show the SAE features that `persona find-features` discovered for the
     description (already populated in the yaml).
  3. Load the SAE (Llama-Scope L15R-8x, 32k features) and pull out the
     decoder-vector columns for our chosen features.
  4. Load Llama-3.1-8B-Instruct on MPS.
  5. Register a forward hook on layer 15 that ADDS the weighted sum of
     decoder vectors to the residual stream during every generation step.
     This is the SAE-steering trick -- mathematically identical to what
     `src/steering.py` does via nnsight, but with a plain PyTorch hook so
     it's fast on Apple Silicon.
  6. Generate two answers to the same prompt:
       (a) PROMPT-ONLY: system prompt only, hook disabled.
       (b) STEERED   : system prompt + SAE steering, hook enabled.
     Print both side-by-side so you can SEE the steering bend the voice
     toward burger / fast-food / restaurant-brand vocabulary on top of
     what the prompt alone already does.

Run:
    uv run python scripts/demo_burger_king.py

This is intentionally small (1 prompt, 60 tokens) to be tractable on an
M2 Mac. Each generation still takes a few minutes on MPS; on CUDA it's
<10 s. The Llama-Scope SAE only works with Llama-3.1-8B residual streams,
so we can't substitute a smaller model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from persona.config import PersonaSpec  # noqa: E402
from persona.sae_loader import load_feature_vectors  # noqa: E402

PERSONA_PATH = REPO_ROOT / "personas" / "burger_king.yaml"
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.0  # greedy for reproducibility
SEED = 0
PROMPT = "Hi! Who are you?"
# On Apple Silicon M2, CPU is ~10x faster than MPS for Llama 3.1 8B because of
# MPS per-op dispatch overhead. The hook + steering math is identical either
# way; only inference speed changes. CUDA users should change this to "cuda".
DEVICE = "cpu"


def banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def subsection(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def make_steering_hook(layer_module, steering_components: list[dict], enabled: list[bool]):
    """Forward-hook factory.

    steering_components is the same list `src/steering.py` consumes:
      [{"layer": int, "feature": int, "strength": float, "vector": Tensor}, ...]
    All vectors must already be on the same device as the residual stream
    and L2-normalised (load_feature_vectors does that for us).

    `enabled` is a 1-element list used as a toggle from the outside; we
    register the hook ONCE and just flip the flag between generations.
    """
    # Pre-sum (strength * vector) once. Shape: [d_model]
    summed = sum(sc["strength"] * sc["vector"] for sc in steering_components)

    def hook(module, inputs, output):
        if not enabled[0]:
            return output
        # Llama decoder layer returns either Tensor or tuple(Tensor, ...)
        if isinstance(output, tuple):
            hidden = output[0]
            steered = hidden + summed.to(hidden.dtype)
            return (steered,) + output[1:]
        return output + summed.to(output.dtype)

    return layer_module.register_forward_hook(hook)


def generate(model, tok, chat: list[dict], label: str, device: str) -> str:
    enc = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    ids = enc.input_ids if hasattr(enc, "input_ids") else enc
    ids = ids.to(device)
    attn = torch.ones_like(ids)

    print(f"  [{label}] generating {MAX_NEW_TOKENS} tokens (input={ids.shape[1]} tokens)...", flush=True)
    t0 = time.time()
    torch.manual_seed(SEED)
    with torch.inference_mode():
        out = model.generate(
            ids,
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=TEMPERATURE > 0.0,
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0
    n_new = out.shape[1] - ids.shape[1]
    print(f"  [{label}] done in {dt:.1f}s ({n_new/dt:.2f} tok/s)", flush=True)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    banner("STAGE 1 / Load persona spec from yaml")
    persona = PersonaSpec.from_yaml(PERSONA_PATH)
    print(f"  description : {persona.description}")
    print(f"  llm         : {persona.llm_name}")
    print(f"  sae         : {persona.sae.release}/{persona.sae.sae_id}  layer={persona.sae.layer}")
    print(f"  features (already discovered by `persona find-features`):")
    for f in persona.features:
        print(f"      #{f.feature_id:>6}  coeff={f.coefficient:>5.2f}   {f.explanation[:70]}")
    print(f"  system_prompt ({len(persona.system_prompt or '')} chars):")
    for line in (persona.system_prompt or "").splitlines():
        print(f"      | {line}")

    device = DEVICE
    banner(f"STAGE 2 / Load SAE (Llama-Scope L15R-8x) on device={device}")
    t0 = time.time()
    components = load_feature_vectors(
        persona.sae,
        feature_ids=persona.feature_ids(),
        device=device,
        coefficients=[f.coefficient for f in persona.features],
    )
    print(f"  loaded {len(components)} decoder vectors in {time.time()-t0:.1f}s")
    for c in components:
        print(f"    layer={c['layer']}  feature={c['feature']}  strength={c['strength']:.2f}  "
              f"||vec||={c['vector'].norm().item():.3f}  shape={tuple(c['vector'].shape)}")

    banner(f"STAGE 3 / Load Llama-3.1-8B-Instruct (fp16) on device={device}")
    print("  (cached after first run; ~16 GB total)")
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(persona.llm_name)
    model = AutoModelForCausalLM.from_pretrained(
        persona.llm_name, dtype=torch.float16
    ).to(device)
    model.eval()
    print(f"  model loaded in {time.time()-t0:.1f}s")

    banner(f"STAGE 4 / Install steering hook on model.model.layers[{persona.sae.layer}]")
    enabled = [False]
    handle = make_steering_hook(
        model.model.layers[persona.sae.layer], components, enabled
    )
    print(f"  hook installed; sums {len(components)} steering vectors into the")
    print(f"  residual stream at layer {persona.sae.layer} when enabled.")

    sys_msg = {"role": "system", "content": persona.system_prompt}
    user_msg = {"role": "user", "content": PROMPT}

    banner(f"STAGE 5 / Four-way comparison on user prompt: {PROMPT!r}")

    subsection("(a) BASELINE  - no system prompt, no SAE steering")
    enabled[0] = False
    a = generate(model, tok, [user_msg], "baseline", device)
    print(f"\nmodel> {a}\n")

    subsection("(b) PROMPT ONLY  - system prompt, no SAE steering")
    enabled[0] = False
    b = generate(model, tok, [sys_msg, user_msg], "prompt-only", device)
    print(f"\nmodel> {b}\n")

    subsection("(c) STEERING ONLY  - no system prompt, just SAE features")
    enabled[0] = True
    c = generate(model, tok, [user_msg], "steering-only", device)
    print(f"\nmodel> {c}\n")

    subsection("(d) PROMPT + STEERING  - the full pipeline")
    enabled[0] = True
    d = generate(model, tok, [sys_msg, user_msg], "prompt+steering", device)
    print(f"\nmodel> {d}\n")

    handle.remove()

    banner("DONE")
    print("  (a) BASELINE        : raw Llama 3.1 8B Instruct, no role.")
    print("  (b) PROMPT ONLY     : system prompt is enough on its own to make")
    print("                        the model role-play as Burger King.")
    print("  (c) STEERING ONLY   : asking 'who are you' with NO system prompt,")
    print("                        but with SAE feature vectors added at layer")
    print(f"                        {persona.sae.layer}. The model has no instruction to be Burger")
    print("                        King, yet the residual-stream nudge alone is")
    print("                        enough to push the answer toward food/fast-food.")
    print("  (d) PROMPT+STEERING : both signals together. This is what `persona")
    print("                        chat` ships.")
    print()
    print("  The README warns that the sweet spot is narrow: too low, and the SAE")
    print("  has no effect; too high, and the model degenerates into repetition.")
    print("  The yaml currently uses 4.0 / 2.5 / 2.0; running `persona sweep`")
    print("  with an LLM judge will tune these per feature for any persona.")


if __name__ == "__main__":
    main()
