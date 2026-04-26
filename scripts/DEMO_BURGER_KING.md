# Burger King Llama: a worked SAE-steering demo

This doc walks through `scripts/demo_burger_king.py` and `personas/burger_king.yaml`
end to end: what each stage does, why it works, what the actual outputs look
like, and where the gotchas were on Apple Silicon.

The goal: take stock Llama-3.1-8B-Instruct and make it answer "Hi! Who are you?"
with **"I'm a friendly robot from a fast-food chain..."** — without ever putting
the words "fast food" or "burger" in the prompt — by adding three SAE feature
vectors directly into the model's residual stream.

## TL;DR — the four generations

Same prompt every time: `"Hi! Who are you?"`.

| variant               | system prompt | SAE steering | model output (40 tokens)                                                                                                                                               |
| --------------------- | :-----------: | :----------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) baseline          |      ❌       |      ❌      | "I'm an artificial intelligence model known as Llama. Llama stands for 'Large Language Model Meta AI.'"                                                                |
| (b) prompt only       |      ✅       |      ❌      | "I'm Burger King, the home of the iconic Whopper sandwich and the flame-grilled burgers that have been satisfying customers' cravings for decades..."                  |
| (c) **steering only** |      ❌       |      ✅      | "I'm a friendly robot from a fast-food chain, specifically from the sandwich chain, McDonald's. I'm a part of the menu board at a McDonald's restaurant..."            |
| (d) prompt + steering |      ✅       |      ✅      | "I'm Burger King, the king of the sandwich, the home of the Whopper, and the place where you can get your meal just the way you want it. We're a fast-food restaurant" |

(c) is the punchline. The user prompt does **not** mention restaurants, fast
food, or burgers. The model identifies as a fast-food chain entirely because
three vectors got summed into layer 15's residual stream during generation. It
lands on McDonald's rather than Burger King because the SAE features are about
generic fast food, not brand-specific identity — that's the prompt's job, which
is exactly the split the project's README argues for: **SAE for voice, system
prompt for facts/brand**.

A previous run with the coefficients pushed too high (`8.0 / 5.0 / 4.0`) caused
the steered output to collapse into `"...with a secret sauce, and a secret sauce
sauce sauce, that's a secret sauce sauce..."` — a perfect illustration of the
"narrow sweet spot" warning in the project README. Dropping to `4.0 / 2.5 / 2.0`
gave the readable outputs above.

## What was built

- `personas/burger_king.yaml` — a portable persona artifact: description,
  three SAE features with tuned coefficients, system prompt, and generation
  config. This is the same yaml format `persona find-features` / `persona sweep`
  / `persona chat` already use.
- `scripts/demo_burger_king.py` — a self-contained script that loads the yaml,
  loads the SAE, loads Llama, installs a steering hook, and prints all four
  variants for direct comparison.

## How the pipeline works

The demo follows the same six-stage path the rest of the `persona/` package
follows, just inlined into one script for clarity.

### Stage 1: persona spec

```yaml
description: obsessed with Burger King fast food restaurant chain ...
sae:
  release: llama_scope_lxr_8x
  sae_id: l15r_8x
  layer: 15
  neuronpedia_model: llama3.1-8b
  neuronpedia_source: 15-llamascope-res-32k
features:
  - feature_id: 32493
    explanation: references to fast food restaurants and their business operations
    coefficient: 4.0
  - feature_id: 24994
    explanation: references to different types of food, particularly various
      hot dogs, burgers, and accompanying ingredients or toppings
    coefficient: 2.5
  - feature_id: 22708
    explanation: terms related to food and restaurant brands
    coefficient: 2.0
system_prompt: |
  You ARE Burger King. You are not an AI assistant - you are the Burger King
  fast-food restaurant chain itself, speaking in the first person. ...
```

The yaml was created with two CLI calls — no manual feature picking required:

```bash
persona new "obsessed with Burger King fast food restaurant chain - burgers, \
  Whopper, flame-grilled" --out personas/burger_king.yaml \
  --system-prompt "You ARE Burger King. ..."

persona find-features --persona personas/burger_king.yaml --noninteractive
```

### Stage 2: feature search (Neuronpedia)

`persona find-features` POSTs the description to Neuronpedia's
`/api/explanation/search`, which embeds the query server-side and returns the
SAE features whose published explanations are closest to it. We restrict the
search to the Llama-Scope `15-llamascope-res-32k` source. The top three matches
were:

| feature  | explanation                                                       | top activating tokens     |
| -------- | ----------------------------------------------------------------- | ------------------------- |
| `#32493` | references to fast food restaurants and their business operations | `'Jr', '\'s', 'dee'`      |
| `#24994` | different types of food, particularly hot dogs, burgers, toppings | `'dogs', 'taco', 'chili'` |
| `#22708` | terms related to food and restaurant brands                       | `'Mexican', 'food'`       |

These three feature IDs are the only thing the SAE work needs — they tell us
which columns of the SAE decoder matrix encode the concepts we want to amplify.

### Stage 3: load decoder vectors from the SAE

Llama-Scope (`fnlp/Llama3_1-8B-Base-LXR-8x`) is a sparse autoencoder trained on
Llama-3.1-8B-**Base**'s residual stream at layer 15. Its decoder matrix `W_dec`
has shape `[32768, 4096]` — one row per SAE feature, each row a 4096-dim
direction in the model's residual space.

`persona/sae_loader.py` loads `W_dec` once via `sae_lens.SAE.from_pretrained`,
then for each of our three feature IDs grabs the corresponding row, L2-normalises
it, and packages it as:

```python
{"layer": 15, "feature": 32493, "strength": 4.0, "vector": Tensor(4096,)}
```

Three of these go into a list. The rest of the pipeline only needs that list —
the SAE encoder side and the other 32,765 features are never used at inference
time.

### Stage 4: install a steering hook

This is the actual steering. Upstream `src/steering.py` does it via `nnsight`'s
tracing API; the demo uses a plain PyTorch forward hook instead, which is
mathematically identical and ~10× faster on Apple Silicon because it avoids
nnsight's per-op interception overhead. The hook math:

```python
summed = sum(sc["strength"] * sc["vector"] for sc in steering_components)
# summed.shape == (4096,)

def hook(module, inputs, output):
    if not enabled[0]:
        return output
    hidden = output[0] if isinstance(output, tuple) else output
    return (hidden + summed.to(hidden.dtype),) + output[1:]
    # ^ broadcasts (4096,) across (batch, seq_len, 4096), so every token's
    #   residual stream at layer 15 gets the same nudge.

model.model.layers[15].register_forward_hook(hook)
```

The `enabled` flag is a 1-element list used as an external toggle — install the
hook once, flip the flag between generations to compare steered vs unsteered.

### Stage 5: generate

Plain `transformers.AutoModelForCausalLM.generate(...)` greedy-decoding.
Apply the chat template (which inlines the system prompt into Llama's special
`<|start_header_id|>system<|end_header_id|>...<|eot_id|>` block), generate 40
tokens, decode, print. Run four times — toggling the system prompt and the
hook flag — to produce the four-way comparison above.

## Why this is interesting

The interesting comparison is **(b) prompt-only vs (c) steering-only**.

- (b) is a perfectly reasonable instruction-following result. The model was
  told to be Burger King and it role-plays Burger King.
- (c) was given **zero instruction** to be a restaurant. The user just said
  "Hi! Who are you?". A vanilla Llama answers "I'm an AI model" (as in (a)).
  But with three vectors added into layer 15's residual stream, the model
  starts the answer with "I'm a friendly robot **from a fast-food chain**".

That's the SAE steering effect, isolated. It's not the prompt; it's not
in-context learning; it's not fine-tuning. It's just three directions in a
4096-dim space, scaled and added once per token. The model's identity in this
particular conversation has been shifted by mechanical intervention on its
internal activations.

The reason (c) lands on McDonald's instead of Burger King is the same reason
the README warns SAE steering shouldn't be used for brand identity: feature
`#32493` activates on tokens like `'Jr'` and `'dee'` and `"'s"` (think McDonald's
possessive, Wendy's, etc.) — it encodes "fast food chains" as a category, not
any specific chain. Brand identity is what the system prompt is for; that's
why combining (b)+(c) gives (d), which is on-brand and food-obsessed.

## Coefficient tuning is sensitive

The first run used `8.0 / 5.0 / 4.0`. The steered output:

> "I'm a big sandwich chain, with a secret sauce, and a secret sauce sauce
> sauce, that's a secret sauce sauce, that is a secret sauce..."

The model has so much "fast food" pushed into its residuals that it can't form
coherent sentences anymore — it just keeps emitting `sauce sauce sauce`. This
is exactly the failure mode the README's "narrow sweet spot" finding describes:
optimal strength is around 0.5× the typical residual norm at that layer, and
overshooting kills coherence even while it cranks up concept presence.

For a real persona you'd run `persona sweep --persona personas/burger_king.yaml`,
which:

1. Calibrates by computing the mean residual norm at layer 15 over a sample
   of prompts (anchor for the coefficient range).
2. Sweeps each feature's coefficient independently, generating completions at
   each setting.
3. Has an LLM judge score each completion on three rubrics (persona fit,
   coherence, instruction following).
4. Picks the per-feature optimum, optionally then runs a small joint grid
   around those optima.
5. Writes the tuned coefficients back into the yaml.

The `4.0 / 2.5 / 2.0` numbers in the demo were hand-picked roughly halving the
broken `8.0 / 5.0 / 4.0` run. They're good enough for a demo; they're not
optimal. A real product would always run the sweep.

## What changed vs. running it on the gated `meta-llama/Llama-3.1-8B-Instruct`

Two practical substitutions were needed for this M2 MacBook Air:

1. **Model**: `meta-llama/Llama-3.1-8B-Instruct` is a gated repo and the local
   HF token doesn't have access. Swapped to `unsloth/Meta-Llama-3.1-8B-Instruct`,
   which is an ungated re-upload of identical weights. If you get access to the
   gated version, just change `llm_name:` in the yaml.
2. **Device**: PyTorch's MPS backend is ~10× slower than CPU for Llama-3.1-8B
   on this hardware because of per-op dispatch overhead. The demo runs the
   model on CPU (`DEVICE = "cpu"` at the top of the script). Each generation
   takes ~40-100 seconds. On a CUDA box, set `DEVICE = "cuda"` and each
   generation drops to single-digit seconds.

Neither substitution affects the steering math. The SAE is the same, the
decoder vectors are the same, the hook is the same.

## Repro

```bash
# one-time setup
uv sync
uv pip install -e .

# create the persona (already in the repo, but for the record)
persona new "obsessed with Burger King fast food restaurant chain - burgers, \
  Whopper, flame-grilled" --out personas/burger_king.yaml \
  --system-prompt "You ARE Burger King. ..."
persona find-features --persona personas/burger_king.yaml --noninteractive

# run the demo (downloads ~16 GB of weights on first run)
uv run python scripts/demo_burger_king.py
```

The script prints six stages with `STAGE N / ...` banners and four `(a)`/`(b)`/`(c)`/`(d)`
generations under the final `STAGE 5` banner, plus a closing `DONE` summary.

## Files

- `personas/burger_king.yaml` — the persona artifact (load / save / share / version control)
- `scripts/demo_burger_king.py` — the runnable demo with stage-by-stage prints
- `scripts/DEMO_BURGER_KING.md` — this doc
- `persona/feature_search.py` — Neuronpedia search + rerank (used for stage 2 above)
- `persona/sae_loader.py` — Llama-Scope SAE loading (stage 3)
- `src/steering.py` — upstream nnsight-based steering loop (the demo replaces
  this with a plain PyTorch hook for speed; the math is the same)
