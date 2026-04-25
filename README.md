# The Eiffel Tower Llama 

## **Reproducing the Golden Gate Claude demo with open-source models.**

This projects tentatively mimics the [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) experiment with open-source models, and establish a methodology for finding the optimal steering procedure.

We steered **Llama 3.1 8B Instruct** to become obsessed with the Eiffel Tower using **sparse autoencoders (SAEs)**.

<div align="center">
    <img src="banner.png" alt="A Llama photobombing the Eiffel Tower" width="640"/>
</div>

For more information, see the corresponding **[blog post](https://huggingface.co/dlouapre/spaces/eiffel-tower-llama)** on 🤗Hugging Face Spaces.

You can also **[try the demo](https://huggingface.co/dlouapre/spaces/eiffel-tower-llama-demo)**.

## Overview

This repository contains code for:
- Steering LLMs using SAE features from [Neuronpedia](https://www.neuronpedia.org) using `nnsight`;
- Systematic evaluation with LLM-as-a-judge metrics (concept inclusion, fluency, instruction following), calling GPT-OSS on 🤗Inference Providers.
- Bayesian optimization for multi-layer steering coefficients
- Analysis comparing SAE steering to prompt engineering

## Key Findings

- **The steering sweet spot is narrow** - optimal strength is ~0.5x the typical activation magnitude but ensuring concept inclusion without harming fluency and instruction following is tricky.
- **Clamping works better than adding** - clamping activations leads to better results than additive steering. This is consistent with what has been reported for the Golden Gate Claude demo, but contrasts with AxBench findings.
- **More features is not necessarily better** - steering multiple features only yields marginal improvements, suggesting those features are redundant rather than complementary.
- **Prompting still wins** - all in all, careful prompt engineering still outperforms SAE steering in this setup.

## Quick Start

```bash
# Install dependencies
uv sync

# Run 1D parameter sweep, configure in scripts/sweep_1D/sweep_1D.yaml
uv run scripts/sweep_1D/sweep_1D.py

# Evaluate steered model, configure in scripts/evaluation/evaluation.yaml
uv run scripts/evaluation/evaluation.py

# Optimize multi-layer steering, configure in scripts/optimize/optimize_botorch.yaml
uv run scripts/optimize/optimize_botorch.py
```

For using LLM-judge, do not forget to set up your Inference Providers API keys for GPT-OSS calls in `steering.py` or rely on local models.

## Structure

```
src/
  steering.py       # Core steering functions with nnsight
  optimize.py       # Bayesian optimization utilities
scripts/
  sweep_1D/         # Parameter sweep experiments
  evaluation/       # Model evaluation with LLM judges
  optimize/         # Multi-layer optimization
data/
  alpaca_eval.json  # Evaluation prompts from Alpaca
```

## Persona Steering (custom personas, any description)

This fork adds a `persona/` package that lets a non-ML user describe **any
persona in plain English** and get a steered Llama 3.1 8B Instruct, no Python
required after `pip install`. The pipeline:

1. Search Neuronpedia for SAE features whose published explanations match
   the persona description.
2. User picks 1–N features from a top-10 list (default: top-3 pre-checked).
3. Auto-sweep coefficients per feature with an LLM judge scoring each
   generation on three rubrics (persona fit / coherence / instruction
   following), then a joint validation pass at the per-feature optima.
4. Save the result as a portable `persona.yaml` and chat with the steered
   model in a REPL.

### Install

```bash
uv sync
uv pip install -e .
# optional: better feature ranking
uv pip install -e .[persona-rerank]
```

### A persona = steering + system prompt

SAE steering shapes **voice and obsession** (the model *thinks* about
McDonald's all the time). The system prompt carries **facts and role** (this
McDonald's-obsessed model is also a *helpful assistant* answering the user's
real question, never recommends Burger King, etc.). You almost always want
both — that's why `persona.yaml` has slots for both, and the CLI prompts you
to fill them in.

### Quick start (one persona, end to end)

```bash
persona new "obsessive McDonald's superfan" --out personas/mcd.yaml \
  --system-prompt "You are a die-hard McDonald's superfan. You always answer
  the user's actual question, but you find a way to relate everything back
  to McDonald's menu items. Never recommend a competitor."
persona find-features --persona personas/mcd.yaml             # interactive picker
persona sweep --persona personas/mcd.yaml                     # ~10 min on a single H100
persona chat --persona personas/mcd.yaml                      # talk to the steered model
```

Forgot the `--system-prompt` flag, or want to tweak it later? Use:

```bash
persona set-sysprompt --persona personas/mcd.yaml --text "You are ..."
# or load from a file
persona set-sysprompt --persona personas/mcd.yaml --from-file prompts/mcd.txt
# or remove it
persona set-sysprompt --persona personas/mcd.yaml --clear
```

You can also live-tweak inside the chat REPL with `/sysprompt <text>`.

### Commercial example (multi-feature + system prompt for facts)

```bash
persona new "warm and professional Acme Bank customer service rep" \
  --out personas/acme.yaml \
  --system-prompt "$(cat <<'EOF'
You are an assistant for Acme Bank.
Our hours are 9am to 5pm Monday through Friday.
You may not provide investment advice.
For compliance questions, refer customers to legal@acmebank.com.
EOF
)"
persona find-features --persona personas/acme.yaml
persona sweep --persona personas/acme.yaml --joint
persona chat --persona personas/acme.yaml
```

See [examples/acme_bank.yaml](examples/acme_bank.yaml) for a fully populated
multi-feature commercial persona.

### `persona.yaml` schema

```yaml
description: "Acme Bank customer service rep"
sae:
  release: llama_scope_lxr_8x
  sae_id: l15r_8x
  layer: 15
  neuronpedia_model: llama3.1-8b
  neuronpedia_source: 15-llamascope-res-32k
features:
  - feature_id: 12345
    explanation: "polite customer-service language"
    coefficient: 3.2
  - feature_id: 67890
    explanation: "personal banking and accounts"
    coefficient: 2.1
system_prompt: |
  You are an assistant for Acme Bank. Our hours are 9-5 Mon-Fri.
  You may not provide investment advice. Refer compliance to legal@acmebank.com.
generation:
  temperature: 0.7
  max_new_tokens: 256
llm_name: meta-llama/Llama-3.1-8B-Instruct
```

A `persona.yaml` is the **commercial unit of distribution** — one file = one
shippable persona. Version-control it, share it, drop it into product code.

### CLI reference

| Command | Purpose |
|---|---|
| `persona new "<desc>" --out <path>` | Create a blank persona.yaml from a description. Optional `--system-prompt`. |
| `persona find-features --persona <path>` | Search Neuronpedia, interactively pick 1–N features. `--noninteractive` auto-picks top 3. |
| `persona set-sysprompt --persona <path> --text "..."` | Set / replace the system prompt on an existing persona. Also `--from-file <path>` and `--clear`. |
| `persona sweep --persona <path>` | Per-feature linear sweep + joint validation. Writes coefficients back to yaml + `logs/<ts>_<slug>/report.md`. Use `--joint` for an extra 3^N grid search around per-feature optima (capped at N≤3). |
| `persona chat --persona <path>` | Interactive REPL. Slash commands: `/reset`, `/coeff <fid> <val>`, `/sysprompt <text>`, `/save <path>`, `/show`. |
| `persona save --feature-ids 1,2 --coefficients 3.0,2.5 -d "..." -o <path>` | Skip find-features/sweep; build a yaml from inline args. |

### What SAE steering can and can't do for commercial personas

**Yes** (style + topical obsession):
- "speaks like a pirate" — strong (sea-words feature)
- "obsessed with McDonald's" — strong (fast-food feature)
- "polite customer-service tone" — strong (politeness/hedging features)
- "Steve Jobs-style minimalism + product-launch language" — moderate

**No** (specific facts, brand identity, exact phrasing):
- "knows Acme Bank's hours are 9-5" — must come from the **system prompt**
- "never recommends competitors" — system prompt + RAG, not SAE
- "always uses the phrase 'Welcome to Acme'" — system prompt, not SAE
- "remembers what the user told them three turns ago" — chat history, not SAE

The right pattern for commercial deployments is **SAE steering for voice +
system prompt for facts + RAG for live data**. The `system_prompt` field on
`PersonaSpec` is there exactly for this.

### Caveats

- **SAE source change.** Upstream's Eiffel Tower demo used the
  `andyrdt/saes-llama-3.1-8b-instruct` SAEs, which have no published feature
  explanations on Neuronpedia. The persona toolkit instead uses **Llama-Scope**
  (`fnlp/Llama3_1-8B-Base-LXR-8x`, layer 15, residual stream, 32k features) which
  *does* have explanations searchable by the Neuronpedia API. The original
  Eiffel Tower scripts in `scripts/` still work against `andyrdt`; only the new
  `persona/` package switches sources.
- **Llama-Scope was trained on the *Base* model**, not Instruct. Steering on
  Instruct still works (the residual stream geometry is preserved across
  fine-tuning) but persona signals are slightly weaker than they would be on
  Base. If you care about absolute steering quality more than instruction
  following, swap `llm_name: meta-llama/Llama-3.1-8B` in the yaml.
- **No GGUF / Ollama / llama.cpp export is possible.** SAE steering hooks live
  inside the PyTorch forward pass via `nnsight`. They don't translate to
  llama.cpp's quantised graph. For deployment, ship the model + persona.yaml +
  the `persona chat` runtime (or wrap `persona.chat.run_chat` in your own HTTP
  server).
- **Neuronpedia rate limits.** The `/api/explanation/search` endpoint allows
  ~200 requests/hour for unauthenticated traffic. Persona search results are
  cached on disk under `.cache/persona/search/` for 7 days; re-running with the
  same description is free. Set `NEURONPEDIA_API_KEY` to bypass the limit.
- **Joint optimisation above 3 features** falls back to upstream's
  `noisy_blackbox_optimization` (in [src/optimize.py](src/optimize.py)) for
  proper Bayesian optimisation across many dimensions. The built-in `--joint`
  grid search is hard-capped at N≤3 to keep costs sane.

## Citation

If you use this work, please cite:
```
Louapre, D. (2025). The Eiffel Tower Llama: Reproducing the Golden Gate Claude
experiment with open-source models.
```

## Related Work

- [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) - Original Anthropic demo
- [Neuronpedia](https://www.neuronpedia.org) - SAE exploration platform
- [AxBench](https://arxiv.org/abs/2501.17148) - Steering benchmark paper
