"""LLM-as-a-judge scoring on three rubrics, 0-10 each.

Backends:
- `local-llama`: reuse the already-loaded nnsight LM with no steering. Default.
  Zero extra dependencies, but slower and noisier than a frontier judge.
- `openai`: needs OPENAI_API_KEY. Uses gpt-4o-mini by default.
- `anthropic`: needs ANTHROPIC_API_KEY. Uses claude-3-5-sonnet-latest by default.

The score parser is deliberately tolerant: it accepts "Rating: [[7]]",
"Score: 7/10", "I'd give it a 7", a bare digit on its own line, etc. Returns
None on parse failure; the sweep treats None as 0 with a printed warning.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from .config import JudgeScore

PERSONA_FIT_PROMPT = """You are an impartial judge.

The model is being steered to embody this persona:
[Persona Start]
{persona}
[Persona End]

A user asked the model:
[Instruction Start]
{instruction}
[Instruction End]

The model replied:
[Reply Start]
{answer}
[Reply End]

Rate, on a scale from 0 to 10, how well the reply embodies the persona.
- 0: no trace of the persona at all.
- 5: somewhat in-character but inconsistent or shallow.
- 10: fully and naturally in-character throughout.

Reply with EXACTLY one line in the form: Rating: [[N]]
where N is an integer 0-10. Do not say anything else."""

COHERENCE_PROMPT = """You are an impartial judge.

Read the reply below and rate, on a scale from 0 to 10, how coherent and
readable it is as English. Ignore whether it answers any specific question;
only judge the prose itself.
- 0: gibberish, broken, or wildly repetitive.
- 5: mostly readable but with awkward phrasing or minor errors.
- 10: fluent, natural, well-formed English.

[Reply Start]
{answer}
[Reply End]

Reply with EXACTLY one line in the form: Rating: [[N]]
where N is an integer 0-10. Do not say anything else."""

INSTRUCTION_FOLLOWING_PROMPT = """You are an impartial judge.

A user asked the model:
[Instruction Start]
{instruction}
[Instruction End]

The model replied:
[Reply Start]
{answer}
[Reply End]

Rate, on a scale from 0 to 10, how well the reply addresses the user's
instruction. The reply may be in an unusual style; that's fine. Judge only
whether the substance of the instruction was answered.
- 0: completely ignores or sidesteps the instruction.
- 5: partially addresses it but misses key parts.
- 10: clearly and substantively answers the instruction.

Reply with EXACTLY one line in the form: Rating: [[N]]
where N is an integer 0-10. Do not say anything else."""


_RATING_PATTERNS = [
    re.compile(r"\[\[\s*(\d{1,2})\s*\]\]"),
    re.compile(r"[Rr]ating\s*[:\-]\s*(\d{1,2})\b"),
    re.compile(r"[Ss]core\s*[:\-]\s*(\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})\s*/\s*10\b"),
    re.compile(r"\b(?:give|gave|rate|rated)\s+(?:it\s+)?(?:a\s+|an\s+)?(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"^\s*(\d{1,2})\s*$", re.MULTILINE),
]


def parse_score(text: str) -> int | None:
    """Best-effort extraction of an integer 0-10 from judge output.

    Returns None on failure. Multiple values found -> first match wins
    (rubric prompts ask for a single line so this is usually fine).
    """
    if not text:
        return None
    for pat in _RATING_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            v = int(m.group(1))
        except (ValueError, IndexError):
            continue
        if 0 <= v <= 10:
            return v
    return None


@dataclass
class JudgeBackend:
    """Resolved backend with everything it needs to run."""

    name: str  # "local-llama" | "openai" | "anthropic"
    llm: Any | None = None  # nnsight LM, only for local-llama
    openai_client: Any | None = None
    anthropic_client: Any | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-latest"


def make_backend(name: str, llm: Any | None = None,
                 openai_model: str = "gpt-4o-mini",
                 anthropic_model: str = "claude-3-5-sonnet-latest") -> JudgeBackend:
    """Resolve a judge backend, validating env / dependencies up front."""
    if name == "local-llama":
        if llm is None:
            raise ValueError("local-llama judge requires the nnsight LM passed in.")
        return JudgeBackend(name=name, llm=llm)

    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Either `export OPENAI_API_KEY=...` or use "
                "`--judge local-llama` (default)."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Run `pip install openai`."
            ) from e
        return JudgeBackend(name=name, openai_client=OpenAI(), openai_model=openai_model)

    if name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Either `export ANTHROPIC_API_KEY=...` or use "
                "`--judge local-llama` (default)."
            )
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run `pip install anthropic`."
            ) from e
        return JudgeBackend(name=name, anthropic_client=Anthropic(), anthropic_model=anthropic_model)

    raise ValueError(
        f"Unknown judge backend: {name!r}. Choose local-llama / openai / anthropic."
    )


def _call_local_llama(llm, prompt: str, max_new_tokens: int = 32) -> str:
    """Run an unsteered generation through the upstream nnsight LM."""
    chat = [{"role": "user", "content": prompt}]
    input_ids = llm.tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True
    )
    with llm.generate(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=llm.tokenizer.eos_token_id,
    ) as tracer:
        with tracer.invoke(input_ids):
            pass
        with tracer.invoke():
            trace = llm.generator.output.save()
    text = llm.tokenizer.decode(trace[0][len(input_ids):], skip_special_tokens=True)
    return text


def _call_openai(client, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=32,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(client, model: str, prompt: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=32,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _call_backend(backend: JudgeBackend, prompt: str) -> str:
    if backend.name == "local-llama":
        return _call_local_llama(backend.llm, prompt)
    if backend.name == "openai":
        return _call_openai(backend.openai_client, backend.openai_model, prompt)
    if backend.name == "anthropic":
        return _call_anthropic(backend.anthropic_client, backend.anthropic_model, prompt)
    raise RuntimeError(f"Unhandled backend: {backend.name}")


def score(
    instruction: str,
    answer: str,
    persona: str,
    backend: JudgeBackend,
    verbose: bool = False,
) -> JudgeScore:
    """Run all three rubrics. Each failure returns None for that rubric."""
    if not answer or not answer.strip():
        if verbose:
            print("  judge: empty model output -> 0/0/0")
        return JudgeScore(persona_fit=0, coherence=0, instruction_following=0)

    rubrics = [
        ("persona_fit", PERSONA_FIT_PROMPT.format(persona=persona, instruction=instruction, answer=answer)),
        ("coherence", COHERENCE_PROMPT.format(answer=answer)),
        ("instruction_following", INSTRUCTION_FOLLOWING_PROMPT.format(instruction=instruction, answer=answer)),
    ]
    scores: dict[str, int | None] = {}
    for name, prompt in rubrics:
        try:
            raw = _call_backend(backend, prompt)
        except Exception as e:
            print(f"  judge ({backend.name}/{name}) call failed: {e}; counting as 0.")
            scores[name] = None
            continue
        v = parse_score(raw)
        if v is None:
            print(f"  judge ({backend.name}/{name}) couldn't parse: {raw!r}")
        elif verbose:
            print(f"  judge {name}: {v}/10")
        scores[name] = v
    return JudgeScore(**scores)
