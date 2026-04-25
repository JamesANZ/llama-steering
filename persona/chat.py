"""Interactive REPL with the steered model.

Loads a persona.yaml (features + per-feature coefficients + optional system
prompt), then loops on user input. Supports a few slash commands:

  /reset               clear conversation history
  /coeff <fid> <val>   live-tweak one feature's coefficient
  /sysprompt <text>    set or replace the system prompt
  /save <path>         persist current persona state to a yaml
  /show                print current persona state
  /help                list commands
  /quit, /exit         exit
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from ._upstream import generate_steered_answer
from .config import FeatureSpec, PersonaSpec
from .sae_loader import load_feature_vectors, pick_device, update_strengths

console = Console()

HELP = """\
Commands:
  /reset                       clear chat history
  /coeff <feature_id> <value>  set one feature's coefficient
  /sysprompt <text>            set or replace system prompt (use empty arg to clear)
  /save <path>                 save current persona to yaml
  /show                        print current persona state
  /help                        this message
  /quit, /exit                 exit
Anything else is sent to the model."""


def _print_state(persona: PersonaSpec) -> None:
    console.print(f"[bold]description:[/bold] {persona.description}")
    console.print(f"[bold]layer:[/bold] {persona.sae.layer}  "
                  f"[bold]sae:[/bold] {persona.sae.release}/{persona.sae.sae_id}")
    if persona.system_prompt:
        sp = persona.system_prompt
        if len(sp) > 200:
            sp = sp[:200] + "..."
        console.print(f"[bold]system_prompt:[/bold] {sp}")
    else:
        console.print("[bold]system_prompt:[/bold] (none)")
    for f in persona.features:
        console.print(f"  feature {f.feature_id}: coeff={f.coefficient}  expl={f.explanation[:60]}")


def run_chat(
    persona_path: str | Path,
    coefficient_scale: float = 1.0,
    llm: Any | None = None,
) -> None:
    persona = PersonaSpec.from_yaml(persona_path)
    if not persona.features:
        raise ValueError(f"{persona_path} has no features. Run find-features first.")
    if not persona.has_coefficients():
        console.print(
            "[yellow]Warning:[/yellow] persona has features but no coefficients. "
            "Run `persona sweep --persona <yaml>` first, or set per-feature "
            "coefficients manually with `/coeff`."
        )

    device = pick_device()
    console.print(f"Device: {device}")

    if llm is None:
        from nnsight import LanguageModel  # heavy import; lazy
        console.print(f"Loading model {persona.llm_name}...")
        llm = LanguageModel(persona.llm_name, dispatch=True).to(device)
        try:
            llm._model_name = persona.llm_name
        except Exception:
            pass

    components = load_feature_vectors(
        persona.sae,
        feature_ids=persona.feature_ids(),
        device=device,
        coefficients=[(f.coefficient or 0.0) * coefficient_scale for f in persona.features],
    )

    history: list[dict] = []
    if persona.system_prompt:
        history.append({"role": "system", "content": persona.system_prompt})

    _print_state(persona)
    console.print(f"[dim]coefficient_scale = {coefficient_scale}[/dim]")
    console.print("[dim]/help for commands. Ctrl-C to exit.[/dim]")

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, *rest = user_input.split(maxsplit=1)
            arg = rest[0] if rest else ""

            if cmd in ("/quit", "/exit"):
                return
            if cmd == "/help":
                console.print(HELP)
                continue
            if cmd == "/reset":
                history = []
                if persona.system_prompt:
                    history.append({"role": "system", "content": persona.system_prompt})
                console.print("[dim]history cleared[/dim]")
                continue
            if cmd == "/show":
                _print_state(persona)
                continue
            if cmd == "/sysprompt":
                persona.system_prompt = arg or None
                history = []
                if persona.system_prompt:
                    history.append({"role": "system", "content": persona.system_prompt})
                console.print(f"[dim]system_prompt set ({len(arg)} chars), history cleared[/dim]")
                continue
            if cmd == "/save":
                if not arg:
                    console.print("[red]usage: /save <path>[/red]")
                    continue
                persona.to_yaml(arg)
                console.print(f"[green]saved to {arg}[/green]")
                continue
            if cmd == "/coeff":
                parts = arg.split()
                if len(parts) != 2:
                    console.print("[red]usage: /coeff <feature_id> <value>[/red]")
                    continue
                try:
                    fid = int(parts[0])
                    val = float(parts[1])
                except ValueError:
                    console.print("[red]invalid feature_id or value[/red]")
                    continue
                idx = next((i for i, f in enumerate(persona.features) if f.feature_id == fid), None)
                if idx is None:
                    console.print(f"[red]feature {fid} not in this persona[/red]")
                    continue
                persona.features[idx] = FeatureSpec(
                    feature_id=fid,
                    explanation=persona.features[idx].explanation,
                    coefficient=val,
                )
                strengths = [(f.coefficient or 0.0) * coefficient_scale for f in persona.features]
                components = update_strengths(components, strengths)
                console.print(f"[dim]feature {fid} coefficient -> {val}[/dim]")
                continue
            console.print(f"[red]unknown command: {cmd}[/red]")
            continue

        history.append({"role": "user", "content": user_input})
        torch.manual_seed(0)
        try:
            out = generate_steered_answer(
                llm,
                history,
                components,
                max_new_tokens=persona.generation.max_new_tokens,
                temperature=persona.generation.temperature,
                repetition_penalty=persona.generation.repetition_penalty,
                steer_prompt=persona.generation.steer_prompt,
                clamp_intensity=persona.generation.clamp_intensity,
            )
            answer = out["answer"]
        except Exception as e:
            console.print(f"[red]generation failed: {e}[/red]")
            history.pop()  # don't keep the user msg in a broken state
            continue
        console.print(f"\n[bold cyan]model>[/bold cyan] {answer}")
        history.append({"role": "assistant", "content": answer})
