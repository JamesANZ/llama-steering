"""Typer CLI: `persona new / find-features / sweep / chat / save`.

Stays thin: each command parses args and delegates to the module that does
the work. All commands also accept inline flags for power users who don't
want a yaml file.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import (
    DEFAULT_LLM_NAME,
    FeatureSpec,
    GenerationConfig,
    PersonaSpec,
    SAEConfig,
    SweepConfig,
    blank_persona,
    load_persona,
)

app = typer.Typer(
    add_completion=False,
    help="Persona steering for Llama 3.1 8B Instruct (SAE feature steering on Llama-Scope).",
)
console = Console()


@app.command()
def new(
    description: str = typer.Argument(..., help='Persona description, e.g. "obsessive McDonald\'s superfan".'),
    out: Path = typer.Option(..., "--out", "-o", help="Path to write the new persona.yaml to."),
    system_prompt: str | None = typer.Option(None, "--system-prompt", "-s", help="Optional system prompt."),
    llm_name: str = typer.Option(DEFAULT_LLM_NAME, "--llm", help="HF model id."),
):
    """Create a blank persona.yaml with just a description (and optional system prompt)."""
    persona = blank_persona(description)
    persona.system_prompt = system_prompt
    persona.llm_name = llm_name
    if out.exists():
        if not typer.confirm(f"{out} exists, overwrite?", default=False):
            raise typer.Abort()
    persona.to_yaml(out)
    console.print(f"[green]Created[/green] {out}")
    if not persona.system_prompt:
        console.print(
            "[dim]Tip: a persona is steering + system prompt. Add a role/facts prompt with[/dim]\n"
            f"  [dim]persona set-sysprompt --persona {out} --text \"You are ...\"[/dim]"
        )
    console.print("Next: `persona find-features --persona %s`" % out)


@app.command("find-features")
def find_features_cmd(
    persona_path: Path = typer.Option(..., "--persona", "-p", help="Path to persona.yaml."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="How many candidates to fetch."),
    default_n: int = typer.Option(3, "--default-n", "-n", help="How many to pre-check in the picker."),
    noninteractive: bool = typer.Option(False, "--noninteractive", help="Skip the interactive picker; auto-pick top N."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip the on-disk Neuronpedia cache."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Use the server's order; skip local rerank."),
):
    """Search Neuronpedia for features matching the persona description and pick 1-N."""
    from .feature_search import search_features
    from .picker import pick_features

    persona = load_persona(persona_path)
    matches = search_features(
        query=persona.description,
        sae_config=persona.sae,
        top_k=top_k,
        use_cache=not no_cache,
        rerank=not no_rerank,
    )
    if not matches:
        console.print("[red]No matches. Try a more descriptive persona text.[/red]")
        raise typer.Exit(code=1)

    picked = pick_features(matches, default_n=default_n, noninteractive=noninteractive)
    if not picked:
        console.print("[red]No features picked. Persona unchanged.[/red]")
        raise typer.Exit(code=1)

    persona.features = [
        FeatureSpec(feature_id=m.feature_id, explanation=m.explanation, coefficient=None)
        for m in picked
    ]
    persona.to_yaml(persona_path)
    console.print(f"[green]Wrote {len(picked)} features to {persona_path}[/green]")
    if not persona.system_prompt:
        console.print(
            "[dim]Tip: SAE steering shapes voice; a system prompt carries facts and "
            "role. Set one with[/dim]\n"
            f"  [dim]persona set-sysprompt --persona {persona_path} --text \"You are ...\"[/dim]"
        )
    console.print(f"Next: `persona sweep --persona {persona_path}`")


@app.command()
def sweep(
    persona_path: Path = typer.Option(..., "--persona", "-p", help="Path to persona.yaml."),
    judge: str = typer.Option("local-llama", "--judge", help="local-llama | openai | anthropic."),
    num_coefficients: int = typer.Option(8, "--num-coefficients", help="Coefficients per feature."),
    num_eval_prompts: int = typer.Option(18, "--num-eval-prompts", help="Eval prompts per coefficient."),
    num_calibration_prompts: int = typer.Option(50, "--num-calibration-prompts", help="Calibration prompts."),
    joint: bool = typer.Option(False, "--joint", help="Add a 3^N grid search around per-feature optima."),
    out_root: Path = typer.Option(Path("logs"), "--out-root", help="Root dir for log folders."),
):
    """Run the coefficient sweep, write report + samples, update persona.yaml in place."""
    from .sweep import run_sweep

    sweep_cfg = SweepConfig(
        num_coefficients=num_coefficients,
        num_eval_prompts=num_eval_prompts,
        num_calibration_prompts=num_calibration_prompts,
        judge_backend=judge,
        joint=joint,
    )
    run_sweep(persona_path, sweep_cfg=sweep_cfg, out_root=out_root)


@app.command("set-sysprompt")
def set_sysprompt_cmd(
    persona_path: Path = typer.Option(..., "--persona", "-p", help="Path to persona.yaml."),
    text: str | None = typer.Option(None, "--text", "-t", help="System prompt text. Mutually exclusive with --from-file / --clear."),
    from_file: Path | None = typer.Option(None, "--from-file", "-f", help="Read system prompt from a file."),
    clear: bool = typer.Option(False, "--clear", help="Remove the system prompt."),
):
    """Set, replace, or clear the system prompt on an existing persona.yaml.

    The system prompt carries the persona's *facts and identity* (company name,
    rules, hours, role); SAE steering shapes *style and obsession*. You almost
    always want both for a usable persona.
    """
    sources_set = sum(x is not None and x is not False for x in (text, from_file, clear or None))
    if sources_set != 1:
        console.print("[red]Pick exactly one of --text / --from-file / --clear.[/red]")
        raise typer.Exit(code=1)

    persona = load_persona(persona_path)

    if clear:
        persona.system_prompt = None
        persona.to_yaml(persona_path)
        console.print(f"[green]Cleared system prompt on {persona_path}[/green]")
        return

    if from_file is not None:
        if not from_file.exists():
            console.print(f"[red]File not found: {from_file}[/red]")
            raise typer.Exit(code=1)
        new_text = from_file.read_text()
    else:
        new_text = text or ""

    persona.system_prompt = new_text.rstrip("\n") + "\n" if new_text.strip() else None
    persona.to_yaml(persona_path)
    if persona.system_prompt:
        console.print(
            f"[green]Set system prompt on {persona_path}[/green] "
            f"({len(persona.system_prompt)} chars)"
        )
    else:
        console.print(f"[green]Cleared system prompt on {persona_path}[/green]")


@app.command()
def chat(
    persona_path: Path = typer.Option(..., "--persona", "-p", help="Path to persona.yaml."),
    coefficient_scale: float = typer.Option(1.0, "--coefficient-scale", help="Multiplier on all per-feature coefficients."),
):
    """Interactive REPL with the steered model."""
    from .chat import run_chat

    run_chat(persona_path, coefficient_scale=coefficient_scale)


@app.command()
def save(
    feature_ids: str = typer.Option(..., "--feature-ids", help="Comma-separated feature ids."),
    coefficients: str = typer.Option(..., "--coefficients", help="Comma-separated coefficients (same length)."),
    description: str = typer.Option(..., "--description", "-d", help="Persona description."),
    out: Path = typer.Option(..., "--out", "-o", help="Output yaml path."),
    system_prompt: str | None = typer.Option(None, "--system-prompt", "-s"),
    llm_name: str = typer.Option(DEFAULT_LLM_NAME, "--llm"),
):
    """Save a persona.yaml from inline args (skip find-features/sweep)."""
    fids = [int(x) for x in feature_ids.split(",")]
    coeffs = [float(x) for x in coefficients.split(",")]
    if len(fids) != len(coeffs):
        console.print(f"[red]feature_ids ({len(fids)}) and coefficients ({len(coeffs)}) length mismatch[/red]")
        raise typer.Exit(code=1)
    persona = PersonaSpec(
        description=description,
        sae=SAEConfig(),
        features=[FeatureSpec(feature_id=f, coefficient=c) for f, c in zip(fids, coeffs)],
        system_prompt=system_prompt,
        generation=GenerationConfig(),
        llm_name=llm_name,
    )
    persona.to_yaml(out)
    console.print(f"[green]Wrote {out}[/green]")


if __name__ == "__main__":
    app()
