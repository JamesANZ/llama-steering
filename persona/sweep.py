"""Coefficient sweep for one or more SAE features.

Procedural and somewhat long on purpose: easier to read top-to-bottom than to
chase abstractions through five files. The shape:

  1. Load model + SAE feature vectors once.
  2. Calibrate the layer's mean residual norm to anchor the coefficient range.
  3. For each feature: linear sweep of N coefficients, scoring every (coeff,
     prompt) pair with the LLM judge.
  4. Pick per-feature optimum by composite score.
  5. Joint validation pass at the per-feature optima.
  6. Optional `--joint` 3^N grid around per-feature optima.
  7. Write `report.md` + `samples.jsonl` and update persona.yaml in place.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import judge as judge_mod
from ._upstream import generate_steered_answer
from .calibration import mean_residual_norm
from .config import (
    DEFAULT_LLM_NAME,
    FeatureSpec,
    JudgeScore,
    PersonaSpec,
    SweepConfig,
)
from .eval_prompts import EVAL_PROMPTS
from .sae_loader import load_feature_vectors, pick_device, update_strengths

console = Console()


@dataclass
class SweepSample:
    """One generation + its judge scores."""

    feature_id: int
    coefficient: float
    prompt: str
    answer: str
    persona_fit: int | None
    coherence: int | None
    instruction_following: int | None
    composite: float

    @classmethod
    def from_score(
        cls, feature_id: int, coefficient: float, prompt: str, answer: str, score: JudgeScore
    ) -> "SweepSample":
        return cls(
            feature_id=feature_id,
            coefficient=float(coefficient),
            prompt=prompt,
            answer=answer,
            persona_fit=score.persona_fit,
            coherence=score.coherence,
            instruction_following=score.instruction_following,
            composite=score.composite(),
        )


@dataclass
class FeatureSweepResult:
    """Aggregated per-coefficient stats for one feature."""

    feature_id: int
    coefficients: list[float]
    persona_fit: list[float]
    coherence: list[float]
    instruction_following: list[float]
    composite: list[float]
    samples: list[SweepSample] = field(default_factory=list)

    def best_index(self) -> int:
        """Argmax over composite. If all 0, pick the smallest non-zero coeff."""
        comp = np.asarray(self.composite)
        if (comp <= 0).all():
            return 0
        return int(np.argmax(comp))

    def knee_index(self, coherence_floor: float = 6.0) -> int | None:
        """Largest coefficient where mean coherence stays above the floor.

        Used as a secondary recommendation: 'maximum persona signal you can get
        without breaking English'.
        """
        for i in range(len(self.coefficients) - 1, -1, -1):
            if self.coherence[i] >= coherence_floor and self.persona_fit[i] > 0:
                return i
        return None


def _build_chat(system_prompt: str | None, user: str) -> list[dict]:
    chat: list[dict] = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": user})
    return chat


def _generate(llm, components, chat, gen_cfg) -> str:
    out = generate_steered_answer(
        llm,
        chat,
        components,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        repetition_penalty=gen_cfg.repetition_penalty,
        steer_prompt=gen_cfg.steer_prompt,
        clamp_intensity=gen_cfg.clamp_intensity,
    )
    return out["answer"]


def _aggregate(samples: list[SweepSample]) -> tuple[float, float, float, float]:
    """Mean of (persona_fit, coherence, instruction_following, composite)."""
    if not samples:
        return 0.0, 0.0, 0.0, 0.0
    pf = np.mean([s.persona_fit or 0 for s in samples])
    co = np.mean([s.coherence or 0 for s in samples])
    instr = np.mean([s.instruction_following or 0 for s in samples])
    comp = np.mean([s.composite for s in samples])
    return float(pf), float(co), float(instr), float(comp)


def _sweep_one_feature(
    llm,
    backend: judge_mod.JudgeBackend,
    persona: PersonaSpec,
    feature_idx_in_list: int,
    base_components: list[dict],
    coefficients: list[float],
    eval_prompts: list[str],
    samples_path: Path,
    progress: Progress,
    overall_task,
) -> FeatureSweepResult:
    """Linear sweep on one feature; all other features held at strength 0."""
    feat = persona.features[feature_idx_in_list]
    fid = feat.feature_id

    pf_list: list[float] = []
    co_list: list[float] = []
    instr_list: list[float] = []
    comp_list: list[float] = []
    all_samples: list[SweepSample] = []

    feat_task = progress.add_task(
        f"[cyan]feature {fid}[/cyan]", total=len(coefficients) * len(eval_prompts)
    )

    for coeff in coefficients:
        per_coeff_samples: list[SweepSample] = []
        # Build a coefficient vector with all features at 0 except this one.
        strengths = [0.0] * len(base_components)
        strengths[feature_idx_in_list] = float(coeff)
        components = update_strengths(base_components, strengths)

        torch.manual_seed(0)
        for prompt in eval_prompts:
            chat = _build_chat(persona.system_prompt, prompt)
            try:
                answer = _generate(llm, components, chat, persona.generation)
            except Exception as e:
                console.print(f"[red]generation failed at coeff={coeff:.2f} prompt={prompt!r}: {e}[/red]")
                answer = ""
            score = judge_mod.score(prompt, answer, persona.description, backend)
            sample = SweepSample.from_score(fid, coeff, prompt, answer, score)
            per_coeff_samples.append(sample)
            all_samples.append(sample)
            with samples_path.open("a") as f:
                f.write(json.dumps(asdict(sample)) + "\n")
            progress.advance(feat_task)
            progress.advance(overall_task)

        pf, co, instr, comp = _aggregate(per_coeff_samples)
        pf_list.append(pf)
        co_list.append(co)
        instr_list.append(instr)
        comp_list.append(comp)
        console.print(
            f"  feat {fid} coeff {coeff:6.2f}  pf={pf:4.1f}  co={co:4.1f}  instr={instr:4.1f}  composite={comp:6.1f}"
        )

    progress.remove_task(feat_task)
    return FeatureSweepResult(
        feature_id=fid,
        coefficients=list(coefficients),
        persona_fit=pf_list,
        coherence=co_list,
        instruction_following=instr_list,
        composite=comp_list,
        samples=all_samples,
    )


def _evaluate_combination(
    llm,
    backend: judge_mod.JudgeBackend,
    persona: PersonaSpec,
    base_components: list[dict],
    strengths: list[float],
    eval_prompts: list[str],
    samples_path: Path,
    label: str,
) -> tuple[list[SweepSample], tuple[float, float, float, float]]:
    """Generate + score a single feature-strength combination over eval prompts."""
    components = update_strengths(base_components, strengths)
    samples: list[SweepSample] = []
    torch.manual_seed(0)
    for prompt in eval_prompts:
        chat = _build_chat(persona.system_prompt, prompt)
        try:
            answer = _generate(llm, components, chat, persona.generation)
        except Exception as e:
            console.print(f"[red]{label} generation failed for prompt={prompt!r}: {e}[/red]")
            answer = ""
        score_obj = judge_mod.score(prompt, answer, persona.description, backend)
        sample = SweepSample.from_score(-1, 0.0, prompt, answer, score_obj)  # marker fid -1
        samples.append(sample)
        with samples_path.open("a") as f:
            row = asdict(sample)
            row["label"] = label
            row["strengths"] = strengths
            f.write(json.dumps(row) + "\n")
    agg = _aggregate(samples)
    console.print(
        f"  [bold]{label}[/bold] strengths={[round(s, 2) for s in strengths]}  "
        f"pf={agg[0]:4.1f}  co={agg[1]:4.1f}  instr={agg[2]:4.1f}  composite={agg[3]:6.1f}"
    )
    return samples, agg


def _joint_grid(
    per_feature_optima: list[float], scales: tuple[float, ...]
) -> list[list[float]]:
    """All combinations of per-feature-optimum * scale, length len(scales)^N."""
    options = [[opt * s for s in scales] for opt in per_feature_optima]
    return [list(combo) for combo in itertools.product(*options)]


def _write_report(
    persona: PersonaSpec,
    sweep_cfg: SweepConfig,
    anchor: float,
    per_feature: list[FeatureSweepResult],
    chosen_coefficients: list[float],
    joint_agg: tuple[float, float, float, float],
    joint_warning: str | None,
    joint_grid_results: list[tuple[list[float], tuple[float, float, float, float]]] | None,
    out_dir: Path,
    elapsed_s: float,
) -> Path:
    lines: list[str] = []
    lines.append(f"# Persona steering report: {persona.description}")
    lines.append("")
    lines.append(f"- Generated in {elapsed_s / 60:.1f} min")
    lines.append(f"- SAE: `{persona.sae.release}` / `{persona.sae.sae_id}` (layer {persona.sae.layer})")
    lines.append(f"- Calibration anchor (mean residual norm × {sweep_cfg.anchor_multiplier}) = **{anchor:.2f}**")
    lines.append(f"- Coefficient range per feature: `[0, {anchor * sweep_cfg.coefficient_max_multiplier:.2f}]` "
                 f"in {sweep_cfg.num_coefficients} steps")
    lines.append(f"- Eval prompts: {sweep_cfg.num_eval_prompts}")
    lines.append(f"- Judge backend: `{sweep_cfg.judge_backend}`")
    if persona.system_prompt:
        lines.append(f"- System prompt: yes ({len(persona.system_prompt)} chars)")
    lines.append("")

    lines.append("## Recommended coefficients")
    lines.append("")
    lines.append("| feature | explanation | recommended coeff |")
    lines.append("|--------:|-------------|------------------:|")
    for feat, c in zip(persona.features, chosen_coefficients):
        expl = (feat.explanation or "").replace("|", "/")
        lines.append(f"| {feat.feature_id} | {expl[:70]} | {c:.2f} |")
    lines.append("")

    lines.append("## Joint validation pass")
    lines.append("")
    lines.append(f"All features at recommended coefficients, mean over {sweep_cfg.num_eval_prompts} prompts:")
    lines.append("")
    lines.append(
        f"- persona_fit: **{joint_agg[0]:.1f}** / 10\n"
        f"- coherence: **{joint_agg[1]:.1f}** / 10\n"
        f"- instruction_following: **{joint_agg[2]:.1f}** / 10\n"
        f"- composite (product of means): **{joint_agg[3]:.1f}**"
    )
    lines.append("")
    if joint_warning:
        lines.append(f"> WARNING: {joint_warning}")
        lines.append("")

    if joint_grid_results:
        lines.append("## Joint grid (--joint)")
        lines.append("")
        lines.append("| strengths | persona_fit | coherence | instruction_following | composite |")
        lines.append("|-----------|------------:|----------:|----------------------:|----------:|")
        for strengths, agg in joint_grid_results:
            lines.append(
                f"| {[round(s, 2) for s in strengths]} | {agg[0]:.1f} | {agg[1]:.1f} | {agg[2]:.1f} | {agg[3]:.1f} |"
            )
        lines.append("")

    lines.append("## Per-feature sweeps")
    lines.append("")
    for fr in per_feature:
        lines.append(f"### Feature {fr.feature_id}")
        lines.append("")
        lines.append("| coefficient | persona_fit | coherence | instruction_following | composite |")
        lines.append("|-----------:|------------:|----------:|----------------------:|----------:|")
        for c, pf, co, instr, comp in zip(
            fr.coefficients, fr.persona_fit, fr.coherence, fr.instruction_following, fr.composite
        ):
            lines.append(
                f"| {c:.2f} | {pf:.1f} | {co:.1f} | {instr:.1f} | {comp:.1f} |"
            )
        lines.append("")
        knee = fr.knee_index()
        if knee is not None:
            lines.append(
                f"Knee (largest coefficient with mean coherence ≥ 6 and any persona signal): "
                f"`{fr.coefficients[knee]:.2f}` "
                f"(persona_fit={fr.persona_fit[knee]:.1f}, coherence={fr.coherence[knee]:.1f})."
            )
        lines.append("")

    lines.append("## Sample generations")
    lines.append("")
    lines.append("Per feature, one prompt per coefficient. Full data in `samples.jsonl`.")
    lines.append("")
    sample_prompt = EVAL_PROMPTS[0] if EVAL_PROMPTS else ""
    for fr in per_feature:
        lines.append(f"### Feature {fr.feature_id} — prompt: {sample_prompt!r}")
        lines.append("")
        for c in fr.coefficients:
            for s in fr.samples:
                if s.coefficient == c and s.prompt == sample_prompt:
                    lines.append(f"**coeff {c:.2f}** (pf {s.persona_fit}/co {s.coherence}/instr {s.instruction_following}):")
                    lines.append("")
                    lines.append(f"> {s.answer.strip().replace(chr(10), ' ')[:600]}")
                    lines.append("")
                    break
        lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def run_sweep(
    persona_path: Path | str,
    sweep_cfg: SweepConfig | None = None,
    eval_prompts: list[str] | None = None,
    out_root: Path | str = Path("logs"),
    llm: Any | None = None,
) -> Path:
    """Run the full sweep on `persona_path`. Mutates the yaml in place.

    Returns the path to the generated report.md.
    """
    sweep_cfg = sweep_cfg or SweepConfig()
    eval_prompts = (eval_prompts or EVAL_PROMPTS)[: sweep_cfg.num_eval_prompts]
    persona = PersonaSpec.from_yaml(persona_path)

    if not persona.features:
        raise ValueError(
            f"{persona_path} has no features. Run "
            f"`persona find-features --persona {persona_path}` first."
        )

    out_root = Path(out_root)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / f"{ts}_{persona.slug()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    samples_path.write_text("")
    console.print(f"[bold]Sweep log dir:[/bold] {out_dir}")

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

    base_components = load_feature_vectors(
        persona.sae,
        feature_ids=persona.feature_ids(),
        device=device,
    )

    # Pull fresh calibration prompts from the eval pool (cheap, local).
    calib_prompts = (eval_prompts * (sweep_cfg.num_calibration_prompts // max(len(eval_prompts), 1) + 1))[
        : sweep_cfg.num_calibration_prompts
    ]
    anchor_norm = mean_residual_norm(
        llm, persona.sae.layer, calib_prompts, llm_name=persona.llm_name
    )
    anchor = anchor_norm * sweep_cfg.anchor_multiplier
    coeff_max = anchor * sweep_cfg.coefficient_max_multiplier
    coefficients = list(np.linspace(0.0, coeff_max, sweep_cfg.num_coefficients))
    console.print(
        f"Anchor (0.5 × mean residual norm) = {anchor:.2f}; "
        f"sweeping {len(coefficients)} coefficients in [0, {coeff_max:.2f}]"
    )

    backend = judge_mod.make_backend(
        sweep_cfg.judge_backend,
        llm=llm,
        openai_model=sweep_cfg.judge_model_openai,
        anthropic_model=sweep_cfg.judge_model_anthropic,
    )
    console.print(f"Judge backend: {backend.name}")

    n_features = len(persona.features)
    total_evals = n_features * len(coefficients) * len(eval_prompts)
    console.print(
        f"[bold]Sweeping {n_features} feature(s) × {len(coefficients)} coefficients "
        f"× {len(eval_prompts)} prompts = {total_evals} generations[/bold]"
    )

    start = time.time()
    per_feature: list[FeatureSweepResult] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[green]overall sweep[/green]", total=total_evals)
        for i in range(n_features):
            console.rule(f"Feature {i + 1}/{n_features}: id={persona.features[i].feature_id}")
            result = _sweep_one_feature(
                llm,
                backend,
                persona,
                feature_idx_in_list=i,
                base_components=base_components,
                coefficients=coefficients,
                eval_prompts=eval_prompts,
                samples_path=samples_path,
                progress=progress,
                overall_task=overall,
            )
            per_feature.append(result)

    chosen_coefficients = [fr.coefficients[fr.best_index()] for fr in per_feature]
    console.rule("Joint validation")
    joint_samples, joint_agg = _evaluate_combination(
        llm, backend, persona, base_components, chosen_coefficients, eval_prompts,
        samples_path, label="joint",
    )

    per_feature_mean_composite = float(np.mean([fr.composite[fr.best_index()] for fr in per_feature]))
    joint_warning: str | None = None
    if per_feature_mean_composite > 0:
        drop = (per_feature_mean_composite - joint_agg[3]) / per_feature_mean_composite
        if drop > sweep_cfg.joint_regression_threshold:
            joint_warning = (
                f"Joint composite dropped {drop * 100:.0f}% vs the per-feature mean "
                f"({joint_agg[3]:.1f} vs {per_feature_mean_composite:.1f}). "
                f"Features may be interfering. Rerun with `--joint` to fine-tune."
            )
            console.print(f"[yellow]{joint_warning}[/yellow]")

    joint_grid_results = None
    if sweep_cfg.joint:
        if n_features > 3:
            console.print(
                "[yellow]--joint with >3 features uses too many evaluations. "
                "Run upstream's optimize_botorch.py for proper multi-dim BO.[/yellow]"
            )
        else:
            console.rule(f"Joint grid ({len(sweep_cfg.joint_grid_scales)}^{n_features} combinations)")
            joint_grid_results = []
            for combo in _joint_grid(chosen_coefficients, sweep_cfg.joint_grid_scales):
                _, agg = _evaluate_combination(
                    llm, backend, persona, base_components, combo, eval_prompts,
                    samples_path, label=f"grid_{[round(c, 2) for c in combo]}",
                )
                joint_grid_results.append((combo, agg))
            best_combo, best_agg = max(joint_grid_results, key=lambda x: x[1][3])
            if best_agg[3] > joint_agg[3]:
                console.print(
                    f"[green]Joint-grid winner improved composite "
                    f"{joint_agg[3]:.1f} -> {best_agg[3]:.1f}; updating coefficients.[/green]"
                )
                chosen_coefficients = best_combo
                joint_agg = best_agg

    persona.features = [
        FeatureSpec(feature_id=feat.feature_id, explanation=feat.explanation, coefficient=float(c))
        for feat, c in zip(persona.features, chosen_coefficients)
    ]
    persona.to_yaml(persona_path)
    console.print(f"[green]Updated[/green] {persona_path} with recommended coefficients.")

    elapsed_s = time.time() - start
    report_path = _write_report(
        persona, sweep_cfg, anchor, per_feature, chosen_coefficients, joint_agg,
        joint_warning, joint_grid_results, out_dir, elapsed_s,
    )
    console.print(f"[bold green]Report:[/bold green] {report_path}")
    return report_path
