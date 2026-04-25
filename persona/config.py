"""Pydantic config + persona.yaml schema.

A `persona.yaml` is the unit of distribution for a steered persona. It bundles
the SAE configuration, selected features with per-feature steering coefficients,
an optional system prompt for facts/identity, and generation settings.

Example yaml:

    description: "Acme Bank customer service rep"
    sae:
      release: llama_scope_lxr_8x
      sae_id: l15r_8x
      layer: 15
    features:
      - feature_id: 12345
        explanation: "polite customer-service language and refusals"
        coefficient: 3.2
      - feature_id: 67890
        explanation: "personal banking, accounts, deposits"
        coefficient: 2.1
    system_prompt: |
      You are an assistant for Acme Bank. Our hours are 9-5 Mon-Fri.
    generation:
      temperature: 0.7
      max_new_tokens: 256
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


DEFAULT_SAE_RELEASE = "llama_scope_lxr_8x"
DEFAULT_SAE_ID = "l15r_8x"
DEFAULT_LAYER = 15
DEFAULT_NEURONPEDIA_MODEL = "llama3.1-8b"
DEFAULT_NEURONPEDIA_SOURCE = "15-llamascope-res-32k"
DEFAULT_LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"


class SAEConfig(BaseModel):
    """Identifies which SAE we're using and where its features live."""

    release: str = DEFAULT_SAE_RELEASE
    sae_id: str = DEFAULT_SAE_ID
    layer: int = DEFAULT_LAYER
    neuronpedia_model: str = DEFAULT_NEURONPEDIA_MODEL
    neuronpedia_source: str = DEFAULT_NEURONPEDIA_SOURCE


class FeatureSpec(BaseModel):
    """One picked feature, optionally with a tuned coefficient."""

    feature_id: int
    explanation: str = ""
    coefficient: float | None = None

    @field_validator("feature_id")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"feature_id must be >= 0, got {v}")
        return v


class GenerationConfig(BaseModel):
    """Generation defaults applied at chat / sweep time."""

    temperature: float = 0.7
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0
    steer_prompt: bool = True
    clamp_intensity: bool = False


class PersonaSpec(BaseModel):
    """The persona artifact. Loaded from / saved to persona.yaml."""

    description: str
    sae: SAEConfig = Field(default_factory=SAEConfig)
    features: list[FeatureSpec] = Field(default_factory=list)
    system_prompt: str | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    llm_name: str = DEFAULT_LLM_NAME

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PersonaSpec":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Persona file not found: {path}. "
                f"Create one with `persona new \"<description>\" --out {path}`."
            )
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path, backup: bool = True) -> None:
        """Atomically write to yaml. Backs up the existing file to .bak first."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if backup and path.exists():
            shutil.copy(path, path.with_suffix(path.suffix + ".bak"))

        data = self.model_dump(mode="python", exclude_none=False)
        # exclude None coefficients so freshly-picked yamls stay readable
        for feat in data.get("features", []):
            if feat.get("coefficient") is None:
                feat.pop("coefficient", None)
        if data.get("system_prompt") is None:
            data.pop("system_prompt", None)

        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, width=100)
        tmp.replace(path)

    def slug(self) -> str:
        """Filesystem-safe slug derived from description for log folder names."""
        out = "".join(c if c.isalnum() else "_" for c in self.description.lower())
        out = "_".join(p for p in out.split("_") if p)
        return out[:48] or "persona"

    def feature_ids(self) -> list[int]:
        return [f.feature_id for f in self.features]

    def has_coefficients(self) -> bool:
        return bool(self.features) and all(f.coefficient is not None for f in self.features)


class SweepConfig(BaseModel):
    """Sweep run configuration. Mostly defaults; overridable on the CLI."""

    num_coefficients: int = 8
    anchor_multiplier: float = 0.5  # coeff range = [0, 2 * anchor_multiplier * mean_norm]
    coefficient_max_multiplier: float = 2.0
    num_eval_prompts: int = 18
    num_calibration_prompts: int = 50
    judge_backend: str = "local-llama"  # "local-llama" | "openai" | "anthropic"
    judge_model_openai: str = "gpt-4o-mini"
    judge_model_anthropic: str = "claude-3-5-sonnet-latest"
    joint: bool = False
    joint_grid_points: int = 3
    joint_grid_scales: tuple[float, float, float] = (0.7, 1.0, 1.3)
    joint_regression_threshold: float = 0.20  # warn if joint composite drops >20% vs mean
    seed: int = 16


class JudgeScore(BaseModel):
    """Three rubrics scored 0-10 each. None = parse failure."""

    persona_fit: int | None = None
    coherence: int | None = None
    instruction_following: int | None = None

    def composite(self) -> float:
        """Product score with None treated as 0."""
        pf = self.persona_fit or 0
        co = self.coherence or 0
        instr = self.instruction_following or 0
        return float(pf * co * instr)

    def is_complete(self) -> bool:
        return all(
            v is not None
            for v in (self.persona_fit, self.coherence, self.instruction_following)
        )


def blank_persona(description: str) -> PersonaSpec:
    """A minimal, fresh persona with just a description filled in."""
    return PersonaSpec(description=description)


def load_persona(path: str | Path) -> PersonaSpec:
    """Convenience wrapper."""
    return PersonaSpec.from_yaml(path)


def save_persona(persona: PersonaSpec, path: str | Path) -> None:
    """Convenience wrapper."""
    persona.to_yaml(path)


_KEY_ORDER = ("description", "sae", "features", "system_prompt", "generation", "llm_name")


def dict_to_yaml_str(d: dict[str, Any]) -> str:
    """Stable yaml dump with our preferred key order. Used in tests."""
    ordered: dict[str, Any] = {}
    for k in _KEY_ORDER:
        if k in d:
            ordered[k] = d[k]
    for k, v in d.items():
        if k not in ordered:
            ordered[k] = v
    return yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False, width=100)
