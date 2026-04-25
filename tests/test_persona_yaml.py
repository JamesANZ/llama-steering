"""PersonaSpec yaml round-trip and schema validation."""

from pathlib import Path

import pytest

from persona.config import (
    DEFAULT_LAYER,
    FeatureSpec,
    PersonaSpec,
    SAEConfig,
    blank_persona,
)


def test_blank_persona_has_no_features(tmp_path):
    p = blank_persona("test persona")
    assert p.description == "test persona"
    assert p.features == []
    assert p.system_prompt is None
    assert p.sae.layer == DEFAULT_LAYER


def test_yaml_roundtrip_preserves_features(tmp_path):
    p = PersonaSpec(
        description="acme bank rep",
        features=[
            FeatureSpec(feature_id=12345, explanation="banking", coefficient=2.5),
            FeatureSpec(feature_id=67890, explanation="customer service", coefficient=1.8),
        ],
        system_prompt="You are an Acme Bank assistant.",
    )
    out = tmp_path / "acme.yaml"
    p.to_yaml(out)
    loaded = PersonaSpec.from_yaml(out)
    assert loaded.description == "acme bank rep"
    assert len(loaded.features) == 2
    assert loaded.features[0].feature_id == 12345
    assert loaded.features[0].coefficient == pytest.approx(2.5)
    assert loaded.features[1].feature_id == 67890
    assert loaded.system_prompt == "You are an Acme Bank assistant."


def test_yaml_roundtrip_drops_none_coefficients(tmp_path):
    p = PersonaSpec(
        description="fresh persona",
        features=[FeatureSpec(feature_id=1, explanation="a"), FeatureSpec(feature_id=2)],
    )
    out = tmp_path / "fresh.yaml"
    p.to_yaml(out)
    text = out.read_text()
    assert "coefficient" not in text
    loaded = PersonaSpec.from_yaml(out)
    assert all(f.coefficient is None for f in loaded.features)
    assert not loaded.has_coefficients()


def test_to_yaml_creates_backup(tmp_path):
    p = blank_persona("v1")
    out = tmp_path / "p.yaml"
    p.to_yaml(out)
    assert not out.with_suffix(".yaml.bak").exists()
    p2 = PersonaSpec(description="v2")
    p2.to_yaml(out)
    assert out.with_suffix(".yaml.bak").exists()
    assert "v1" in out.with_suffix(".yaml.bak").read_text()


def test_invalid_feature_id_rejected():
    with pytest.raises(ValueError):
        FeatureSpec(feature_id=-1)


def test_load_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        PersonaSpec.from_yaml(tmp_path / "nope.yaml")
    assert "persona new" in str(exc.value)


def test_slug_filesystem_safe():
    p = blank_persona("Obsessive McDonald's superfan!!!")
    s = p.slug()
    assert s
    assert all(c.isalnum() or c == "_" for c in s)
    assert "mcdonald" in s


def test_has_coefficients_only_when_all_set():
    p = PersonaSpec(
        description="x",
        features=[
            FeatureSpec(feature_id=1, coefficient=1.0),
            FeatureSpec(feature_id=2),
        ],
    )
    assert not p.has_coefficients()

    p.features[1] = FeatureSpec(feature_id=2, coefficient=2.0)
    assert p.has_coefficients()


def test_default_sae_config_matches_neuronpedia_llama_scope():
    cfg = SAEConfig()
    assert cfg.release == "llama_scope_lxr_8x"
    assert cfg.sae_id == "l15r_8x"
    assert cfg.layer == 15
    assert cfg.neuronpedia_model == "llama3.1-8b"
    assert cfg.neuronpedia_source == "15-llamascope-res-32k"
