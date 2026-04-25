"""Sweep recommendation logic on synthetic data (no model required)."""

import itertools

import pytest

from persona.config import JudgeScore
from persona.sweep import (
    FeatureSweepResult,
    SweepSample,
    _aggregate,
    _joint_grid,
)


def _mk_sample(fid, coeff, prompt, pf, co, instr, answer="x"):
    score = JudgeScore(persona_fit=pf, coherence=co, instruction_following=instr)
    return SweepSample.from_score(fid, coeff, prompt, answer, score)


def test_best_index_picks_max_composite():
    fr = FeatureSweepResult(
        feature_id=1,
        coefficients=[0.0, 1.0, 2.0, 3.0, 4.0],
        persona_fit=[0, 3, 7, 8, 6],
        coherence=[10, 9, 8, 5, 2],
        instruction_following=[10, 8, 7, 4, 1],
        composite=[0, 3 * 9 * 8, 7 * 8 * 7, 8 * 5 * 4, 6 * 2 * 1],
    )
    assert fr.best_index() == 2  # 7*8*7 = 392 > 8*5*4 = 160 > 3*9*8 = 216


def test_best_index_falls_back_when_all_zero():
    fr = FeatureSweepResult(
        feature_id=1,
        coefficients=[0.0, 1.0, 2.0],
        persona_fit=[0, 0, 0],
        coherence=[0, 0, 0],
        instruction_following=[0, 0, 0],
        composite=[0, 0, 0],
    )
    assert fr.best_index() == 0


def test_knee_index_returns_largest_safe_coeff():
    fr = FeatureSweepResult(
        feature_id=1,
        coefficients=[0.0, 1.0, 2.0, 3.0, 4.0],
        persona_fit=[0, 4, 6, 7, 9],
        coherence=[10, 9, 7, 5, 2],
        instruction_following=[10, 8, 7, 5, 1],
        composite=[0, 4 * 9 * 8, 6 * 7 * 7, 7 * 5 * 5, 9 * 2 * 1],
    )
    knee = fr.knee_index(coherence_floor=6.0)
    # Largest coefficient with coherence >= 6 AND persona_fit > 0 is index 2 (coeff 2.0).
    assert knee == 2
    assert fr.coefficients[knee] == 2.0


def test_knee_index_none_if_persona_never_appears():
    fr = FeatureSweepResult(
        feature_id=1,
        coefficients=[0.0, 1.0, 2.0],
        persona_fit=[0, 0, 0],
        coherence=[10, 9, 8],
        instruction_following=[10, 9, 8],
        composite=[0, 0, 0],
    )
    assert fr.knee_index() is None


def test_aggregate_handles_empty_list():
    assert _aggregate([]) == (0.0, 0.0, 0.0, 0.0)


def test_aggregate_treats_none_as_zero():
    samples = [
        _mk_sample(1, 0, "p", None, 5, 5),
        _mk_sample(1, 0, "p", 4, None, 6),
    ]
    pf, co, instr, comp = _aggregate(samples)
    assert pf == pytest.approx((0 + 4) / 2)
    assert co == pytest.approx((5 + 0) / 2)
    assert instr == pytest.approx((5 + 6) / 2)


def test_joint_grid_size_matches_scales_to_the_n():
    optima = [3.0, 2.0]
    scales = (0.7, 1.0, 1.3)
    grid = _joint_grid(optima, scales)
    assert len(grid) == len(scales) ** len(optima)
    expected = list(itertools.product(
        [3.0 * 0.7, 3.0, 3.0 * 1.3],
        [2.0 * 0.7, 2.0, 2.0 * 1.3],
    ))
    assert sorted(tuple(c) for c in grid) == sorted(expected)


def test_joint_grid_single_feature():
    grid = _joint_grid([5.0], (0.7, 1.0, 1.3))
    assert grid == [[5.0 * 0.7], [5.0], [5.0 * 1.3]]


def test_joint_grid_zero_optimum_keeps_zero():
    grid = _joint_grid([0.0, 1.0], (0.5, 1.0))
    # 0.5 * 0 = 0, 1.0 * 0 = 0; first dim collapses but combinations still enumerate.
    assert all(combo[0] == 0.0 for combo in grid)


def test_sweep_sample_round_trip():
    s = SweepSample.from_score(
        feature_id=42,
        coefficient=2.5,
        prompt="hi",
        answer="hello",
        score=JudgeScore(persona_fit=7, coherence=8, instruction_following=9),
    )
    assert s.feature_id == 42
    assert s.composite == pytest.approx(7 * 8 * 9)
