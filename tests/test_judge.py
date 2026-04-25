"""Judge score parser must tolerate noisy LLM output."""

import pytest

from persona.config import JudgeScore
from persona.judge import parse_score


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Rating: [[7]]", 7),
        ("Rating: [[10]]", 10),
        ("Rating: [[0]]", 0),
        ("Rating:7", 7),
        ("Rating: 7", 7),
        ("rating - 4", 4),
        ("Score: 6/10", 6),
        ("I would rate it a 8.", 8),
        ("I'd give it a 5 because the persona is partially present.", 5),
        ("\n\n7\n\n", 7),
        ("Some preamble. Rating: [[ 9 ]] trailing junk.", 9),
        ("[[3]]", 3),
    ],
)
def test_parse_score_extracts_number(text, expected):
    assert parse_score(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "n/a",
        "I refuse to score this.",
        "Rating: ten",  # word, not digit; we only match digits
        "12345",  # 12345 is not a valid 0-10 rating; pattern requires word boundary
    ],
)
def test_parse_score_rejects_garbage(text):
    assert parse_score(text) is None


def test_parse_score_clamps_out_of_range():
    # 99 is not in 0-10 so we reject it; the parser should NOT silently round.
    assert parse_score("Rating: 99") is None
    assert parse_score("Score: 11/10") is None


def test_parse_score_first_match_wins():
    # Prompts ask for one rating; if they sneak in two, accept the first.
    assert parse_score("Rating: [[3]] then Rating: [[7]]") == 3


def test_judge_score_composite_handles_none():
    s = JudgeScore(persona_fit=None, coherence=5, instruction_following=5)
    assert s.composite() == 0.0
    assert not s.is_complete()


def test_judge_score_composite_product():
    s = JudgeScore(persona_fit=2, coherence=3, instruction_following=4)
    assert s.composite() == 24.0
    assert s.is_complete()
