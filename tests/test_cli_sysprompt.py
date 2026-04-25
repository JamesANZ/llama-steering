"""set-sysprompt CLI command + multiline-prompt yaml round-trip."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from persona.cli import app
from persona.config import PersonaSpec, blank_persona

runner = CliRunner()


def _write_blank(path: Path, description: str = "test persona") -> None:
    blank_persona(description).to_yaml(path)


def test_set_sysprompt_with_text(tmp_path):
    p = tmp_path / "p.yaml"
    _write_blank(p)

    result = runner.invoke(app, ["set-sysprompt", "--persona", str(p), "--text", "You are a helper."])
    assert result.exit_code == 0, result.output

    loaded = PersonaSpec.from_yaml(p)
    assert loaded.system_prompt is not None
    assert "helper" in loaded.system_prompt


def test_set_sysprompt_from_file_preserves_multiline(tmp_path):
    p = tmp_path / "p.yaml"
    _write_blank(p)
    sp_file = tmp_path / "sp.txt"
    sp_file.write_text("line one\nline two\n  indented line\n")

    result = runner.invoke(app, ["set-sysprompt", "--persona", str(p), "--from-file", str(sp_file)])
    assert result.exit_code == 0, result.output

    loaded = PersonaSpec.from_yaml(p)
    assert loaded.system_prompt is not None
    assert "line one" in loaded.system_prompt
    assert "line two" in loaded.system_prompt
    assert "indented line" in loaded.system_prompt


def test_set_sysprompt_clear(tmp_path):
    p = tmp_path / "p.yaml"
    persona = blank_persona("test")
    persona.system_prompt = "preexisting"
    persona.to_yaml(p)

    result = runner.invoke(app, ["set-sysprompt", "--persona", str(p), "--clear"])
    assert result.exit_code == 0, result.output

    loaded = PersonaSpec.from_yaml(p)
    assert loaded.system_prompt is None


def test_set_sysprompt_requires_exactly_one_source(tmp_path):
    p = tmp_path / "p.yaml"
    _write_blank(p)

    no_args = runner.invoke(app, ["set-sysprompt", "--persona", str(p)])
    assert no_args.exit_code != 0

    both = runner.invoke(
        app,
        ["set-sysprompt", "--persona", str(p), "--text", "a", "--clear"],
    )
    assert both.exit_code != 0


def test_set_sysprompt_missing_file(tmp_path):
    p = tmp_path / "p.yaml"
    _write_blank(p)
    result = runner.invoke(app, ["set-sysprompt", "--persona", str(p), "--from-file", str(tmp_path / "nope.txt")])
    assert result.exit_code != 0


def test_new_command_with_inline_system_prompt(tmp_path):
    out = tmp_path / "p.yaml"
    sp = "You are a die-hard McDonald's superfan."
    result = runner.invoke(app, ["new", "test description", "--out", str(out), "-s", sp])
    assert result.exit_code == 0, result.output
    loaded = PersonaSpec.from_yaml(out)
    assert loaded.system_prompt == sp


def test_yaml_roundtrip_multiline_system_prompt(tmp_path):
    p = PersonaSpec(
        description="acme bank rep",
        system_prompt=(
            "You are an assistant for Acme Bank.\n"
            "Our hours are 9am to 5pm Monday through Friday.\n"
            "You may not provide investment advice.\n"
            "    Indented bullet survives.\n"
        ),
    )
    out = tmp_path / "acme.yaml"
    p.to_yaml(out)
    loaded = PersonaSpec.from_yaml(out)
    assert loaded.system_prompt == p.system_prompt
    text = out.read_text()
    assert "system_prompt" in text


def test_chat_history_seeded_with_system_prompt(tmp_path):
    """If the persona has a system_prompt, chat() seeds history with a system message.

    We don't actually start the REPL; just check the build_chat helper used by
    sweep does the same thing chat does.
    """
    from persona.sweep import _build_chat

    chat_with_sp = _build_chat("you are helpful", "hello")
    assert chat_with_sp[0] == {"role": "system", "content": "you are helpful"}
    assert chat_with_sp[1] == {"role": "user", "content": "hello"}

    chat_without_sp = _build_chat(None, "hello")
    assert len(chat_without_sp) == 1
    assert chat_without_sp[0]["role"] == "user"
