from __future__ import annotations

import pytest

from llama_agents.install import RecordedPrompter


def test_recorded_prompter_ask_returns_seeded_answer():
    p = RecordedPrompter(answers=["alice", "bob"])
    assert p.ask("name?") == "alice"
    assert p.ask("again?") == "bob"
    assert p.prompts_seen == ["name?", "again?"]


def test_recorded_prompter_ask_returns_default_when_empty():
    p = RecordedPrompter(answers=[""])
    assert p.ask("name?", default="alice") == "alice"


def test_recorded_prompter_confirm_yes_no_parsing():
    p = RecordedPrompter(answers=["y", "n", "", "yes", "no"])
    assert p.confirm("a?") is True
    assert p.confirm("b?") is False
    assert p.confirm("c?", default=True) is True
    assert p.confirm("d?") is True
    assert p.confirm("e?") is False


def test_recorded_prompter_choose_returns_index():
    p = RecordedPrompter(answers=["2"])
    assert p.choose("pick", ["a", "b", "c"]) == 1  # 1-indexed input -> 0-indexed return


def test_recorded_prompter_choose_default_index_on_empty():
    p = RecordedPrompter(answers=[""])
    assert p.choose("pick", ["a", "b", "c"], default_index=1) == 1


def test_recorded_prompter_info_and_warn_collected():
    p = RecordedPrompter(answers=[])
    p.info("hello")
    p.warn("careful")
    assert p.messages == [("info", "hello"), ("warn", "careful")]


def test_recorded_prompter_raises_when_out_of_answers():
    p = RecordedPrompter(answers=[])
    with pytest.raises(RuntimeError, match="ran out of scripted answers"):
        p.ask("name?")
