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


from llama_agents.install import CATALOGUE, ModelSpec, recommend_tier, tier_defaults


def test_catalogue_has_three_tiers():
    tiers = {m.tier for m in CATALOGUE}
    assert tiers == {"L", "M", "S"}


def test_catalogue_filenames_unique():
    names = [m.hf_filename for m in CATALOGUE]
    assert len(names) == len(set(names))


def test_recommend_tier_thresholds():
    assert recommend_tier(None) == "unknown"
    assert recommend_tier(0.0) == "unknown"
    assert recommend_tier(7.99) == "unknown"
    assert recommend_tier(8.0) == "S"
    assert recommend_tier(13.99) == "S"
    assert recommend_tier(14.0) == "M"
    assert recommend_tier(23.99) == "M"
    assert recommend_tier(24.0) == "L"
    assert recommend_tier(48.0) == "L"


def test_tier_defaults_match_spec():
    assert tier_defaults("L") == (65536, 2)
    assert tier_defaults("M") == (32768, 2)
    assert tier_defaults("S") == (8192, 1)
    assert tier_defaults("unknown") == (8192, 1)
