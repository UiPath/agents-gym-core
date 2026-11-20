"""Versions whose names contain ``__`` keep their full name in aggregated results."""

from agents_gym_core.runtime.runners.result_aggregator import _parse_agent_key


def test_a_version_with_a_double_underscore_keeps_its_name() -> None:
    assert _parse_agent_key("aime_2025_agent__reapo__opus-4.6-test") == (
        "aime_2025_agent",
        "reapo__opus-4.6-test",
    )
    assert _parse_agent_key("hover__gepa__gpt-5-test") == ("hover", "gepa__gpt-5-test")


def test_a_plain_key_still_parses() -> None:
    assert _parse_agent_key("sql__reapo__gpt-5-test") == ("sql", "reapo__gpt-5-test")
    assert _parse_agent_key("calculator") == ("calculator", "base")
