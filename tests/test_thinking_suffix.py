"""Tests for the `-thinking-<level>` suffix parsing on gemini model strings."""

import pytest

from feather_ai.internal_utils._provider import parse_thinking_suffix


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
def test_valid_levels_parse_and_strip(level):
    base, parsed = parse_thinking_suffix(f"gemini-3.5-flash-thinking-{level}")
    assert base == "gemini-3.5-flash"
    assert parsed == level


def test_no_suffix_returns_model_unchanged():
    assert parse_thinking_suffix("gemini-3.5-flash") == ("gemini-3.5-flash", None)


def test_non_gemini_with_coincidental_suffix_not_parsed():
    # A non-gemini model that happens to end in `-thinking-low` must be left alone.
    assert parse_thinking_suffix("claude-thinking-low") == ("claude-thinking-low", None)


def test_pre_gemini3_still_parses():
    # We do not gate on the gemini version; the API itself rejects thinking_level
    # on pre-3 models.
    base, parsed = parse_thinking_suffix("gemini-2.5-flash-thinking-high")
    assert base == "gemini-2.5-flash"
    assert parsed == "high"


def test_invalid_level_not_parsed():
    assert parse_thinking_suffix("gemini-3.5-flash-thinking-max") == (
        "gemini-3.5-flash-thinking-max",
        None,
    )
