"""
Pytest suite for the pure, LLM-free HER2 scoring stage.
No network or model calls — see CLAUDE.md's determinism requirement.
"""

import pytest

from her2_tools import HER2_SCORE_TOOL, score_her2_status


# --- IHC-only categories -----------------------------------------------

def test_ihc_0_is_negative():
    result = score_her2_status(ihc_score="0")
    assert result["status"] == "Negative"


def test_ihc_1plus_is_negative():
    result = score_her2_status(ihc_score="1+")
    assert result["status"] == "Negative"


def test_ihc_3plus_is_positive():
    result = score_her2_status(ihc_score="3+")
    assert result["status"] == "Positive"


def test_ihc_2plus_without_ish_is_equivocal():
    result = score_her2_status(ihc_score="2+")
    assert result["status"] == "Equivocal"
    assert "ish" in result["rationale"].lower()


def test_ihc_0_ignores_ish_values():
    result = score_her2_status(ihc_score="0", ish_ratio=5.0, ish_copy_number=10.0)
    assert result["status"] == "Negative"


def test_ihc_3plus_ignores_ish_values():
    result = score_her2_status(ihc_score="3+", ish_ratio=0.5, ish_copy_number=1.0)
    assert result["status"] == "Positive"


# --- IHC 2+ with ISH: five guideline groups, including boundaries -------

def test_ish_group1_ratio_and_copy_high_is_positive():
    # ratio >= 2.0 and copy >= 4.0 -> Positive (boundary values included)
    result = score_her2_status(ihc_score="2+", ish_ratio=2.0, ish_copy_number=4.0)
    assert result["status"] == "Positive"


def test_ish_group2_ratio_high_copy_low_is_negative():
    # ratio >= 2.0 and copy < 4.0 -> Negative (IHC isn't 3+)
    result = score_her2_status(ihc_score="2+", ish_ratio=2.0, ish_copy_number=3.999)
    assert result["status"] == "Negative"


def test_ish_group3_ratio_low_copy_very_high_is_negative():
    # ratio < 2.0 and copy >= 6.0 -> Negative (IHC isn't 3+; recount recommended)
    result = score_her2_status(ihc_score="2+", ish_ratio=1.999, ish_copy_number=6.0)
    assert result["status"] == "Negative" 


def test_ish_group4_ratio_low_copy_midrange_is_equivocal():
    # ratio < 2.0 and 4.0 <= copy < 6.0 -> Equivocal (IHC isn't clearly 3+)
    result = score_her2_status(ihc_score="2+", ish_ratio=1.999, ish_copy_number=4.0)
    assert result["status"] == "Equivocal"


def test_ish_group4_upper_boundary_is_equivocal():
    result = score_her2_status(ihc_score="2+", ish_ratio=1.0, ish_copy_number=5.999)
    assert result["status"] == "Equivocal"


def test_ish_group5_ratio_and_copy_low_is_negative():
    # ratio < 2.0 and copy < 4.0 -> Negative
    result = score_her2_status(ihc_score="2+", ish_ratio=1.0, ish_copy_number=2.0)
    assert result["status"] == "Negative"


# --- Invalid / missing input --------------------------------------------

def test_invalid_ihc_score_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="4+")


def test_missing_ihc_score_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score=None)


def test_empty_ihc_score_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="")


def test_non_string_ihc_score_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score=3)


def test_ihc_score_is_stripped_of_whitespace():
    result = score_her2_status(ihc_score=" 3+ ")
    assert result["status"] == "Positive"


def test_non_numeric_ish_ratio_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="2+", ish_ratio="high", ish_copy_number=5.0)


def test_non_numeric_ish_copy_number_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="2+", ish_ratio=2.0, ish_copy_number="lots")


def test_empty_string_ish_ratio_raises():
    # "" is not the same as None (i.e. "ISH not performed") — it's an invalid type.
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="2+", ish_ratio="", ish_copy_number=5.0)


def test_empty_string_ish_copy_number_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="2+", ish_ratio=2.0, ish_copy_number="")


def test_negative_ish_ratio_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="2+", ish_ratio=-1.0, ish_copy_number=5.0)


def test_negative_ish_copy_number_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="2+", ish_ratio=2.0, ish_copy_number=-1.0)


def test_partial_ish_values_raises():
    # Only one of ish_ratio/ish_copy_number provided is an ambiguous state,
    # not the same as "no ISH performed" (both None) — reject it.
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="2+", ish_ratio=2.0, ish_copy_number=None)


def test_partial_ish_values_other_direction_raises():
    with pytest.raises(ValueError):
        score_her2_status(ihc_score="2+", ish_ratio=None, ish_copy_number=5.0)


# --- Tool schema sanity check --------------------------------------------

def test_tool_schema_name_matches_function():
    assert HER2_SCORE_TOOL["function"]["name"] == score_her2_status.__name__
