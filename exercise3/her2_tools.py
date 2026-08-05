"""
HER2 scoring stage — pure, deterministic, LLM-free.

Implements the ASCO/CAP 2018 HER2 testing guideline (Wolff et al., J Clin
Oncol) as described in exercise3/CLAUDE.md. This module never calls a model;
it only classifies structured fields that some other stage (the agent) has
already extracted.

For educational use only — not a clinical decision tool.
"""

from typing import Optional

VALID_IHC_SCORES = ("0", "1+", "2+", "3+")


def _validate_ihc_score(ihc_score: object) -> str:
    if not isinstance(ihc_score, str):
        raise ValueError(f"ihc_score must be a string, got {type(ihc_score).__name__}")
    cleaned = ihc_score.strip()
    if cleaned not in VALID_IHC_SCORES:
        raise ValueError(
            f"ihc_score must be one of {VALID_IHC_SCORES}, got {ihc_score!r}"
        )
    return cleaned


def _validate_ish_value(name: str, value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number or None, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    return float(value)


def score_her2_status(
    ihc_score: str,
    ish_ratio: Optional[float] = None,
    ish_copy_number: Optional[float] = None,
) -> dict:
    """
    Classify HER2 status from structured IHC/ISH fields per ASCO/CAP 2018.

    Raises ValueError on an invalid ihc_score, non-numeric/negative ISH
    values, or only one of ish_ratio/ish_copy_number being provided.
    """
    ihc = _validate_ihc_score(ihc_score)
    ratio = _validate_ish_value("ish_ratio", ish_ratio)
    copy_number = _validate_ish_value("ish_copy_number", ish_copy_number)

    if (ratio is None) != (copy_number is None):
        raise ValueError(
            "ish_ratio and ish_copy_number must be provided together, or not at all"
        )

    if ihc in ("0", "1+"):
        return {
            "status": "Negative",
            "rationale": f"IHC {ihc} is Negative regardless of ISH.",
        }

    if ihc == "3+":
        return {
            "status": "Positive",
            "rationale": "IHC 3+ is Positive regardless of ISH.",
        }

    # ihc == "2+"
    if ratio is None:
        return {
            "status": "Equivocal",
            "rationale": "IHC 2+ with no ISH result is Equivocal; reflex to ISH/FISH recommended.",
        }

    if ratio >= 2.0 and copy_number >= 4.0:
        return {
            "status": "Positive",
            "rationale": "ISH group 1 (ratio >= 2.0, avg copies >= 4.0) is Positive.",
        }
    if ratio >= 2.0 and copy_number < 4.0:
        return {
            "status": "Negative",
            "rationale": "ISH group 2 (ratio >= 2.0, avg copies < 4.0) is Negative when IHC isn't 3+.",
        }
    if ratio < 2.0 and copy_number >= 6.0:
        return {
            "status": "Negative",
            "rationale": (
                "ISH group 3 (ratio < 2.0, avg copies >= 6.0) is Negative when IHC isn't "
                "3+; recount recommended."
            ),
        }
    if ratio < 2.0 and 4.0 <= copy_number < 6.0:
        return {
            "status": "Equivocal",
            "rationale": (
                "ISH group 4 (ratio < 2.0, avg copies 4.0-6.0) is Equivocal when IHC isn't "
                "clearly 3+."
            ),
        }
    # ratio < 2.0 and copy_number < 4.0
    return {
        "status": "Negative",
        "rationale": "ISH group 5 (ratio < 2.0, avg copies < 4.0) is Negative.",
    }


HER2_SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "score_her2_status",
        "description": (
            "Deterministically classify HER2 status per the ASCO/CAP 2018 guideline "
            "from structured IHC/ISH fields extracted from a pathology report. Call "
            "this after extracting the fields — never compute the classification "
            "yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ihc_score": {
                    "type": "string",
                    "enum": list(VALID_IHC_SCORES),
                    "description": "Immunohistochemistry HER2 score as reported.",
                },
                "ish_ratio": {
                    "type": ["number", "null"],
                    "description": "HER2/CEP17 ratio from dual-probe ISH, if reported.",
                },
                "ish_copy_number": {
                    "type": ["number", "null"],
                    "description": "Average HER2 copies per cell from ISH, if reported.",
                },
            },
            "required": ["ihc_score"],
        },
    },
}
