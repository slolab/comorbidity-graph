"""Parse prediction-token strings into ICD-10 3-char code fields.

Examples:
    "I50 Heart failure"        -> code="I50", desc="Heart failure",  category="icd10"
    "Female"                   -> code=None,  desc=None,             category="non_disease"
    "BMI low"                  -> code=None,  desc=None,             category="non_disease"
    "CXX Unknown Cancer"       -> code=None,  desc=None,             category="non_disease"
    "XYZ weird"                -> code=None,  desc=None,             category="unknown"

``is_disease`` is True iff ``category == "icd10"``.
"""

from __future__ import annotations

import re
from typing import TypedDict

import pandas as pd


# Matches an ICD-10 3-char block code (letter + 2 digits), optionally followed
# by a space and a free-text description.
_ICD10_PATTERN = re.compile(r"^([A-Z]\d{2})(?:\s+(.*))?$")

# Known non-disease prediction tokens. Confirmed empirically by scanning all
# 4 CSVs in data/ — exactly these 5 appear and nothing else.
NON_DISEASE_TOKENS: frozenset[str] = frozenset(
    {
        "Female",
        "Male",
        "BMI low",
        "BMI mid",
        "CXX Unknown Cancer",
    }
)


class NormalizedToken(TypedDict):
    pred_code: str | None
    pred_description: str | None
    is_disease: bool
    category: str  # "icd10" | "non_disease" | "unknown"


def normalize_token(token: str) -> NormalizedToken:
    """Classify a single prediction token.

    Check the non-disease allowlist first so that hypothetical future tokens
    starting with `[A-Z]\\d{2}` by accident (none observed so far) don't shadow
    an explicit demographic entry.
    """
    if token in NON_DISEASE_TOKENS:
        return {
            "pred_code": None,
            "pred_description": None,
            "is_disease": False,
            "category": "non_disease",
        }
    m = _ICD10_PATTERN.match(token)
    if m is not None:
        return {
            "pred_code": m.group(1),
            "pred_description": m.group(2) or "",
            "is_disease": True,
            "category": "icd10",
        }
    return {
        "pred_code": None,
        "pred_description": None,
        "is_disease": False,
        "category": "unknown",
    }


def normalize_tokens(tokens: pd.Series) -> pd.DataFrame:
    """Apply ``normalize_token`` to every element of ``tokens``.

    Returns a DataFrame with columns ``pred_code``, ``pred_description``,
    ``is_disease``, ``category`` and the same index as ``tokens``.
    """
    records = [normalize_token(t) for t in tokens]
    return pd.DataFrame.from_records(records, index=tokens.index)
