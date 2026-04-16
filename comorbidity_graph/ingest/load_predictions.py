"""Load the 4 prediction CSVs into a single long-format DataFrame.

Layout of `data/`:
    csae_stc_matched_control.csv                 - SAE,    matched_control, set_to_case
    delphi_stc_matched_control.csv               - Delphi, matched_control, set_to_case
    sae_sth_separate_control.csv                 - SAE,    separate_control, set_to_healthy
    delphi_token_removal_sth_separate_control.csv- Delphi, separate_control, token_removal

Each CSV has one row per perturbed disease with a `top_k_tokens` JSON list of
10 affected tokens and a parallel `top_k_shifts` list of signed floats. This
loader unpivots those lists so every rank becomes its own row.

Expected output: (30 + 30 + 18 + 18) * 10 = 960 rows.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TypedDict

import pandas as pd

from comorbidity_graph.mapping.normalize_icd10 import normalize_tokens


class _FileSpec(TypedDict):
    filename: str
    method: str
    set_label: str


# SAE plays the role of `guided`; Delphi plays `naive`. Confirmed with user.
ROLE_BY_METHOD: dict[str, str] = {"SAE": "guided", "Delphi": "naive"}

FILE_SPECS: list[_FileSpec] = [
    {
        "filename": "csae_stc_matched_control.csv",
        "method": "SAE",
        "set_label": "matched_control",
    },
    {
        "filename": "delphi_stc_matched_control.csv",
        "method": "Delphi",
        "set_label": "matched_control",
    },
    {
        "filename": "sae_sth_separate_control.csv",
        "method": "SAE",
        "set_label": "separate_control",
    },
    {
        "filename": "delphi_token_removal_sth_separate_control.csv",
        "method": "Delphi",
        "set_label": "separate_control",
    },
]


# Parses the perturbed `token_name`, e.g. "I50 Heart failure".
_PERTURBED_PATTERN = re.compile(r"^([A-Z]\d{2})\s+(.*)$")


def _parse_perturbed(token_name: str) -> tuple[str, str]:
    """Split a perturbed-row ``token_name`` into (code, description).

    All perturbed tokens observed in the 4 CSVs match this pattern; if one
    ever doesn't, fail loudly rather than silently produce a bad row_key.
    """
    m = _PERTURBED_PATTERN.match(token_name)
    if m is None:
        raise ValueError(
            f"perturbed token_name does not match ICD-10 3-char pattern: "
            f"{token_name!r}"
        )
    return m.group(1), m.group(2)


def _load_one(path: Path, method: str, set_label: str) -> pd.DataFrame:
    """Load one CSV and return its unpivoted long-format frame."""
    raw = pd.read_csv(path)

    # Parse the two JSON-array columns. ``ast.literal_eval`` handles both
    # single- and double-quoted list literals (sae_sth_separate_control.csv
    # uses single quotes; the others use double quotes).
    raw["top_k_tokens"] = raw["top_k_tokens"].map(ast.literal_eval)
    raw["top_k_shifts"] = raw["top_k_shifts"].map(ast.literal_eval)

    # Sanity check: the two lists must have equal length per row.
    bad = raw[raw["top_k_tokens"].map(len) != raw["top_k_shifts"].map(len)]
    if not bad.empty:
        raise ValueError(
            f"{path.name}: top_k_tokens and top_k_shifts length mismatch at rows "
            f"{bad.index.tolist()}"
        )

    # Unpivot both list columns in lock-step using .explode().
    long = raw.reset_index(drop=True).copy()
    long["rank"] = long["top_k_tokens"].map(
        lambda xs: list(range(1, len(xs) + 1))
    )
    long = long.explode(
        ["top_k_tokens", "top_k_shifts", "rank"], ignore_index=True
    )
    long = long.rename(
        columns={"top_k_tokens": "pred_token", "top_k_shifts": "pred_shift"}
    )

    # Parse the perturbed `token_name` into code + description.
    perturbed = long["token_name"].map(_parse_perturbed)
    long["perturbed_code"] = perturbed.map(lambda t: t[0])
    long["perturbed_desc"] = perturbed.map(lambda t: t[1])

    # Decorate with method/set/role.
    long["method"] = method
    long["set"] = set_label
    long["role"] = ROLE_BY_METHOD[method]
    long["row_key"] = set_label + "|" + long["token_name"]

    # Normalize prediction token into (pred_code, pred_description,
    # is_disease, category).
    norm = normalize_tokens(long["pred_token"])
    long = pd.concat([long, norm], axis=1)

    # Cast dtypes.
    long["rank"] = long["rank"].astype("int64")
    long["pred_shift"] = long["pred_shift"].astype("float64")
    long["specificity"] = long["specificity"].astype("float64")

    long = long.rename(columns={"chapter": "perturbed_chapter"})

    # Final column order.
    cols = [
        "row_key",
        "set",
        "method",
        "role",
        "steer_mode",
        "perturbed_code",
        "perturbed_desc",
        "perturbed_chapter",
        "specificity",
        "rank",
        "pred_token",
        "pred_code",
        "pred_description",
        "pred_shift",
        "is_disease",
        "category",
    ]
    return long[cols]


def load_predictions(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load all 4 prediction CSVs into one long-format DataFrame.

    Parameters
    ----------
    data_dir
        Directory containing the 4 CSVs listed in :data:`FILE_SPECS`.

    Returns
    -------
    pandas.DataFrame
        One row per (CSV row, rank) pair — 960 rows total for the current
        data drop. Columns are documented in the plan file at
        ``/Users/sebastian.lobentanzer/.claude/plans/soft-growing-torvalds.md``
        section "Canonical long-format prediction table".
    """
    data_dir = Path(data_dir)
    parts = [
        _load_one(
            data_dir / spec["filename"],
            method=spec["method"],
            set_label=spec["set_label"],
        )
        for spec in FILE_SPECS
    ]
    return pd.concat(parts, ignore_index=True)
