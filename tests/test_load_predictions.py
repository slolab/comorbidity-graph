"""Tests for comorbidity_graph.ingest.load_predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from comorbidity_graph.ingest.load_predictions import (
    FILE_SPECS,
    ROLE_BY_METHOD,
    load_predictions,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def long_df() -> pd.DataFrame:
    return load_predictions(DATA_DIR)


def test_row_count_is_960(long_df):
    # (30 matched_control * 2 methods + 18 separate_control * 2 methods) * 10.
    assert len(long_df) == (30 + 30 + 18 + 18) * 10 == 960


def test_expected_columns(long_df):
    assert list(long_df.columns) == [
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


def test_role_mapping(long_df):
    assert ROLE_BY_METHOD == {"SAE": "guided", "Delphi": "naive"}
    roles_by_method = (
        long_df.groupby("method")["role"].agg(set).to_dict()
    )
    assert roles_by_method == {"SAE": {"guided"}, "Delphi": {"naive"}}


def test_sets_and_methods_present(long_df):
    assert set(long_df["set"].unique()) == {"matched_control", "separate_control"}
    assert set(long_df["method"].unique()) == {"SAE", "Delphi"}
    # Each (set, method) combination must exist.
    pairs = set(zip(long_df["set"], long_df["method"]))
    expected = {
        ("matched_control", "SAE"),
        ("matched_control", "Delphi"),
        ("separate_control", "SAE"),
        ("separate_control", "Delphi"),
    }
    assert pairs == expected


def test_steer_modes_match_method_and_set(long_df):
    combos = (
        long_df.groupby(["set", "method"])["steer_mode"].agg(set).to_dict()
    )
    assert combos == {
        ("matched_control", "SAE"): {"set_to_case"},
        ("matched_control", "Delphi"): {"set_to_case"},
        ("separate_control", "SAE"): {"set_to_healthy"},
        ("separate_control", "Delphi"): {"token_removal"},
    }


def test_rank_is_1_to_10(long_df):
    # Every (row_key, method) group has ranks 1..10 exactly.
    per_row = long_df.groupby(["row_key", "method"])["rank"].agg(list)
    for ranks in per_row:
        assert ranks == list(range(1, 11))


def test_paired_token_names_identical(long_df):
    # For each set, SAE and Delphi must share the same set of perturbed tokens
    # (verified via shell diff during exploration; re-asserted here).
    for set_label in ("matched_control", "separate_control"):
        sub = long_df[long_df["set"] == set_label]
        by_method = sub.groupby("method")["perturbed_code"].agg(set).to_dict()
        assert by_method["SAE"] == by_method["Delphi"], (
            f"{set_label}: perturbed_code sets differ between SAE and Delphi"
        )


def test_perturbed_code_format(long_df):
    # All perturbed codes must be 3-char ICD-10 (letter + 2 digits).
    assert long_df["perturbed_code"].str.match(r"^[A-Z]\d{2}$").all()


def test_non_disease_tokens_present(long_df):
    # At least the 5 empirically observed non-disease tokens should be
    # categorized as non_disease.
    non_dis = set(
        long_df.loc[long_df["category"] == "non_disease", "pred_token"].unique()
    )
    assert {"Female", "Male", "BMI low", "BMI mid", "CXX Unknown Cancer"} <= non_dis


def test_no_unknown_tokens_in_current_data(long_df):
    # If any token falls into "unknown", the allowlist should probably grow.
    # Today's data has exactly 5 non-disease tokens and the rest ICD-10 — no
    # unknowns. If this ever fails, inspect the `unknown` set and decide.
    unknown = set(
        long_df.loc[long_df["category"] == "unknown", "pred_token"].unique()
    )
    assert unknown == set(), f"unexpected unknown tokens: {unknown}"


def test_pred_shifts_numeric(long_df):
    assert pd.api.types.is_float_dtype(long_df["pred_shift"])
    assert long_df["pred_shift"].notna().all()


def test_row_key_uniquely_identifies_pair(long_df):
    # For a given (row_key, method, rank), exactly one row exists.
    groups = long_df.groupby(["row_key", "method", "rank"]).size()
    assert (groups == 1).all()


def test_file_specs_all_exist():
    for spec in FILE_SPECS:
        assert (DATA_DIR / spec["filename"]).is_file()
