"""Tests for comorbidity_graph.mapping.normalize_icd10."""

from __future__ import annotations

import pandas as pd
import pytest

from comorbidity_graph.mapping.normalize_icd10 import (
    NON_DISEASE_TOKENS,
    normalize_token,
    normalize_tokens,
)


class TestNormalizeToken:
    def test_icd10_block_with_description(self):
        r = normalize_token("I50 Heart failure")
        assert r == {
            "pred_code": "I50",
            "pred_description": "Heart failure",
            "is_disease": True,
            "category": "icd10",
        }

    def test_icd10_block_description_with_commas(self):
        # One of the predicted tokens contains commas after the code.
        r = normalize_token(
            "C72 Malignant neoplasm of spinal cord, cranial nerves and other parts"
        )
        assert r["pred_code"] == "C72"
        assert r["is_disease"] is True
        assert "," in r["pred_description"]

    def test_icd10_code_only(self):
        r = normalize_token("J22")
        assert r == {
            "pred_code": "J22",
            "pred_description": "",
            "is_disease": True,
            "category": "icd10",
        }

    @pytest.mark.parametrize("tok", sorted(NON_DISEASE_TOKENS))
    def test_known_non_disease(self, tok):
        r = normalize_token(tok)
        assert r["is_disease"] is False
        assert r["category"] == "non_disease"
        assert r["pred_code"] is None

    def test_unknown_token(self):
        r = normalize_token("banana")
        assert r == {
            "pred_code": None,
            "pred_description": None,
            "is_disease": False,
            "category": "unknown",
        }

    def test_cxx_not_confused_with_icd10(self):
        # "CXX Unknown Cancer" has two X's, not digits — must NOT match ICD-10
        # regex, and must be classified as non_disease (it's in the allowlist).
        r = normalize_token("CXX Unknown Cancer")
        assert r["category"] == "non_disease"
        assert r["pred_code"] is None

    def test_bmi_tokens(self):
        for tok in ("BMI low", "BMI mid"):
            r = normalize_token(tok)
            assert r["category"] == "non_disease"


class TestNormalizeTokens:
    def test_vectorized_matches_scalar(self):
        tokens = pd.Series(
            ["I50 Heart failure", "Female", "banana", "E11 Diabetes"]
        )
        df = normalize_tokens(tokens)
        assert list(df.columns) == [
            "pred_code",
            "pred_description",
            "is_disease",
            "category",
        ]
        assert df["category"].tolist() == [
            "icd10",
            "non_disease",
            "unknown",
            "icd10",
        ]
        # pandas promotes None to NaN inside object columns with mixed str/None.
        assert df["pred_code"].iloc[0] == "I50"
        assert df["pred_code"].iloc[3] == "E11"
        assert df["pred_code"].iloc[[1, 2]].isna().all()
        assert df["is_disease"].tolist() == [True, False, False, True]

    def test_preserves_index(self):
        tokens = pd.Series(
            {"a": "I50 Heart failure", "b": "Female"},
        )
        df = normalize_tokens(tokens)
        assert df.index.tolist() == ["a", "b"]
