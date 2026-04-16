"""Tests for scoring, per-example evaluation, and aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comorbidity_graph.score.class_distance import (
    pair_distance,
    pair_distances_vectorized,
)
from comorbidity_graph.score.score_topk import score_topk
from comorbidity_graph.evaluation.per_example import build_per_example
from comorbidity_graph.evaluation.aggregate import aggregate


@pytest.fixture
def distance_matrix():
    """Small 4x4 distance matrix for testing."""
    codes = ["A00", "B00", "C00", "D00"]
    mat = pd.DataFrame(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0],
        ],
        index=codes,
        columns=codes,
    )
    return mat


class TestPairDistance:
    def test_known_pair(self, distance_matrix):
        assert pair_distance("A00", "B00", distance_matrix) == 1.0

    def test_self_distance(self, distance_matrix):
        assert pair_distance("A00", "A00", distance_matrix) == 0.0

    def test_missing_code(self, distance_matrix):
        assert np.isnan(pair_distance("Z99", "A00", distance_matrix))

    def test_symmetric(self, distance_matrix):
        d1 = pair_distance("A00", "C00", distance_matrix)
        d2 = pair_distance("C00", "A00", distance_matrix)
        assert d1 == d2


class TestPairDistancesVectorized:
    def test_basic(self, distance_matrix):
        perturbed = pd.Series(["A00", "A00", "B00"])
        predicted = pd.Series(["B00", "C00", "D00"])
        result = pair_distances_vectorized(perturbed, predicted, distance_matrix)
        expected = [1.0, 2.0, 2.5]
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_missing_returns_nan(self, distance_matrix):
        perturbed = pd.Series(["A00", "Z99"])
        predicted = pd.Series(["B00", "A00"])
        result = pair_distances_vectorized(perturbed, predicted, distance_matrix)
        assert result.iloc[0] == 1.0
        assert np.isnan(result.iloc[1])


@pytest.fixture
def toy_long_df():
    """Minimal long-format prediction table for scoring tests.

    Two row_keys, each with guided (SAE) and naive (Delphi).
    Perturbed code is A00 for both.
    """
    rows = []
    for row_key, steer_mode in [("key1", "set_to_case"), ("key2", "set_to_healthy")]:
        for role, method in [("guided", "SAE"), ("naive", "Delphi")]:
            for rank, (pred_code, shift) in enumerate(
                [("B00", 0.5), ("C00", -0.3), ("D00", 0.1)], start=1
            ):
                rows.append({
                    "row_key": row_key,
                    "role": role,
                    "method": method,
                    "steer_mode": steer_mode,
                    "perturbed_code": "A00",
                    "pred_code": pred_code,
                    "pred_token": f"{pred_code} Test",
                    "pred_shift": shift,
                    "is_disease": True,
                    "rank": rank,
                    "set": "test",
                    "category": "icd10",
                })
    return pd.DataFrame(rows)


class TestScoreTopk:
    def test_output_shape(self, toy_long_df, distance_matrix):
        scored = score_topk(toy_long_df, distance_matrix)
        # 2 row_keys × 2 roles = 4 rows
        assert len(scored) == 4
        assert "score_unsigned" in scored.columns
        assert "score_sign_aware" in scored.columns
        assert "score_magnitude" in scored.columns

    def test_unsigned_score(self, toy_long_df, distance_matrix):
        scored = score_topk(toy_long_df, distance_matrix)
        row = scored[
            (scored["row_key"] == "key1") & (scored["role"] == "guided")
        ].iloc[0]
        # Distances from A00: B00=1.0, C00=2.0, D00=3.0  ->  mean = 2.0
        assert abs(row["score_unsigned"] - 2.0) < 1e-10
        assert row["effective_k_unsigned"] == 3


class TestPerExample:
    def test_output_structure(self, toy_long_df, distance_matrix):
        scored = score_topk(toy_long_df, distance_matrix)
        per_ex = build_per_example(scored)
        assert len(per_ex) == 2
        assert "delta_unsigned" in per_ex.columns
        assert "winner_unsigned" in per_ex.columns

    def test_tie_when_same_scores(self, toy_long_df, distance_matrix):
        scored = score_topk(toy_long_df, distance_matrix)
        per_ex = build_per_example(scored)
        # guided and naive have identical predictions, so delta=0 -> tie
        assert (per_ex["winner_unsigned"] == "tie").all()


class TestAggregate:
    def test_overall_keys(self, toy_long_df, distance_matrix):
        scored = score_topk(toy_long_df, distance_matrix)
        per_ex = build_per_example(scored)
        summaries = aggregate(per_ex)
        assert "overall" in summaries
        assert "by_steer_mode" in summaries
        overall = summaries["overall"]
        assert "unsigned" in overall["variant"].values

    def test_tie_rate_is_one_when_identical(self, toy_long_df, distance_matrix):
        scored = score_topk(toy_long_df, distance_matrix)
        per_ex = build_per_example(scored)
        summaries = aggregate(per_ex)
        overall = summaries["overall"]
        row = overall[overall["variant"] == "unsigned"].iloc[0]
        assert row["tie_rate"] == 1.0
