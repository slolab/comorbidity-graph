"""Tests for comorbidity_graph.graph (co-occurrence + distance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from comorbidity_graph.graph.build_cooccurrence import build_cooccurrence
from comorbidity_graph.graph.build_distance import build_distance, build_distance_matrix


@pytest.fixture
def toy_incidence():
    """Three patients, four codes — hand-verifiable co-occurrence.

    Patient 0: A00, B00
    Patient 1: A00, B00, C00
    Patient 2: B00, C00, D00
    """
    codes = np.array(["A00", "B00", "C00", "D00"])
    data = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
    ], dtype=np.int8)
    return csr_matrix(data), codes


class TestBuildCooccurrence:
    def test_N(self, toy_incidence):
        incidence, codes = toy_incidence
        N, _, _ = build_cooccurrence(incidence, codes)
        assert N == 3

    def test_prevalence(self, toy_incidence):
        incidence, codes = toy_incidence
        _, prevalence, _ = build_cooccurrence(incidence, codes)
        assert prevalence["A00"] == 2
        assert prevalence["B00"] == 3
        assert prevalence["C00"] == 2
        assert prevalence["D00"] == 1

    def test_cooccurrence_values(self, toy_incidence):
        incidence, codes = toy_incidence
        _, _, cooc = build_cooccurrence(incidence, codes)
        lookup = {(r.code_i, r.code_j): r.n_ij for _, r in cooc.iterrows()}
        assert lookup[("A00", "B00")] == 2
        assert lookup[("B00", "C00")] == 2
        assert lookup[("A00", "C00")] == 1
        assert lookup[("C00", "D00")] == 1
        assert lookup[("B00", "D00")] == 1
        # A00-D00 never co-occur
        assert ("A00", "D00") not in lookup

    def test_no_self_edges(self, toy_incidence):
        incidence, codes = toy_incidence
        _, _, cooc = build_cooccurrence(incidence, codes)
        assert not (cooc["code_i"] == cooc["code_j"]).any()


class TestBuildDistance:
    def test_distance_symmetric_and_finite(self, toy_incidence):
        incidence, codes = toy_incidence
        N, prevalence, cooc = build_cooccurrence(incidence, codes)
        edges = build_distance(N, prevalence, cooc)
        assert np.isfinite(edges["distance"]).all()
        assert (edges["distance"] > 0).all()

    def test_distance_decreases_with_cooccurrence(self, toy_incidence):
        incidence, codes = toy_incidence
        N, prevalence, cooc = build_cooccurrence(incidence, codes)
        edges = build_distance(N, prevalence, cooc)
        lookup = {(r.code_i, r.code_j): r.distance for _, r in edges.iterrows()}
        # A00-B00 (n_ij=2) should be closer than A00-C00 (n_ij=1)
        assert lookup[("A00", "B00")] < lookup[("A00", "C00")]

    def test_smoothed_probability(self, toy_incidence):
        incidence, codes = toy_incidence
        N, prevalence, cooc = build_cooccurrence(incidence, codes)
        edges = build_distance(N, prevalence, cooc, alpha=1.0, beta=2.0)
        row = edges[
            (edges["code_i"] == "A00") & (edges["code_j"] == "B00")
        ].iloc[0]
        expected_p = (2 + 1.0) / (3 + 2.0)
        assert abs(row["p_ij_smoothed"] - expected_p) < 1e-10
        assert abs(row["distance"] - (-np.log(expected_p))) < 1e-10


class TestBuildDistanceMatrix:
    def test_symmetric_and_zero_diagonal(self, toy_incidence):
        incidence, codes = toy_incidence
        N, prevalence, cooc = build_cooccurrence(incidence, codes)
        edges = build_distance(N, prevalence, cooc)
        mat = build_distance_matrix(edges, codes, N=N)
        np.testing.assert_array_almost_equal(mat.values, mat.values.T)
        np.testing.assert_array_equal(np.diag(mat.values), 0.0)

    def test_unobserved_pair_gets_floor_distance(self, toy_incidence):
        incidence, codes = toy_incidence
        N, prevalence, cooc = build_cooccurrence(incidence, codes)
        edges = build_distance(N, prevalence, cooc, alpha=1.0, beta=2.0)
        mat = build_distance_matrix(edges, codes, N=N, alpha=1.0, beta=2.0)
        # A00 and D00 never co-occur
        floor = -np.log(1.0 / (3 + 2.0))
        assert abs(mat.loc["A00", "D00"] - floor) < 1e-10

    def test_observed_pair_matches_edge(self, toy_incidence):
        incidence, codes = toy_incidence
        N, prevalence, cooc = build_cooccurrence(incidence, codes)
        edges = build_distance(N, prevalence, cooc)
        mat = build_distance_matrix(edges, codes, N=N)
        row = edges[
            (edges["code_i"] == "A00") & (edges["code_j"] == "B00")
        ].iloc[0]
        assert abs(mat.loc["A00", "B00"] - row["distance"]) < 1e-10
