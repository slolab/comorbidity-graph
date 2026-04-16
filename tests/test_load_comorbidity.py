"""Tests for comorbidity_graph.ingest.load_comorbidity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from comorbidity_graph.ingest.load_comorbidity import (
    MIMIC_DEMO_DIAGNOSES,
    build_incidence_matrix,
    load_mimic_diagnoses,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MIMIC_PATH = DATA_DIR / MIMIC_DEMO_DIAGNOSES


@pytest.fixture(scope="module")
def long_df():
    return load_mimic_diagnoses(MIMIC_PATH)


@pytest.fixture(scope="module")
def incidence_triple(long_df):
    return build_incidence_matrix(long_df)


class TestLoadMimicDiagnoses:
    def test_columns(self, long_df):
        assert list(long_df.columns) == ["subject_id", "icd3"]

    def test_no_duplicates(self, long_df):
        assert long_df.duplicated().sum() == 0

    def test_icd3_format(self, long_df):
        # ICD-10-CM allows alphanumeric 3rd char (e.g. C7A, C7B)
        assert long_df["icd3"].str.match(r"^[A-Z][0-9A-Z]{2}$").all()

    def test_patients_are_subset_of_demo(self, long_df):
        demo_ids = set(
            __import__("pandas")
            .read_csv(DATA_DIR / "mimic-iv-clinical-database-demo-2.2/demo_subject_id.csv")
            ["subject_id"]
        )
        assert set(long_df["subject_id"].unique()) <= demo_ids

    def test_default_path(self):
        df = load_mimic_diagnoses(data_dir=DATA_DIR)
        assert len(df) > 0
        assert list(df.columns) == ["subject_id", "icd3"]


class TestBuildIncidenceMatrix:
    def test_shape(self, long_df, incidence_triple):
        incidence, patients, codes = incidence_triple
        assert incidence.shape == (len(patients), len(codes))
        assert len(patients) == long_df["subject_id"].nunique()
        assert len(codes) == long_df["icd3"].nunique()

    def test_binary(self, incidence_triple):
        incidence, _, _ = incidence_triple
        vals = incidence.data
        assert set(np.unique(vals)) <= {0, 1}

    def test_sorted_labels(self, incidence_triple):
        _, patients, codes = incidence_triple
        assert list(patients) == sorted(patients)
        assert list(codes) == sorted(codes)

    def test_row_sums_positive(self, incidence_triple):
        incidence, _, _ = incidence_triple
        row_sums = np.asarray(incidence.sum(axis=1)).ravel()
        assert (row_sums > 0).all()
