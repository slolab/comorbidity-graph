"""Load MIMIC-IV diagnoses_icd.csv into patient x ICD-10 3-char incidence.

Reads the gzipped diagnoses table, filters to ICD-10 rows (icd_version == 10),
truncates codes to 3-character blocks, deduplicates per patient, and returns
both the long-format table and a sparse binary incidence matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


MIMIC_DEMO_DIAGNOSES = (
    "mimic-iv-clinical-database-demo-2.2/hosp/diagnoses_icd.csv.gz"
)


def load_mimic_diagnoses(
    path: str | Path | None = None,
    *,
    data_dir: str | Path = "data",
) -> pd.DataFrame:
    """Load MIMIC-IV diagnoses_icd.csv restricted to ICD-10 rows.

    Parameters
    ----------
    path
        Explicit path to the ``diagnoses_icd.csv(.gz)`` file. When *None*,
        defaults to ``data_dir / MIMIC_DEMO_DIAGNOSES``.
    data_dir
        Root data directory (used only when *path* is None).

    Returns
    -------
    pd.DataFrame
        Columns: ``subject_id``, ``icd3`` (first 3 characters of
        ``icd_code`` for ICD-10 rows).  Deduplicated per patient so each
        (subject_id, icd3) pair appears at most once.
    """
    if path is None:
        path = Path(data_dir) / MIMIC_DEMO_DIAGNOSES
    else:
        path = Path(path)

    raw = pd.read_csv(path)
    icd10 = raw.loc[raw["icd_version"] == 10, ["subject_id", "icd_code"]].copy()
    icd10["icd3"] = icd10["icd_code"].str[:3]
    long = icd10[["subject_id", "icd3"]].drop_duplicates().reset_index(drop=True)
    return long


def build_incidence_matrix(
    long: pd.DataFrame,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Convert a long-format (subject_id, icd3) table to a sparse binary matrix.

    Parameters
    ----------
    long
        DataFrame with columns ``subject_id`` and ``icd3``, as returned by
        :func:`load_mimic_diagnoses`.

    Returns
    -------
    incidence : csr_matrix
        Binary (patients × codes) matrix.
    patients : np.ndarray
        Sorted patient IDs (row labels).
    codes : np.ndarray
        Sorted ICD-10 3-char codes (column labels).
    """
    patients = np.sort(long["subject_id"].unique())
    codes = np.sort(long["icd3"].unique())

    pat_idx = {pid: i for i, pid in enumerate(patients)}
    code_idx = {c: j for j, c in enumerate(codes)}

    rows = long["subject_id"].map(pat_idx).values
    cols = long["icd3"].map(code_idx).values

    incidence = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(len(patients), len(codes)),
    )
    return incidence, patients, codes
