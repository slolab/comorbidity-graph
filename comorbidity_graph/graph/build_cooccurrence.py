"""Compute N, n_i, n_ij from a patient × ICD-10 3-char incidence matrix.

Given a binary incidence matrix (patients × codes), compute:
  - N:    total number of patients
  - n_i:  number of patients with each code
  - n_ij: number of patients with both code i and code j
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse


def build_cooccurrence(
    incidence: csr_matrix | np.ndarray,
    codes: np.ndarray,
) -> tuple[int, pd.Series, pd.DataFrame]:
    """Compute co-occurrence statistics from a binary incidence matrix.

    Parameters
    ----------
    incidence
        Binary (patients × codes) matrix — sparse or dense.
    codes
        Sorted ICD-10 3-char codes (column labels matching *incidence* columns).

    Returns
    -------
    N : int
        Total number of patients.
    prevalence : pd.Series
        Index = code, value = patient count (n_i).
    cooccurrence : pd.DataFrame
        Long-format edge list with columns ``code_i``, ``code_j``, ``n_ij``.
        Only the upper triangle (code_i < code_j) is stored; self-edges are
        excluded.
    """
    if issparse(incidence):
        mat = incidence
    else:
        mat = csr_matrix(incidence)

    N = mat.shape[0]

    # n_i: column sums
    n_i = np.asarray(mat.sum(axis=0)).ravel()
    prevalence = pd.Series(n_i, index=codes, name="n_i")

    # n_ij = X^T @ X  (symmetric, integer-valued)
    cooc_mat = (mat.T @ mat).toarray()

    # Extract upper triangle (exclude diagonal = self-edges)
    i_idx, j_idx = np.triu_indices(len(codes), k=1)
    n_ij = cooc_mat[i_idx, j_idx]

    # Keep only pairs with at least one co-occurrence for sparsity
    mask = n_ij > 0
    cooccurrence = pd.DataFrame({
        "code_i": codes[i_idx[mask]],
        "code_j": codes[j_idx[mask]],
        "n_ij": n_ij[mask],
    })

    return N, prevalence, cooccurrence
