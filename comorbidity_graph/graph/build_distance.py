"""Convert co-occurrence counts into a symmetric distance matrix.

Distance is defined as ``d(i,j) = -log((n_ij + alpha) / (N + beta))`` with
configurable smoothing constants.  Smaller distance = more co-occurrence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_distance(
    N: int,
    prevalence: pd.Series,
    cooccurrence: pd.DataFrame,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> pd.DataFrame:
    """Build the weighted distance edge list from co-occurrence statistics.

    Parameters
    ----------
    N
        Total number of patients.
    prevalence
        Series indexed by code with patient counts (n_i).
    cooccurrence
        Long-format edge list with columns ``code_i``, ``code_j``, ``n_ij``.
    alpha, beta
        Laplace smoothing constants for the co-occurrence probability.

    Returns
    -------
    pd.DataFrame
        Edge list with columns: ``code_i``, ``code_j``, ``n_i``, ``n_j``,
        ``n_ij``, ``p_ij_smoothed``, ``distance``.
    """
    edges = cooccurrence.copy()
    edges["n_i"] = edges["code_i"].map(prevalence).values
    edges["n_j"] = edges["code_j"].map(prevalence).values
    edges["p_ij_smoothed"] = (edges["n_ij"] + alpha) / (N + beta)
    edges["distance"] = -np.log(edges["p_ij_smoothed"])

    return edges[["code_i", "code_j", "n_i", "n_j", "n_ij",
                   "p_ij_smoothed", "distance"]]


def build_distance_matrix(
    edges: pd.DataFrame,
    codes: np.ndarray,
    *,
    N: int,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> pd.DataFrame:
    """Build a dense symmetric distance matrix from the edge list.

    Pairs absent from ``edges`` (zero observed co-occurrence) get their
    distance from the smoothing floor: ``-log(alpha / (N + beta))``.

    Self-distance is 0.

    Parameters
    ----------
    edges
        Edge list as returned by :func:`build_distance`.
    codes
        All ICD-10 3-char codes to include in the matrix.
    N
        Total patient count (for smoothing unobserved pairs).
    alpha, beta
        Smoothing constants (must match those used in :func:`build_distance`).

    Returns
    -------
    pd.DataFrame
        Square symmetric DataFrame indexed and columned by *codes*.
    """
    floor_distance = -np.log(alpha / (N + beta))
    n = len(codes)
    mat = np.full((n, n), floor_distance)
    np.fill_diagonal(mat, 0.0)

    code_to_idx = {c: i for i, c in enumerate(codes)}
    for _, row in edges.iterrows():
        ci = row["code_i"]
        cj = row["code_j"]
        if ci in code_to_idx and cj in code_to_idx:
            i, j = code_to_idx[ci], code_to_idx[cj]
            mat[i, j] = row["distance"]
            mat[j, i] = row["distance"]

    return pd.DataFrame(mat, index=codes, columns=codes)
