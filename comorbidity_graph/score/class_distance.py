"""Look up the 3-char ICD-10 distance between perturbed and predicted codes.

Because the graph is already at 3-char ICD-10 granularity (matching the token
space), this is a direct matrix lookup — no CCSR expected-pairwise aggregation
needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_distance(
    code_a: str,
    code_b: str,
    distance_matrix: pd.DataFrame,
) -> float:
    """Return the distance between two ICD-10 3-char codes.

    Parameters
    ----------
    code_a, code_b
        ICD-10 3-character block codes (e.g. ``"I50"``, ``"E11"``).
    distance_matrix
        Symmetric distance matrix indexed and columned by ICD-10 3-char codes,
        as returned by :func:`~comorbidity_graph.graph.build_distance.build_distance_matrix`.

    Returns
    -------
    float
        Distance value if both codes are in the matrix, ``np.nan`` otherwise.
    """
    if code_a not in distance_matrix.index or code_b not in distance_matrix.columns:
        return np.nan
    return float(distance_matrix.loc[code_a, code_b])


def pair_distances_vectorized(
    perturbed_codes: pd.Series,
    predicted_codes: pd.Series,
    distance_matrix: pd.DataFrame,
) -> pd.Series:
    """Vectorized distance lookup for aligned Series of code pairs.

    Parameters
    ----------
    perturbed_codes, predicted_codes
        Aligned Series of ICD-10 3-char codes.
    distance_matrix
        Symmetric distance matrix.

    Returns
    -------
    pd.Series
        Distances, with ``np.nan`` where either code is absent from the matrix.
    """
    valid_codes = set(distance_matrix.index)
    mask = perturbed_codes.isin(valid_codes) & predicted_codes.isin(valid_codes)

    result = pd.Series(np.nan, index=perturbed_codes.index)
    if mask.any():
        mat = distance_matrix.values
        code_to_idx = {c: i for i, c in enumerate(distance_matrix.index)}
        row_idx = perturbed_codes[mask].map(code_to_idx).values
        col_idx = predicted_codes[mask].map(code_to_idx).values
        result[mask] = mat[row_idx, col_idx]

    return result
