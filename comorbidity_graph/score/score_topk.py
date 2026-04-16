"""Score top-k prediction sets with three variants side-by-side.

Variants (computed in one pass):
  - unsigned:           mean of d(C0, C) over mappable disease predictions.
  - sign_aware:         filter predictions by expected shift sign per steer_mode,
                        then mean distance.
  - magnitude_weighted: sum |shift| * d / sum |shift| over mappable preds.

Sign-expectation table:
  set_to_case    -> positive shift is expected
  set_to_healthy -> negative shift is expected
  token_removal  -> sign neutral (use all predictions)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from comorbidity_graph.score.class_distance import pair_distances_vectorized

EXPECTED_SIGN: dict[str, str] = {
    "set_to_case": "positive",
    "set_to_healthy": "negative",
    "token_removal": "neutral",
}


def score_topk(
    long_df: pd.DataFrame,
    distance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Score every prediction row and aggregate per (row_key, role).

    Parameters
    ----------
    long_df
        Long-format prediction table from
        :func:`~comorbidity_graph.ingest.load_predictions.load_predictions`.
        Must contain columns: ``row_key``, ``role``, ``steer_mode``,
        ``perturbed_code``, ``pred_code``, ``pred_shift``, ``is_disease``,
        ``rank``.
    distance_matrix
        Symmetric distance matrix as from
        :func:`~comorbidity_graph.graph.build_distance.build_distance_matrix`.

    Returns
    -------
    pd.DataFrame
        One row per (row_key, role) with columns:

        - ``row_key``, ``role``, ``steer_mode``, ``perturbed_code``
        - ``score_unsigned``, ``effective_k_unsigned``
        - ``score_sign_aware``, ``effective_k_sign_aware``
        - ``score_magnitude``, ``effective_k_magnitude``
    """
    df = long_df.copy()

    # Compute distance for every (perturbed, predicted) disease pair
    df["distance"] = np.nan
    disease_mask = df["is_disease"].fillna(False).astype(bool)
    if disease_mask.any():
        df.loc[disease_mask, "distance"] = pair_distances_vectorized(
            df.loc[disease_mask, "perturbed_code"],
            df.loc[disease_mask, "pred_code"],
            distance_matrix,
        ).values

    df["has_distance"] = df["distance"].notna()

    # Sign filter: keep predictions matching expected shift direction
    df["sign_ok"] = True
    for mode, expected in EXPECTED_SIGN.items():
        mask = df["steer_mode"] == mode
        if expected == "positive":
            df.loc[mask, "sign_ok"] = df.loc[mask, "pred_shift"] > 0
        elif expected == "negative":
            df.loc[mask, "sign_ok"] = df.loc[mask, "pred_shift"] < 0
        # "neutral" keeps all

    group_cols = ["row_key", "role", "steer_mode", "perturbed_code"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        usable = g[g["has_distance"]]

        # --- unsigned ---
        k_unsigned = len(usable)
        s_unsigned = usable["distance"].mean() if k_unsigned > 0 else np.nan

        # --- sign_aware ---
        signed = usable[usable["sign_ok"]]
        k_sign = len(signed)
        s_sign = signed["distance"].mean() if k_sign > 0 else np.nan

        # --- magnitude_weighted ---
        abs_shift = usable["pred_shift"].abs()
        total_abs = abs_shift.sum()
        if total_abs > 0:
            s_mag = (abs_shift * usable["distance"]).sum() / total_abs
            k_mag = (abs_shift > 0).sum()
        else:
            s_mag = np.nan
            k_mag = 0

        return pd.Series({
            "score_unsigned": s_unsigned,
            "effective_k_unsigned": k_unsigned,
            "score_sign_aware": s_sign,
            "effective_k_sign_aware": k_sign,
            "score_magnitude": s_mag,
            "effective_k_magnitude": k_mag,
        })

    scored = df.groupby(group_cols, sort=False).apply(_agg).reset_index()

    int_cols = [c for c in scored.columns if c.startswith("effective_k")]
    scored[int_cols] = scored[int_cols].astype(int)

    return scored
