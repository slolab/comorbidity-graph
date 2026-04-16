"""Aggregate summaries: means, paired deltas, win rates, per-chapter strata.

Consumes the per-example table and produces summary statistics comparing
guided vs naive across all scoring variants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from comorbidity_graph.evaluation.per_example import SCORE_VARIANTS


def aggregate(per_example: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute aggregate evaluation summaries.

    Parameters
    ----------
    per_example
        Output of :func:`~comorbidity_graph.evaluation.per_example.build_per_example`.

    Returns
    -------
    dict
        ``"overall"``  — one-row-per-variant summary table.
        ``"by_steer_mode"`` — summary stratified by steer_mode.
    """
    overall = _summarize(per_example)
    by_mode = (
        per_example
        .groupby("steer_mode", group_keys=False)
        .apply(_summarize)
        .reset_index(level=0)
    )
    return {"overall": overall, "by_steer_mode": by_mode}


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize one slice of the per-example table across scoring variants."""
    rows = []
    for variant in SCORE_VARIANTS:
        delta_col = f"delta_{variant}"
        g_col = f"guided_score_{variant}"
        n_col = f"naive_score_{variant}"
        gk_col = f"guided_effective_k_{variant}"
        nk_col = f"naive_effective_k_{variant}"
        winner_col = f"winner_{variant}"

        valid = df[df[delta_col].notna()]
        n_valid = len(valid)
        n_invalid = len(df) - n_valid

        if n_valid == 0:
            rows.append({
                "variant": variant,
                "n_valid": 0,
                "n_invalid": n_invalid,
            })
            continue

        deltas = valid[delta_col]
        winners = valid[winner_col]

        row = {
            "variant": variant,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "mean_guided_score": valid[g_col].mean(),
            "mean_naive_score": valid[n_col].mean(),
            "mean_delta": deltas.mean(),
            "median_delta": deltas.median(),
            "std_delta": deltas.std(),
            "guided_win_rate": (winners == "guided").mean(),
            "naive_win_rate": (winners == "naive").mean(),
            "tie_rate": (winners == "tie").mean(),
            "mean_guided_effective_k": valid[gk_col].mean(),
            "mean_naive_effective_k": valid[nk_col].mean(),
        }

        # Paired Wilcoxon signed-rank test (needs >= 10 non-zero diffs)
        nonzero = deltas[deltas != 0]
        if len(nonzero) >= 10:
            stat, pval = wilcoxon(nonzero)
            row["wilcoxon_stat"] = stat
            row["wilcoxon_pval"] = pval
        else:
            row["wilcoxon_stat"] = np.nan
            row["wilcoxon_pval"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)
