"""Produce the per-row evaluation table (guided vs naive for every example).

Pivots scored data so each row_key has guided and naive scores side by side,
with delta and winner columns for each scoring variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_VARIANTS = ["unsigned", "sign_aware", "magnitude"]


def build_per_example(scored: pd.DataFrame) -> pd.DataFrame:
    """Build the per-example comparison table from scored prediction sets.

    Parameters
    ----------
    scored
        Output of :func:`~comorbidity_graph.score.score_topk.score_topk`,
        with one row per (row_key, role).

    Returns
    -------
    pd.DataFrame
        One row per ``row_key`` with guided vs naive scores, deltas, and
        winners for each scoring variant.
    """
    guided = scored[scored["role"] == "guided"].set_index("row_key")
    naive = scored[scored["role"] == "naive"].set_index("row_key")

    common_keys = guided.index.intersection(naive.index)
    guided = guided.loc[common_keys]
    naive = naive.loc[common_keys]

    result = pd.DataFrame(index=common_keys)
    result.index.name = "row_key"
    result["steer_mode"] = guided["steer_mode"]
    result["perturbed_code"] = guided["perturbed_code"]

    for variant in SCORE_VARIANTS:
        sc = f"score_{variant}"
        ek = f"effective_k_{variant}"

        result[f"guided_{sc}"] = guided[sc].values
        result[f"naive_{sc}"] = naive[sc].values
        result[f"delta_{variant}"] = (
            guided[sc].values - naive[sc].values
        )

        result[f"guided_{ek}"] = guided[ek].values
        result[f"naive_{ek}"] = naive[ek].values

        # Winner: guided wins when delta < 0 (closer to comorbidity structure)
        delta = result[f"delta_{variant}"]
        result[f"winner_{variant}"] = np.where(
            delta.isna(),
            "na",
            np.where(delta < 0, "guided", np.where(delta > 0, "naive", "tie")),
        )

    return result.reset_index()
