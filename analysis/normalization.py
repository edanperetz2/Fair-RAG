"""
Per-query metric normalization, shared across all analysis notebooks.

Reconciles four independent implementations that previously existed across
fair_rag_experiment.ipynb (two), fair_rag_diversity_story.ipynb, and
fair_rag_stats_exploration.ipynb. One real behavior difference was found and
resolved deliberately: on a zero/invalid normalization denominator, this returns
None/NaN (statistically conservative - "undefined, exclude from analysis") rather
than fabricating a value of 1.0 ("already at the max").
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

EU_UPPER_BOUND = 4.0  # LaMP-3's MAE metric ranges [0, 4]; used to flip it to higher-is-better.


def safe_div(num, den) -> Optional[float]:
    """Return num/den, or None if either is missing/NaN or den is zero."""
    if num is None or den is None:
        return None
    if pd.isna(num) or pd.isna(den):
        return None
    den = float(den)
    if den == 0.0:
        return None
    return float(num) / den


def normalize_query_rows(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("lamp_num", "generator_name", "qid"),
) -> pd.DataFrame:
    """
    Per-query max-normalization of Expected Utility / EE-Relevance / ILD to [0, 1]-ish
    ranges so different LaMP tasks and metrics become comparable. EE-Disparity passes
    through as a plain float cast (no division) - it's already in [0, 1] by construction.

    LaMP-3 uses MAE (lower is better); its utility values are flipped via
    `EU_UPPER_BOUND - value` before normalizing so "higher normalized EU is better"
    holds uniformly across every LaMP task.
    """
    if df.empty:
        return df.copy()

    group_cols = list(group_cols)
    normalized_parts = []

    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_map = dict(zip(group_cols, group_key))
        lamp_num = key_map.get("lamp_num")
        is_lower_better_eu = pd.notna(lamp_num) and int(lamp_num) == 3

        part = group.copy()

        ee_rel_denom = part["ee_relevance"].dropna().max() if "ee_relevance" in part.columns else None
        ild_denom = part["avg_ild_jaccard"].dropna().max() if "avg_ild_jaccard" in part.columns else None

        if is_lower_better_eu:
            eu_candidates = part["min_utility"].dropna() if "min_utility" in part.columns else pd.Series(dtype=float)
            if eu_candidates.empty and "expected_utility" in part.columns:
                eu_candidates = part["expected_utility"].dropna()
            flipped = (EU_UPPER_BOUND - eu_candidates) if not eu_candidates.empty else pd.Series(dtype=float)
            eu_denom = flipped.max() if not flipped.empty else None
            eu_numerators = (
                EU_UPPER_BOUND - part["expected_utility"] if "expected_utility" in part.columns else pd.Series(dtype=float)
            )
        else:
            eu_candidates = part["max_utility"].dropna() if "max_utility" in part.columns else pd.Series(dtype=float)
            if eu_candidates.empty and "expected_utility" in part.columns:
                eu_candidates = part["expected_utility"].dropna()
            eu_denom = eu_candidates.max() if not eu_candidates.empty else None
            eu_numerators = part["expected_utility"] if "expected_utility" in part.columns else pd.Series(dtype=float)

        part["ee_relevance_norm"] = (
            part["ee_relevance"].apply(lambda x: safe_div(x, ee_rel_denom)) if "ee_relevance" in part.columns else None
        )
        part["ee_disparity_norm"] = (
            part["ee_disparity"].apply(lambda x: float(x) if pd.notna(x) else None)
            if "ee_disparity" in part.columns
            else None
        )
        part["avg_ild_jaccard_norm"] = (
            part["avg_ild_jaccard"].apply(lambda x: safe_div(x, ild_denom)) if "avg_ild_jaccard" in part.columns else None
        )
        part["expected_utility_norm"] = eu_numerators.apply(lambda x: safe_div(x, eu_denom))

        normalized_parts.append(part)

    return pd.concat(normalized_parts, ignore_index=True)
