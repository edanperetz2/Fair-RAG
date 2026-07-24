"""
Metric binning + pooled-delta-vs-baseline analysis, shared across all analysis
notebooks.

Reconciles five independent implementations that previously existed (three
EE-D-bin variants in fair_rag_experiment.ipynb, one EE-D-bin and one ILD-bin
variant in fair_rag_diversity_story.ipynb) into one generic function. Both
framings observed in the original notebooks - "one method vs. every other
method" and "one method vs. two different baselines" - are expressible as
different arguments here.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import pandas as pd


def bin_label_for_value(value, bin_edges: Sequence[float], bin_labels: Sequence[str]) -> Optional[str]:
    """Assign `value` to one of `len(bin_labels)` half-open bins [edges[i], edges[i+1])."""
    if value is None or pd.isna(value):
        return None
    value = float(value)
    for left, right, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
        if left <= value < right:
            return label
    return bin_labels[-1]


def pool_delta_by_bin(
    df: pd.DataFrame,
    *,
    metric_col: str,
    bin_edges: Sequence[float],
    bin_labels: Sequence[str],
    baseline_method=None,
    baseline_filter: Optional[Callable[[pd.Series], bool]] = None,
    comparison_methods: Optional[Iterable] = None,
    method_col: str = "rerank_method",
    value_col: str = "expected_utility_norm",
    group_by: Optional[List[str]] = None,
    bin_baseline: bool = True,
) -> pd.DataFrame:
    """
    Bin rows by `metric_col`, then for one or more `comparison_methods` report the
    pooled mean-`value_col` delta against a baseline, per bin (and per `group_by` key,
    e.g. per LaMP task / ranker).

    Exactly one of `baseline_method` (equality check against `method_col`) or
    `baseline_filter` (row predicate - needed for e.g. "MMR at a specific lambda" as
    the baseline) must be given. `comparison_methods=None` means "every method other
    than the baseline".

    `bin_baseline=True` (default) requires baseline rows to fall into the same bin as
    the comparison rows they're differenced against - appropriate when the baseline
    itself is stochastically sampled and has a `metric_col` spread of its own.
    `bin_baseline=False` instead pools *all* baseline rows into one scalar mean
    (ignoring `metric_col`) and compares every bin's comparison-method mean against
    that single number. This matches the original paper's Table 2 methodology, where
    the baseline is a single deterministic run (one EE-D value, e.g. always 1.0) and
    can't itself populate every bin - binning it too would silently produce nothing
    but NaNs outside its one bin.

    Returns a tidy DataFrame: group_by columns + ["method", "bin", "n_baseline",
    "n_comparison", "delta"].
    """
    if df.empty:
        return pd.DataFrame()
    if (baseline_method is None) == (baseline_filter is None):
        raise ValueError("Pass exactly one of baseline_method or baseline_filter")

    group_by = list(group_by) if group_by else []
    work = df.copy()
    work["_bin"] = work[metric_col].apply(lambda v: bin_label_for_value(v, bin_edges, bin_labels))

    rows = []
    grouped = work.groupby(group_by, dropna=False) if group_by else [((), work)]
    for group_key, part in grouped:
        if group_by and not isinstance(group_key, tuple):
            group_key = (group_key,)

        if baseline_filter is not None:
            baseline_rows = part[part.apply(baseline_filter, axis=1)]
        else:
            baseline_rows = part[part[method_col] == baseline_method]
        if baseline_rows.empty:
            continue

        if comparison_methods is None:
            exclude = {baseline_method} if baseline_method is not None else set()
            candidate_methods = [m for m in part[method_col].unique() if m not in exclude]
        else:
            candidate_methods = list(comparison_methods)

        pooled_base_vals = baseline_rows[value_col].dropna() if not bin_baseline else None

        for method in candidate_methods:
            comparison_rows = part[part[method_col] == method]
            if comparison_rows.empty:
                continue
            for bin_label in bin_labels:
                base_vals = (
                    pooled_base_vals
                    if not bin_baseline
                    else baseline_rows.loc[baseline_rows["_bin"] == bin_label, value_col].dropna()
                )
                comp_vals = comparison_rows.loc[comparison_rows["_bin"] == bin_label, value_col].dropna()
                delta = (comp_vals.mean() - base_vals.mean()) if (not base_vals.empty and not comp_vals.empty) else None
                row = {
                    "method": method,
                    "bin": bin_label,
                    "n_baseline": int(len(base_vals)),
                    "n_comparison": int(len(comp_vals)),
                    "delta": delta,
                }
                if group_by:
                    row.update(dict(zip(group_by, group_key)))
                rows.append(row)

    if not rows:
        # Return a well-typed empty frame (not a columnless one) so callers can safely
        # groupby/filter on the expected columns even when nothing matched.
        return pd.DataFrame(columns=group_by + ["method", "bin", "n_baseline", "n_comparison", "delta"])
    return pd.DataFrame(rows)
