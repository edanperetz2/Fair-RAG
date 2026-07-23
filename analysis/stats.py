"""
Generic statistical primitives (bootstrap CI, quantile binning, weighted mean, trend
lines), shared across all analysis notebooks. Originates from
fair_rag_stats_exploration.ipynb; `bootstrap_ci` no longer reaches for a notebook-local
config global for its `rounds`/`seed` defaults - callers pass them explicitly.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd


def weighted_mean(values, weights) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not mask.any() or weights[mask].sum() == 0:
        return None
    return float(np.average(values[mask], weights=weights[mask]))


def bootstrap_ci(values, stat_fn: Callable = np.mean, rounds: int = 2000, seed: Optional[int] = None):
    """Percentile bootstrap confidence interval. Returns (ci_low, point_estimate, ci_high)."""
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    if arr.size == 1:
        scalar = float(stat_fn(arr))
        return (scalar, scalar, scalar)

    rng = np.random.default_rng(seed)
    samples = [float(stat_fn(rng.choice(arr, size=arr.size, replace=True))) for _ in range(rounds)]
    return (
        float(np.quantile(samples, 0.025)),
        float(stat_fn(arr)),
        float(np.quantile(samples, 0.975)),
    )


def assign_quantile_bins(series: pd.Series, bins: int) -> pd.Series:
    clean = pd.Series(series).astype(float)
    usable_bins = min(bins, clean.dropna().nunique())
    if usable_bins < 2:
        return pd.Series(["all"] * len(clean), index=clean.index)
    return pd.qcut(clean, q=usable_bins, duplicates="drop").astype(str)


def summarize_binned_metric(
    df: pd.DataFrame,
    *,
    metric_col: str,
    target_col: str,
    bins: int,
    group_cols: Optional[List[str]] = None,
    fallback_bins: int = 5,
    label_name: str = "bin",
    bootstrap_rounds: int = 2000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    group_cols = group_cols or []
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for group_key, part in grouped:
        part = part.dropna(subset=[metric_col, target_col]).copy()
        if part.empty:
            continue

        local_bins = bins if part[metric_col].nunique() >= bins else min(fallback_bins, part[metric_col].nunique())
        part[label_name] = assign_quantile_bins(part[metric_col], max(local_bins, 2))

        for bin_label, bin_df in part.groupby(label_name, dropna=False):
            ci_low, center, ci_high = bootstrap_ci(
                bin_df[target_col].to_numpy(), np.mean, rounds=bootstrap_rounds, seed=seed
            )
            row = {
                label_name: bin_label,
                "n": int(len(bin_df)),
                "mean_target": center,
                "median_target": float(bin_df[target_col].median()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "metric_mean": float(bin_df[metric_col].mean()),
            }
            if group_cols:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                row.update(dict(zip(group_cols, group_key)))
            rows.append(row)

    return pd.DataFrame(rows)


def middle_vs_tail_summary(
    df: pd.DataFrame, *, metric_col: str, target_col: str, group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    group_cols = group_cols or []
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    for group_key, part in grouped:
        part = part.dropna(subset=[metric_col, target_col]).copy()
        if part.empty or part[metric_col].nunique() < 3:
            continue
        part["bucket"] = pd.qcut(part[metric_col], q=3, labels=["low", "mid", "high"], duplicates="drop")
        if part["bucket"].isna().all():
            continue
        summary = part.groupby("bucket")[target_col].agg(["count", "mean", "median"]).reset_index()
        present = set(summary["bucket"])
        summary["delta_mid_minus_low"] = (
            float(summary.loc[summary["bucket"] == "mid", "mean"].iloc[0] - summary.loc[summary["bucket"] == "low", "mean"].iloc[0])
            if {"mid", "low"}.issubset(present)
            else np.nan
        )
        summary["delta_mid_minus_high"] = (
            float(summary.loc[summary["bucket"] == "mid", "mean"].iloc[0] - summary.loc[summary["bucket"] == "high", "mean"].iloc[0])
            if {"mid", "high"}.issubset(present)
            else np.nan
        )
        if group_cols:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            for col, value in zip(group_cols, group_key):
                summary[col] = value
        rows.append(summary)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_quantile_trend(
    ax, df: pd.DataFrame, *, x_col: str, y_col: str, bins: int = 12, color: str = "black", label: Optional[str] = None
) -> None:
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    plot_df["trend_bin"] = assign_quantile_bins(plot_df[x_col], bins)
    trend = plot_df.groupby("trend_bin", dropna=False)[[x_col, y_col]].mean().sort_values(x_col)
    ax.plot(trend[x_col], trend[y_col], color=color, linewidth=2, label=label)
