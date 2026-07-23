"""
Shared analysis toolkit for the research notebooks (fair_rag_diversity_story.ipynb,
fair_rag_stats_exploration.ipynb): plotting, normalization, binning, run-loading, and
statistics helpers that aren't specific to any one thesis. Builds on top of
framework.cross_run_analysis rather than duplicating it - this package is for analyzing
already-completed runs, not for running experiments (that stays in fair_rag_experiment.ipynb).
"""

from analysis.labels import format_rerank_label
from analysis.normalization import safe_div, normalize_query_rows, macro_from_normalized_query_rows
from analysis.binning import bin_label_for_value, pool_delta_by_bin
from analysis.loading import (
    find_completed_run_dirs,
    existing_setting_ids,
    load_relevance_mapping,
    load_retrieval_scores,
)
from analysis.plotting import plot_metric_scatter_panels, format_macro_table_for_display
from analysis.stats import (
    weighted_mean,
    bootstrap_ci,
    assign_quantile_bins,
    summarize_binned_metric,
    middle_vs_tail_summary,
    add_quantile_trend,
    fit_ols,
)

__all__ = [
    "format_rerank_label",
    "safe_div",
    "normalize_query_rows",
    "macro_from_normalized_query_rows",
    "bin_label_for_value",
    "pool_delta_by_bin",
    "find_completed_run_dirs",
    "existing_setting_ids",
    "load_relevance_mapping",
    "load_retrieval_scores",
    "plot_metric_scatter_panels",
    "format_macro_table_for_display",
    "weighted_mean",
    "bootstrap_ci",
    "assign_quantile_bins",
    "summarize_binned_metric",
    "middle_vs_tail_summary",
    "add_quantile_trend",
    "fit_ols",
]
