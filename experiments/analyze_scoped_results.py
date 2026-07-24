"""
Standalone analysis over whatever's currently in experiment_runs/ - the macro
summary table, ILD<->EE-D correlation, pooled-delta-by-bin tables (both the
generic per-bin-baseline framing and the paper-Table-2-style unbinned-baseline
framing), the ILD/EE-D mediation test, and a per-task raw comparison.

This is the scriptable equivalent of what fair_rag_diversity_story.ipynb /
fair_rag_stats_exploration.ipynb do interactively - useful for a quick text
dump after a new batch finishes, without re-running a whole notebook.
"""
import os
import sys

import pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, build_macro_comparison_rows, list_run_dirs, maybe_to_dataframe
from analysis import (
    normalize_query_rows, pool_delta_by_bin, fit_ols, format_rerank_label, format_macro_table_for_display,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

def select_best_precision(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some (lamp_num, pl_alpha) pairs exist at more than one pl_samples (N) value -
    e.g. this repo's pl_alpha_sweep_n30 experiment re-ran some alphas at N=30 while
    others (where the N=30 sweep was stopped partway and backfilled) only ever got
    N=10. Mixing both precisions for the same (task, alpha) in one analysis would
    silently double-count that condition and blend two different noise levels.
    Keep only the highest-pl_samples row per (lamp_num, pl_alpha); non-"pl" rows
    (deterministic/mmr, which don't have a pl_samples axis) pass through untouched.
    """
    is_pl = df["rerank_method"] == "pl"
    pl_df = df[is_pl]
    best_samples = pl_df.groupby(["lamp_num", "pl_alpha"])["pl_samples"].transform("max")
    keep_pl = pl_df["pl_samples"] == best_samples
    return pd.concat([df[~is_pl], pl_df[keep_pl]], ignore_index=True)


run_dirs = list_run_dirs()
print(f"Total completed run dirs: {len(run_dirs)}")

macro_rows = build_macro_comparison_rows(run_dirs)
macro_df_all = maybe_to_dataframe(macro_rows)
macro_df = select_best_precision(macro_df_all)
print(f"Macro rows: {len(macro_df_all)} (deduped to {len(macro_df)} best-precision-per-alpha rows)")

query_rows = build_query_metric_rows(run_dirs)
query_df_all = pd.DataFrame(query_rows)
query_df = select_best_precision(query_df_all)
print(f"Query rows: {len(query_df_all)} (deduped to {len(query_df)} best-precision-per-alpha rows)")

print("\n" + "=" * 80)
print("1. MACRO SUMMARY TABLE (one row per setting)")
print("=" * 80)
display_df = format_macro_table_for_display(macro_df)
print(display_df.to_string(index=False))

print("\n" + "=" * 80)
print("2. RAW ILD <-> EE-D CORRELATION (query-level, all settings pooled)")
print("=" * 80)
corr_df = query_df[["avg_ild_jaccard", "ee_disparity"]].dropna()
r, p = pearsonr(corr_df["avg_ild_jaccard"], corr_df["ee_disparity"])
print(f"Pearson r(ILD, EE-D) = {r:.4f} (p={p:.4g}, n={len(corr_df)})")

print("\n" + "=" * 80)
print("3. NORMALIZED PER-QUERY DATA + PL vs MMR/DET POOLED DELTA BY EE-D BIN")
print("=" * 80)
query_norm_df = normalize_query_rows(query_df)
EE_D_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
EE_D_BIN_LABELS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)"]

print("\n--- Paper-Table-2-style: unbinned deterministic baseline, PL only, per LaMP task ---")
delta_vs_det_paper_style = pool_delta_by_bin(
    query_norm_df, metric_col="ee_disparity_norm", bin_edges=EE_D_BINS, bin_labels=EE_D_BIN_LABELS,
    baseline_method="deterministic", comparison_methods=["pl"], group_by=["lamp_num"], bin_baseline=False,
)
print(delta_vs_det_paper_style.sort_values(["lamp_num", "bin"]).to_string(index=False))

print("\n--- Generic (binned-baseline) framing: PL/MMR vs deterministic, per LaMP task ---")
delta_vs_det = pool_delta_by_bin(
    query_norm_df, metric_col="ee_disparity_norm", bin_edges=EE_D_BINS, bin_labels=EE_D_BIN_LABELS,
    baseline_method="deterministic", group_by=["lamp_num"],
)
print(delta_vs_det.sort_values(["lamp_num", "method", "bin"]).to_string(index=False))

print("\n--- Aggregated across all LaMP tasks (pooled, binned-baseline framing) ---")
delta_vs_det_overall = pool_delta_by_bin(
    query_norm_df, metric_col="ee_disparity_norm", bin_edges=EE_D_BINS, bin_labels=EE_D_BIN_LABELS,
    baseline_method="deterministic",
)
print(delta_vs_det_overall.sort_values(["method", "bin"]).to_string(index=False))

delta_vs_mmr = pool_delta_by_bin(
    query_norm_df, metric_col="ee_disparity_norm", bin_edges=EE_D_BINS, bin_labels=EE_D_BIN_LABELS,
    baseline_method="mmr", comparison_methods=["pl"],
)
print("\n--- PL vs MMR baseline, pooled (all tasks) ---")
print(delta_vs_mmr.sort_values(["method", "bin"]).to_string(index=False))

print("\n" + "=" * 80)
print("4. MEDIATION TEST: does EE-D's effect on EU shrink once ILD is controlled for?")
print("=" * 80)
mediation_cols = ["ee_disparity_norm", "avg_ild_jaccard_norm", "expected_utility_norm"]
mediation_df = query_norm_df.dropna(subset=mediation_cols)
simple_model = fit_ols(mediation_df, ["ee_disparity_norm"], "expected_utility_norm")
multi_model = fit_ols(mediation_df, ["ee_disparity_norm", "avg_ild_jaccard_norm"], "expected_utility_norm")
interaction_df = mediation_df.copy()
interaction_df["ee_d_x_ild"] = interaction_df["ee_disparity_norm"] * interaction_df["avg_ild_jaccard_norm"]
interaction_model = fit_ols(interaction_df, ["ee_disparity_norm", "avg_ild_jaccard_norm", "ee_d_x_ild"], "expected_utility_norm")

print(f"n = {simple_model['n']} query-run observations\n")
print("Model 1: EU ~ EE-D")
print(f"  coef(EE-D) = {simple_model['coef']['ee_disparity_norm']:.4f}  (p={simple_model['p_value']['ee_disparity_norm']:.4g})")
print(f"  R^2 = {simple_model['r_squared']:.4f}\n")
print("Model 2: EU ~ EE-D + ILD")
print(f"  coef(EE-D) = {multi_model['coef']['ee_disparity_norm']:.4f}  (p={multi_model['p_value']['ee_disparity_norm']:.4g})")
print(f"  coef(ILD)  = {multi_model['coef']['avg_ild_jaccard_norm']:.4f}  (p={multi_model['p_value']['avg_ild_jaccard_norm']:.4g})")
print(f"  R^2 = {multi_model['r_squared']:.4f}\n")
ee_d_simple = simple_model["coef"]["ee_disparity_norm"]
ee_d_multi = multi_model["coef"]["ee_disparity_norm"]
shrinkage = (1.0 - abs(ee_d_multi) / abs(ee_d_simple)) if ee_d_simple != 0 else float("nan")
print(f"EE-D coefficient shrinkage after controlling for ILD: {shrinkage:.1%}\n")
print("Model 3: EU ~ EE-D + ILD + EE-D:ILD (interaction)")
print(f"  coef(EE-D)       = {interaction_model['coef']['ee_disparity_norm']:.4f}  (p={interaction_model['p_value']['ee_disparity_norm']:.4g})")
print(f"  coef(ILD)        = {interaction_model['coef']['avg_ild_jaccard_norm']:.4f}  (p={interaction_model['p_value']['avg_ild_jaccard_norm']:.4g})")
print(f"  coef(EE-D x ILD) = {interaction_model['coef']['ee_d_x_ild']:.4f}  (p={interaction_model['p_value']['ee_d_x_ild']:.4g})")
print(f"  R^2 = {interaction_model['r_squared']:.4f}")

print("\n" + "=" * 80)
print("5. PER-TASK MACRO COMPARISON: raw (non-normalized) EU/EE-D/ILD, one row per setting")
print("=" * 80)
for lamp_num in sorted(macro_df["lamp_num"].unique()):
    sub = macro_df[macro_df["lamp_num"] == lamp_num].copy()
    sub["label"] = sub.apply(lambda r: format_rerank_label(r), axis=1)
    print(f"\n--- LaMP-{lamp_num} (metric: {sub['metric_name'].iloc[0]}) ---")
    print(sub[["label", "generator_name", "ranker", "expected_utility", "ee_disparity", "ee_relevance", "avg_ild_jaccard"]].to_string(index=False))
