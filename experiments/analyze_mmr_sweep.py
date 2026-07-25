"""
Analysis for the MMR-lambda diversity sweep (docs/PROJECT_STATUS.md section 9).

The sweep's point: MMR is deterministic (EE-D pinned at 1.0 for every lambda), so its
lambda parameter manipulates diversity (ILD) with fairness held perfectly fixed - the
cleanest test this framework can produce of whether diversity, not fairness, drives
utility differences.

Sections:
  1. Manipulation check   - does lowering lambda actually raise ILD, per task?
  2. Sanity check         - lambda=1.0 must reproduce the deterministic run.
  3. Core test            - within-MMR regression EU_norm ~ ILD_norm (fairness fixed).
  4. Matched-diversity    - PL vs MMR utility at matched ILD (quantile bins): any gap
                            is the randomization/fairness component net of diversity.
  5. Interaction re-fit   - the section-8 EE-D x ILD interaction model, now with the
                            much wider ILD range the sweep provides at EE-D = 1.0.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, build_macro_comparison_rows, list_run_dirs, maybe_to_dataframe
from analysis import (
    normalize_query_rows, pool_delta_by_bin, fit_ols, select_best_precision,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

run_dirs = list_run_dirs()
print(f"Total completed run dirs: {len(run_dirs)}")

macro_df = select_best_precision(maybe_to_dataframe(build_macro_comparison_rows(run_dirs)))
query_df = select_best_precision(pd.DataFrame(build_query_metric_rows(run_dirs)))
print(f"Macro rows (best-precision deduped): {len(macro_df)}")
print(f"Query rows (best-precision deduped): {len(query_df)}")

mmr_macro = macro_df[macro_df["rerank_method"] == "mmr"].copy()

print("\n" + "=" * 80)
print("1. MANIPULATION CHECK: lambda -> mean ILD and raw EU, per task")
print("=" * 80)
pivot_ild = mmr_macro.pivot_table(index="mmr_lambda", columns="lamp_num", values="avg_ild_jaccard")
pivot_eu = mmr_macro.pivot_table(index="mmr_lambda", columns="lamp_num", values="expected_utility")
print("\nMean ILD by lambda (rows) x LaMP task (cols):")
print(pivot_ild.round(4).to_string())
print("\nRaw EU by lambda (rows) x LaMP task (cols):")
print(pivot_eu.round(4).to_string())
ild_range = pivot_ild.max() - pivot_ild.min()
print("\nILD range achieved per task (max - min across lambdas):")
print(ild_range.round(4).to_string())

print("\n" + "=" * 80)
print("2. SANITY CHECK: lambda=1.0 vs deterministic (must match)")
print("=" * 80)
det_macro = macro_df[macro_df["rerank_method"] == "deterministic"]
l10_macro = mmr_macro[np.isclose(mmr_macro["mmr_lambda"].astype(float), 1.0)]
merged = det_macro.merge(l10_macro, on="lamp_num", suffixes=("_det", "_l10"))
merged["eu_diff"] = (merged["expected_utility_det"] - merged["expected_utility_l10"]).abs()
merged["ild_diff"] = (merged["avg_ild_jaccard_det"] - merged["avg_ild_jaccard_l10"]).abs()
print(merged[["lamp_num", "expected_utility_det", "expected_utility_l10", "eu_diff", "ild_diff"]].round(6).to_string(index=False))
sanity_ok = bool((merged["eu_diff"] < 1e-6).all() and (merged["ild_diff"] < 1e-6).all())
print(f"\nSANITY: {'PASSED - lambda=1.0 reproduces deterministic exactly' if sanity_ok else 'FAILED - INVESTIGATE BEFORE TRUSTING ANYTHING BELOW'}")

print("\n" + "=" * 80)
print("3. CORE TEST: within-MMR regression EU_norm ~ ILD_norm (EE-D held fixed at 1.0)")
print("=" * 80)
query_norm_df = normalize_query_rows(query_df)
mmr_norm = query_norm_df[query_norm_df["rerank_method"] == "mmr"].copy()
print(f"MMR query-level rows: {len(mmr_norm)}")

pooled = fit_ols(mmr_norm, ["avg_ild_jaccard_norm"], "expected_utility_norm")
print(f"\nPooled (all 7 tasks): coef(ILD) = {pooled['coef']['avg_ild_jaccard_norm']:.4f} "
      f"(p={pooled['p_value']['avg_ild_jaccard_norm']:.4g}, n={pooled['n']}, R^2={pooled['r_squared']:.4f})")

print("\nPer task:")
for lamp_num in sorted(mmr_norm["lamp_num"].dropna().unique()):
    sub = mmr_norm[mmr_norm["lamp_num"] == lamp_num]
    try:
        m = fit_ols(sub, ["avg_ild_jaccard_norm"], "expected_utility_norm")
        print(f"  LaMP-{int(lamp_num)}: coef(ILD) = {m['coef']['avg_ild_jaccard_norm']:+.4f} "
              f"(p={m['p_value']['avg_ild_jaccard_norm']:.4g}, n={m['n']})")
    except ValueError as exc:
        print(f"  LaMP-{int(lamp_num)}: skipped ({exc})")

print("\n" + "=" * 80)
print("4. MATCHED-DIVERSITY FAIRNESS TEST: PL vs MMR utility at matched ILD")
print("=" * 80)
pl_mmr_norm = query_norm_df[query_norm_df["rerank_method"].isin(["pl", "mmr"])].dropna(
    subset=["avg_ild_jaccard_norm", "expected_utility_norm"]
)
quantile_edges = pl_mmr_norm["avg_ild_jaccard_norm"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
quantile_edges[-1] += 1e-9
labels = [f"Q{i+1} [{quantile_edges[i]:.3f},{quantile_edges[i+1]:.3f})" for i in range(5)]
print(f"ILD_norm quantile bin edges (pooled PL+MMR): {[round(e, 4) for e in quantile_edges]}")

matched = pool_delta_by_bin(
    pl_mmr_norm, metric_col="avg_ild_jaccard_norm", bin_edges=quantile_edges, bin_labels=labels,
    baseline_method="mmr", comparison_methods=["pl"],
)
print("\nPooled across tasks (delta = PL mean EU_norm - MMR mean EU_norm, within ILD bin):")
print(matched.sort_values("bin").to_string(index=False))

matched_by_task = pool_delta_by_bin(
    pl_mmr_norm, metric_col="avg_ild_jaccard_norm", bin_edges=quantile_edges, bin_labels=labels,
    baseline_method="mmr", comparison_methods=["pl"], group_by=["lamp_num"],
)
weighted = (
    matched_by_task.dropna(subset=["delta"])
    .assign(w=lambda d: d["n_comparison"])
    .groupby("lamp_num")
    .apply(lambda g: np.average(g["delta"], weights=g["w"]), include_groups=False)
)
print("\nPer-task n-weighted mean delta (PL - MMR at matched ILD):")
print(weighted.round(4).to_string())

print("\n" + "=" * 80)
print("5. INTERACTION RE-FIT: EU ~ EE-D + ILD + EE-D:ILD on pooled data incl. the sweep")
print("=" * 80)
cols = ["ee_disparity_norm", "avg_ild_jaccard_norm", "expected_utility_norm"]
inter_df = query_norm_df.dropna(subset=cols).copy()
inter_df["ee_d_x_ild"] = inter_df["ee_disparity_norm"] * inter_df["avg_ild_jaccard_norm"]
m2 = fit_ols(inter_df, ["ee_disparity_norm", "avg_ild_jaccard_norm"], "expected_utility_norm")
m3 = fit_ols(inter_df, ["ee_disparity_norm", "avg_ild_jaccard_norm", "ee_d_x_ild"], "expected_utility_norm")
print(f"n = {m3['n']} (was 3,552 before the sweep)")
print("\nModel 2: EU ~ EE-D + ILD")
print(f"  coef(EE-D) = {m2['coef']['ee_disparity_norm']:+.4f} (p={m2['p_value']['ee_disparity_norm']:.4g})")
print(f"  coef(ILD)  = {m2['coef']['avg_ild_jaccard_norm']:+.4f} (p={m2['p_value']['avg_ild_jaccard_norm']:.4g})")
print(f"  R^2 = {m2['r_squared']:.4f}")
print("\nModel 3: EU ~ EE-D + ILD + EE-D:ILD")
print(f"  coef(EE-D)       = {m3['coef']['ee_disparity_norm']:+.4f} (p={m3['p_value']['ee_disparity_norm']:.4g})")
print(f"  coef(ILD)        = {m3['coef']['avg_ild_jaccard_norm']:+.4f} (p={m3['p_value']['avg_ild_jaccard_norm']:.4g})")
print(f"  coef(EE-D x ILD) = {m3['coef']['ee_d_x_ild']:+.4f} (p={m3['p_value']['ee_d_x_ild']:.4g})")
print(f"  R^2 = {m3['r_squared']:.4f}")
