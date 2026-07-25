"""
Analysis for the MMR-lambda diversity sweeps (docs/PROJECT_STATUS.md sections 9-10).

The sweep's point: MMR is deterministic (EE-D pinned at 1.0 for every lambda), so its
lambda parameter manipulates diversity (ILD) with fairness held perfectly fixed - the
cleanest test this framework can produce of whether diversity, not fairness, drives
utility differences. Originally run on BM25 (section 9); extended to Contriever as a
generalization test (section 10), so every per-ranker section below loops over
whichever rankers have MMR-sweep data present.

Sections (per ranker):
  1. Manipulation check   - does lowering lambda actually raise ILD, per task?
  2. Sanity check         - lambda=1.0 must reproduce the deterministic run.
  3. Core test            - within-MMR regression EU_norm ~ ILD_norm (fairness fixed).
  4. Matched-diversity    - PL vs MMR utility at matched ILD (quantile bins): any gap
                            is the randomization/fairness component net of diversity.
Pooled (all rankers):
  5. Interaction re-fit   - the EE-D x ILD interaction model on everything.
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

query_norm_df = normalize_query_rows(query_df)

sweep_rankers = sorted(
    macro_df.loc[macro_df["rerank_method"] == "mmr"]
    .groupby("ranker")["mmr_lambda"].nunique()
    .loc[lambda s: s >= 3].index
)
print(f"Rankers with MMR-sweep data: {sweep_rankers}")

for ranker in sweep_rankers:
    tag = ranker.upper()
    r_macro = macro_df[macro_df["ranker"] == ranker]
    mmr_macro = r_macro[r_macro["rerank_method"] == "mmr"].copy()

    print("\n" + "#" * 80)
    print(f"# RANKER: {tag}")
    print("#" * 80)

    print("\n" + "=" * 80)
    print(f"1. [{tag}] MANIPULATION CHECK: lambda -> mean ILD and raw EU, per task")
    print("=" * 80)
    pivot_ild = mmr_macro.pivot_table(index="mmr_lambda", columns="lamp_num", values="avg_ild_jaccard")
    pivot_eu = mmr_macro.pivot_table(index="mmr_lambda", columns="lamp_num", values="expected_utility")
    print("\nMean ILD by lambda (rows) x LaMP task (cols):")
    print(pivot_ild.round(4).to_string())
    print("\nRaw EU by lambda (rows) x LaMP task (cols):")
    print(pivot_eu.round(4).to_string())
    print("\nILD range achieved per task (max - min across lambdas):")
    print((pivot_ild.max() - pivot_ild.min()).round(4).to_string())

    print("\n" + "=" * 80)
    print(f"2. [{tag}] SANITY CHECK: lambda=1.0 vs deterministic (must match)")
    print("=" * 80)
    det_macro = r_macro[r_macro["rerank_method"] == "deterministic"]
    l10_macro = mmr_macro[np.isclose(mmr_macro["mmr_lambda"].astype(float), 1.0)]
    if l10_macro.empty:
        print("No lambda=1.0 runs for this ranker - skipping.")
    else:
        merged = det_macro.merge(l10_macro, on="lamp_num", suffixes=("_det", "_l10"))
        merged["eu_diff"] = (merged["expected_utility_det"] - merged["expected_utility_l10"]).abs()
        merged["ild_diff"] = (merged["avg_ild_jaccard_det"] - merged["avg_ild_jaccard_l10"]).abs()
        print(merged[["lamp_num", "expected_utility_det", "expected_utility_l10", "eu_diff", "ild_diff"]].round(6).to_string(index=False))
        sanity_ok = bool((merged["eu_diff"] < 1e-6).all() and (merged["ild_diff"] < 1e-6).all())
        print(f"\nSANITY: {'PASSED' if sanity_ok else 'FAILED - INVESTIGATE BEFORE TRUSTING ANYTHING BELOW'}")

    print("\n" + "=" * 80)
    print(f"3. [{tag}] CORE TEST: within-MMR regression EU_norm ~ ILD_norm (EE-D fixed)")
    print("=" * 80)
    mmr_norm = query_norm_df[(query_norm_df["rerank_method"] == "mmr") & (query_norm_df["ranker"] == ranker)].copy()
    print(f"MMR query-level rows: {len(mmr_norm)}")
    pooled = fit_ols(mmr_norm, ["avg_ild_jaccard_norm"], "expected_utility_norm")
    print(f"\nPooled (all 7 tasks): coef(ILD) = {pooled['coef']['avg_ild_jaccard_norm']:+.4f} "
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
    print(f"4. [{tag}] MATCHED-DIVERSITY FAIRNESS TEST: PL vs MMR utility at matched ILD")
    print("=" * 80)
    pl_mmr_norm = query_norm_df[
        (query_norm_df["rerank_method"].isin(["pl", "mmr"])) & (query_norm_df["ranker"] == ranker)
    ].dropna(subset=["avg_ild_jaccard_norm", "expected_utility_norm"])
    if pl_mmr_norm[pl_mmr_norm["rerank_method"] == "pl"].empty:
        print("No PL runs for this ranker - skipping.")
    else:
        quantile_edges = pl_mmr_norm["avg_ild_jaccard_norm"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
        quantile_edges[-1] += 1e-9
        labels = [f"Q{i+1} [{quantile_edges[i]:.3f},{quantile_edges[i+1]:.3f})" for i in range(5)]
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

print("\n" + "#" * 80)
print("# POOLED (ALL RANKERS)")
print("#" * 80)
print("\n" + "=" * 80)
print("5. INTERACTION RE-FIT: EU ~ EE-D + ILD + EE-D:ILD on everything")
print("=" * 80)
cols = ["ee_disparity_norm", "avg_ild_jaccard_norm", "expected_utility_norm"]
inter_df = query_norm_df.dropna(subset=cols).copy()
inter_df["ee_d_x_ild"] = inter_df["ee_disparity_norm"] * inter_df["avg_ild_jaccard_norm"]
m2 = fit_ols(inter_df, ["ee_disparity_norm", "avg_ild_jaccard_norm"], "expected_utility_norm")
m3 = fit_ols(inter_df, ["ee_disparity_norm", "avg_ild_jaccard_norm", "ee_d_x_ild"], "expected_utility_norm")
print(f"n = {m3['n']}")
print("\nModel 2: EU ~ EE-D + ILD")
print(f"  coef(EE-D) = {m2['coef']['ee_disparity_norm']:+.4f} (p={m2['p_value']['ee_disparity_norm']:.4g})")
print(f"  coef(ILD)  = {m2['coef']['avg_ild_jaccard_norm']:+.4f} (p={m2['p_value']['avg_ild_jaccard_norm']:.4g})")
print(f"  R^2 = {m2['r_squared']:.4f}")
print("\nModel 3: EU ~ EE-D + ILD + EE-D:ILD")
print(f"  coef(EE-D)       = {m3['coef']['ee_disparity_norm']:+.4f} (p={m3['p_value']['ee_disparity_norm']:.4g})")
print(f"  coef(ILD)        = {m3['coef']['avg_ild_jaccard_norm']:+.4f} (p={m3['p_value']['avg_ild_jaccard_norm']:.4g})")
print(f"  coef(EE-D x ILD) = {m3['coef']['ee_d_x_ild']:+.4f} (p={m3['p_value']['ee_d_x_ild']:.4g})")
print(f"  R^2 = {m3['r_squared']:.4f}")
