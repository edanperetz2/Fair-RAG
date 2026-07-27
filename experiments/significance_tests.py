"""
Formal statistical significance tests for the report (beyond the OLS p-values
already printed by experiments/analyze_mmr_sweep.py).

Four test families, each attached to one report claim:

  A. "Fairness is free" (the paper's claim) - paired Wilcoxon signed-rank per
     (generator, ranker) cell: deterministic vs each fair setting, paired by query
     on normalized EU, pooled across tasks, Holm-corrected within cell.
  B. "PL adds nothing over MMR once diversity is controlled" - per-cell OLS of
     EU_norm on a PL-vs-MMR method dummy, controlling ILD_norm and task fixed
     effects; the dummy's p-value tests the fairness component net of diversity.
  C. "Diversity's effect flips sign with generator size" - within-MMR pooled OLS
     with an ILD x generator interaction (plus ranker/task fixed effects); the
     interaction term formally tests Small-vs-Base coefficient difference.
  D. "The fairness-utility tradeoff emerges with generator size" - all-rows OLS
     with an EE-D x generator interaction (controlling ILD, ranker/task effects).

Uses the same loading/normalization pipeline as analyze_mmr_sweep.py.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from analysis import normalize_query_rows, fit_ols, select_best_precision

pd.set_option("display.width", 160)

CELLS = [
    ("flanT5Small", "bm25"), ("flanT5Small", "contriever"),
    ("flanT5Base", "bm25"), ("flanT5Base", "contriever"),
]
PL_ALPHAS = [1, 2, 4, 8]
MMR_LAMBDAS = [0.15, 0.55]  # strongest-diversity and mid settings vs deterministic


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (returned in the input order)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


print("Loading run data...")
run_dirs = list_run_dirs()
query_df = select_best_precision(pd.DataFrame(build_query_metric_rows(run_dirs)))
qn = normalize_query_rows(query_df)
print(f"{len(run_dirs)} runs -> {len(qn)} normalized query rows")


def add_task_dummies(df, feats):
    df = df.copy()
    tasks = sorted(df["lamp_num"].dropna().unique())[1:]  # drop-one coding
    for t in tasks:
        col = f"task_{int(t)}"
        df[col] = (df["lamp_num"] == t).astype(float)
        feats = feats + [col]
    return df, feats


print("\n" + "=" * 90)
print("A. PAIRED TEST OF 'FAIRNESS IS FREE': deterministic vs each fair setting")
print("   (Wilcoxon signed-rank on per-query normalized EU, pooled across tasks;")
print("    PL settings use each query's mean EU over its 10 sampled lists;")
print("    Holm correction within each cell's 6-test family)")
print("=" * 90)
for gen, ranker in CELLS:
    cell = qn[(qn["generator_name"] == gen) & (qn["ranker"] == ranker)].dropna(
        subset=["expected_utility_norm"])
    det = (cell[cell["rerank_method"] == "deterministic"]
           .groupby(["lamp_num", "qid"])["expected_utility_norm"].mean())
    rows = []
    settings = ([("pl", f"PL a={a}", cell["pl_alpha"] == a) for a in PL_ALPHAS]
                + [("mmr", f"MMR l={l}", np.isclose(cell["mmr_lambda"].astype(float), l))
                   for l in MMR_LAMBDAS])
    for method, label, mask in settings:
        fair = (cell[(cell["rerank_method"] == method) & mask]
                .groupby(["lamp_num", "qid"])["expected_utility_norm"].mean())
        pair = pd.concat([det.rename("det"), fair.rename("fair")], axis=1).dropna()
        diff = pair["fair"] - pair["det"]
        if (diff == 0).all():
            stat_p = 1.0
        else:
            stat_p = scipy_stats.wilcoxon(pair["fair"], pair["det"], zero_method="zsplit").pvalue
        rows.append({"setting": label, "n_queries": len(pair),
                     "mean_delta_vs_det": diff.mean(), "p_wilcoxon": stat_p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"].to_numpy())
    out["significant_.05"] = np.where(out["p_holm"] < 0.05, "YES", "no")
    print(f"\n[{gen}/{ranker}]")
    print(out.round(4).to_string(index=False))

print("\n" + "=" * 90)
print("B. 'PL ADDS NOTHING OVER MMR NET OF DIVERSITY': per-cell OLS")
print("   EU_norm ~ is_pl + ILD_norm + task dummies (PL and MMR rows only)")
print("=" * 90)
for gen, ranker in CELLS:
    sub = qn[(qn["generator_name"] == gen) & (qn["ranker"] == ranker)
             & (qn["rerank_method"].isin(["pl", "mmr"]))].dropna(
        subset=["expected_utility_norm", "avg_ild_jaccard_norm"]).copy()
    sub["is_pl"] = (sub["rerank_method"] == "pl").astype(float)
    sub, feats = add_task_dummies(sub, ["is_pl", "avg_ild_jaccard_norm"])
    m = fit_ols(sub, feats, "expected_utility_norm")
    c, p = m["coef"]["is_pl"], m["p_value"]["is_pl"]
    print(f"[{gen}/{ranker}] coef(is_pl) = {c:+.4f} (p={p:.4g}, n={m['n']})"
          f"  -> {'significant' if p < 0.05 else 'NOT significant'}")

print("\n" + "=" * 90)
print("C. FORMAL TEST OF THE SIGN FLIP: within-MMR pooled OLS")
print("   EU_norm ~ ILD_norm + is_base + ILD_norm x is_base + ranker + task dummies")
print("=" * 90)
mmr = qn[qn["rerank_method"] == "mmr"].dropna(
    subset=["expected_utility_norm", "avg_ild_jaccard_norm"]).copy()
mmr["is_base"] = (mmr["generator_name"] == "flanT5Base").astype(float)
mmr["ild_x_base"] = mmr["avg_ild_jaccard_norm"] * mmr["is_base"]
mmr["is_contriever"] = (mmr["ranker"] == "contriever").astype(float)
mmr, feats = add_task_dummies(mmr, ["avg_ild_jaccard_norm", "is_base", "ild_x_base", "is_contriever"])
m = fit_ols(mmr, feats, "expected_utility_norm")
print(f"n = {m['n']}")
print(f"coef(ILD, Small baseline) = {m['coef']['avg_ild_jaccard_norm']:+.4f} (p={m['p_value']['avg_ild_jaccard_norm']:.4g})")
print(f"coef(ILD x is_base)       = {m['coef']['ild_x_base']:+.4f} (p={m['p_value']['ild_x_base']:.4g})")
print(f"  -> Base-vs-Small difference in the diversity effect is "
      f"{'SIGNIFICANT' if m['p_value']['ild_x_base'] < 0.05 else 'not significant'}")

print("\n" + "=" * 90)
print("D. FORMAL TEST OF TRADEOFF EMERGENCE: all rows, OLS")
print("   EU_norm ~ EE-D + ILD + is_base + EE-D x is_base + ranker + task dummies")
print("=" * 90)
alldf = qn.dropna(subset=["expected_utility_norm", "avg_ild_jaccard_norm", "ee_disparity_norm"]).copy()
alldf["is_base"] = (alldf["generator_name"] == "flanT5Base").astype(float)
alldf["eed_x_base"] = alldf["ee_disparity_norm"] * alldf["is_base"]
alldf["is_contriever"] = (alldf["ranker"] == "contriever").astype(float)
alldf, feats = add_task_dummies(
    alldf, ["ee_disparity_norm", "avg_ild_jaccard_norm", "is_base", "eed_x_base", "is_contriever"])
m = fit_ols(alldf, feats, "expected_utility_norm")
print(f"n = {m['n']}")
print(f"coef(EE-D, Small baseline) = {m['coef']['ee_disparity_norm']:+.4f} (p={m['p_value']['ee_disparity_norm']:.4g})")
print(f"coef(EE-D x is_base)       = {m['coef']['eed_x_base']:+.4f} (p={m['p_value']['eed_x_base']:.4g})")
print(f"  -> Base-vs-Small difference in the fairness-utility slope is "
      f"{'SIGNIFICANT' if m['p_value']['eed_x_base'] < 0.05 else 'not significant'}")
