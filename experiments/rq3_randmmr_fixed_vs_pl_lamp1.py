"""
Compare the corrected RandMMR (pl_randmmr_fixed: per-rank lambda, deterministic
MMR for ranks 2..k) against matched plain PL, for LaMP-1 only, at the same
N=10 precision and matching methodology as experiments/rq3_randmmr_vs_pl_n10.py
(PL alpha matched to whichever of {1,2,4,8} has EE-D closest to the hybrid's
own, biased toward equal-or-higher PL disparity).
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
from analysis import select_best_precision
from analysis.loading import select_full_coverage_runs
from analysis.with_gold_normalization import normalize_query_rows_with_gold

GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]
PL_ALPHAS = [1.0, 2.0, 4.0, 8.0]
CLOSE_ENOUGH = 0.02

print("Loading run directories...")
run_dirs = list_run_dirs()
all_rows_df = pd.DataFrame(build_query_metric_rows(run_dirs))
all_rows_df = select_full_coverage_runs(all_rows_df)
all_rows_df = all_rows_df[all_rows_df["seed"] == 42]

ceiling_source_df = select_best_precision(all_rows_df.copy())

raw_df = all_rows_df[
    (all_rows_df["lamp_num"] == 1)
    & all_rows_df["generator_name"].isin(GENERATORS)
    & all_rows_df["ranker"].isin(RANKERS)
]
raw_df = select_best_precision(raw_df)

print("rerank methods present for LaMP-1:", raw_df["rerank_method"].unique())

pooled = pd.concat([
    raw_df[raw_df["rerank_method"] == "deterministic"],
    raw_df[(raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"].isin(PL_ALPHAS))],
    raw_df[raw_df["rerank_method"] == "pl_randmmr_fixed"],
    raw_df[raw_df["rerank_method"] == "pl_randmmr"],
], ignore_index=True)

norm_cells = pooled[["lamp_num", "generator_name", "ranker"]].drop_duplicates().itertuples(index=False, name=None)
qn = normalize_query_rows_with_gold(pooled, ceiling_source_df, cells=list(norm_cells))
qn["expected_utility_norm"] = pd.to_numeric(qn["expected_utility_norm"], errors="coerce")


def choose_alpha(cell_pl_means, rand_ee_d):
    diffs = {a: abs(v - rand_ee_d) for a, v in cell_pl_means.items()}
    closest_alpha = min(diffs, key=diffs.get)
    if diffs[closest_alpha] <= CLOSE_ENOUGH:
        return closest_alpha
    above = {a: v for a, v in cell_pl_means.items() if v >= rand_ee_d}
    if above:
        return min(above, key=above.get)
    return closest_alpha


def welch(a, b):
    if len(a) >= 2 and len(b) >= 2 and a.nunique() > 1:
        _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    return np.nan


for method_label in ["pl_randmmr_fixed", "pl_randmmr"]:
    print(f"\n=== {method_label} vs matched plain PL, LaMP-1 ===")
    rows = []
    hybrid_all = qn[qn["rerank_method"] == method_label]
    cells = hybrid_all[["lamp_num", "generator_name", "ranker"]].drop_duplicates()
    for _, (task, gen, ranker) in cells.iterrows():
        rh = qn[(qn["lamp_num"] == task) & (qn["generator_name"] == gen) & (qn["ranker"] == ranker)
                & (qn["rerank_method"] == method_label)]
        if rh.empty:
            continue
        rand_ee_d = rh["ee_disparity"].mean()
        pl_cell = qn[(qn["lamp_num"] == task) & (qn["generator_name"] == gen) & (qn["ranker"] == ranker)
                     & (qn["rerank_method"] == "pl")]
        pl_means = pl_cell.groupby("pl_alpha")["ee_disparity"].mean().to_dict()
        if not pl_means:
            continue
        chosen_alpha = choose_alpha(pl_means, rand_ee_d)
        pl_chosen = pl_cell[pl_cell["pl_alpha"] == chosen_alpha]
        p = welch(rh["expected_utility_norm"].dropna(), pl_chosen["expected_utility_norm"].dropna())
        rows.append({
            "generator": gen, "ranker": ranker, "chosen_pl_alpha": int(chosen_alpha),
            "pl_ee_d": pl_means[chosen_alpha], "pl_eu_norm": pl_chosen["expected_utility_norm"].mean(),
            "pl_n": int(pl_chosen["n_lists"].mean()) if "n_lists" in pl_chosen.columns else None,
            "hybrid_ee_d": rand_ee_d, "hybrid_eu_norm": rh["expected_utility_norm"].mean(),
            "hybrid_n": int(rh["n_lists"].mean()) if "n_lists" in rh.columns else None,
            "delta_eu_norm": rh["expected_utility_norm"].mean() - pl_chosen["expected_utility_norm"].mean(),
            "p_value": p,
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(out.round(4).to_string(index=False))
