"""
Compare pl_randmmr_fixed (alpha=8, steady lambda=0.85) against matched plain PL,
LaMP-1 and LaMP-4, for whichever (generator, ranker) cells have a completed run.
Matching uses the paper's own closest-EE-D rule (choose_alpha), at N=10 (the real
precision of these new generation runs) on both sides.
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
from analysis.loading import select_full_coverage_runs
from analysis.with_gold_normalization import normalize_query_rows_with_gold

GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]
LAMP_NUMS = [1, 4]
PL_ALPHAS = [1.0, 2.0, 4.0, 8.0]
CLOSE_ENOUGH = 0.02

run_dirs = list_run_dirs()
all_rows_df = pd.DataFrame(build_query_metric_rows(run_dirs))
all_rows_df = select_full_coverage_runs(all_rows_df)
all_rows_df = all_rows_df[all_rows_df["seed"] == 42]

capped = all_rows_df[(all_rows_df["pl_samples"].isna()) | (all_rows_df["pl_samples"] <= 10)]
ceiling_source_df = capped.copy()

raw_df = capped[
    (capped["lamp_num"].isin(LAMP_NUMS))
    & capped["generator_name"].isin(GENERATORS)
    & capped["ranker"].isin(RANKERS)
]

fixed_all = raw_df[
    (raw_df["rerank_method"] == "pl_randmmr_fixed")
    & (raw_df["pl_alpha"] == 8)
    & (raw_df["pl_randmmr_lambda_low"] == 0.85)
    & (raw_df["pl_randmmr_lambda_high"] == 0.85)
    & (raw_df["pl_samples"] == 10)
]

pl_all = raw_df[(raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"].isin(PL_ALPHAS)) & (raw_df["pl_samples"] == 10)]

pooled = pd.concat([raw_df[raw_df["rerank_method"] == "deterministic"], pl_all, fixed_all], ignore_index=True)
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


rows = []
for lamp_num in LAMP_NUMS:
    for gen in GENERATORS:
        for ranker in RANKERS:
            rf = qn[(qn["lamp_num"] == lamp_num) & (qn["generator_name"] == gen) & (qn["ranker"] == ranker)
                    & (qn["rerank_method"] == "pl_randmmr_fixed")]
            if rf.empty:
                continue  # not finished yet
            rand_ee_d = rf["ee_disparity"].mean()

            pl_cell = qn[(qn["lamp_num"] == lamp_num) & (qn["generator_name"] == gen) & (qn["ranker"] == ranker)
                         & (qn["rerank_method"] == "pl")]
            pl_means = pl_cell.groupby("pl_alpha")["ee_disparity"].mean().to_dict()
            if not pl_means:
                continue
            chosen_alpha = choose_alpha(pl_means, rand_ee_d)
            pl_chosen = pl_cell[pl_cell["pl_alpha"] == chosen_alpha]

            p = welch(rf["expected_utility_norm"].dropna(), pl_chosen["expected_utility_norm"].dropna())
            rows.append({
                "lamp_num": lamp_num, "generator": gen, "ranker": ranker,
                "chosen_pl_alpha": int(chosen_alpha),
                "pl_ee_d": pl_means[chosen_alpha], "pl_eu_norm": pl_chosen["expected_utility_norm"].mean(),
                "fixed_ee_d": rand_ee_d, "fixed_eu_norm": rf["expected_utility_norm"].mean(),
                "delta_eu": rf["expected_utility_norm"].mean() - pl_chosen["expected_utility_norm"].mean(),
                "p": p, "n_queries": len(rf),
            })

out = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("\n=== pl_randmmr_fixed(a=8, l=0.85 steady) vs matched PL, LaMP-1 & LaMP-4 (finished settings only) ===")
print(out.round(4).to_string(index=False))
out.to_csv("/private/tmp/claude-501/-Users-asimk-Code-Fair-RAG-1/2ca6f218-27f0-49c7-8fcd-155c4cdafcd3/scratchpad/randmmr_fixed_a8_l085_vs_matched_pl.csv", index=False)
