"""
Compare pl_randmmr_fixed (alpha, lambda_low-lambda_high) against BOTH plain PL(alpha=4)
and PL(alpha=8), for a given LaMP task, strictly N=10 vs N=10 on all sides.
"""
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--lamp_num", type=int, required=True)
parser.add_argument("--alpha", type=float, default=8.0)
parser.add_argument("--lambda_low", type=float, required=True)
parser.add_argument("--lambda_high", type=float, required=True)
args = parser.parse_args()

GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]

run_dirs = list_run_dirs()
all_rows_df = pd.DataFrame(build_query_metric_rows(run_dirs))
all_rows_df = select_full_coverage_runs(all_rows_df)
all_rows_df = all_rows_df[all_rows_df["seed"] == 42]

capped = all_rows_df[(all_rows_df["pl_samples"].isna()) | (all_rows_df["pl_samples"] <= 10)]
ceiling_source_df = capped.copy()

raw_df = capped[
    (capped["lamp_num"] == args.lamp_num)
    & capped["generator_name"].isin(GENERATORS)
    & capped["ranker"].isin(RANKERS)
]

pl4 = raw_df[(raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"] == 4) & (raw_df["pl_samples"] == 10)]
pl8 = raw_df[(raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"] == 8) & (raw_df["pl_samples"] == 10)]
fixed = raw_df[
    (raw_df["rerank_method"] == "pl_randmmr_fixed")
    & (raw_df["pl_alpha"] == args.alpha)
    & (raw_df["pl_randmmr_lambda_low"] == args.lambda_low)
    & (raw_df["pl_randmmr_lambda_high"] == args.lambda_high)
    & (raw_df["pl_samples"] == 10)
]

if fixed.empty:
    print(f"[lamp{args.lamp_num}] No pl_randmmr_fixed rows found for alpha={args.alpha}, lambda={args.lambda_low}-{args.lambda_high}")
    sys.exit(1)

pooled = pd.concat([raw_df[raw_df["rerank_method"] == "deterministic"], pl4, pl8, fixed], ignore_index=True)
norm_cells = pooled[["lamp_num", "generator_name", "ranker"]].drop_duplicates().itertuples(index=False, name=None)
qn = normalize_query_rows_with_gold(pooled, ceiling_source_df, cells=list(norm_cells))
qn["expected_utility_norm"] = pd.to_numeric(qn["expected_utility_norm"], errors="coerce")


def welch(a, b):
    if len(a) >= 2 and len(b) >= 2 and a.nunique() > 1:
        _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    return np.nan


rows = []
for gen in GENERATORS:
    for ranker in RANKERS:
        rf = qn[(qn["generator_name"] == gen) & (qn["ranker"] == ranker) & (qn["rerank_method"] == "pl_randmmr_fixed")]
        rp4 = qn[(qn["generator_name"] == gen) & (qn["ranker"] == ranker) & (qn["rerank_method"] == "pl") & (qn["pl_alpha"] == 4)]
        rp8 = qn[(qn["generator_name"] == gen) & (qn["ranker"] == ranker) & (qn["rerank_method"] == "pl") & (qn["pl_alpha"] == 8)]
        if rf.empty:
            continue
        row = {"lamp_num": args.lamp_num, "generator": gen, "ranker": ranker,
               "fixed_ee_d": rf["ee_disparity"].mean(), "fixed_eu_norm": rf["expected_utility_norm"].mean()}
        if not rp4.empty:
            row["pl4_ee_d"] = rp4["ee_disparity"].mean()
            row["pl4_eu_norm"] = rp4["expected_utility_norm"].mean()
            row["delta_eu_vs_pl4"] = rf["expected_utility_norm"].mean() - rp4["expected_utility_norm"].mean()
            row["p_vs_pl4"] = welch(rf["expected_utility_norm"].dropna(), rp4["expected_utility_norm"].dropna())
        if not rp8.empty:
            row["pl8_ee_d"] = rp8["ee_disparity"].mean()
            row["pl8_eu_norm"] = rp8["expected_utility_norm"].mean()
            row["delta_eu_vs_pl8"] = rf["expected_utility_norm"].mean() - rp8["expected_utility_norm"].mean()
            row["p_vs_pl8"] = welch(rf["expected_utility_norm"].dropna(), rp8["expected_utility_norm"].dropna())
        rows.append(row)

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"\n=== LaMP-{args.lamp_num}: pl_randmmr_fixed(a={args.alpha}, l={args.lambda_low}-{args.lambda_high}) vs PL(4) and PL(8), N=10 vs N=10 ===")
print(out.round(4).to_string(index=False))
