"""
Compare pl_randmmr_stochastic (alpha, lambda_low-lambda_high, tau) against plain
PL(alpha=8), for a given LaMP task, strictly N=10 vs N=10 on both sides.
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
parser.add_argument("--alpha", type=float, default=4.0)
parser.add_argument("--lambda_low", type=float, default=0.7)
parser.add_argument("--lambda_high", type=float, default=1.0)
parser.add_argument("--tau", type=float, required=True)
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

pl8 = raw_df[(raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"] == 8) & (raw_df["pl_samples"] == 10)]
stoch = raw_df[
    (raw_df["rerank_method"] == "pl_randmmr_stochastic")
    & (raw_df["pl_alpha"] == args.alpha)
    & (raw_df["pl_randmmr_lambda_low"] == args.lambda_low)
    & (raw_df["pl_randmmr_lambda_high"] == args.lambda_high)
    & (raw_df["pl_randmmr_tau"] == args.tau)
    & (raw_df["pl_samples"] == 10)
]

if stoch.empty:
    print(f"[lamp{args.lamp_num}] No pl_randmmr_stochastic rows found for alpha={args.alpha}, lambda={args.lambda_low}-{args.lambda_high}, tau={args.tau}")
    sys.exit(1)
if pl8.empty:
    print(f"[lamp{args.lamp_num}] No PL(alpha=8, N=10) rows found")
    sys.exit(1)

pooled = pd.concat([raw_df[raw_df["rerank_method"] == "deterministic"], pl8, stoch], ignore_index=True)
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
        rp = qn[(qn["generator_name"] == gen) & (qn["ranker"] == ranker) & (qn["rerank_method"] == "pl")]
        rs = qn[(qn["generator_name"] == gen) & (qn["ranker"] == ranker) & (qn["rerank_method"] == "pl_randmmr_stochastic")]
        if rp.empty or rs.empty:
            continue
        p = welch(rs["expected_utility_norm"].dropna(), rp["expected_utility_norm"].dropna())
        rows.append({
            "lamp_num": args.lamp_num, "generator": gen, "ranker": ranker,
            "pl8_ee_d": rp["ee_disparity"].mean(), "pl8_eu_norm": rp["expected_utility_norm"].mean(),
            "stoch_ee_d": rs["ee_disparity"].mean(), "stoch_eu_norm": rs["expected_utility_norm"].mean(),
            "delta_eu": rs["expected_utility_norm"].mean() - rp["expected_utility_norm"].mean(),
            "p": p,
        })

out = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(f"\n=== LaMP-{args.lamp_num}: pl_randmmr_stochastic(a={args.alpha}, l={args.lambda_low}-{args.lambda_high}, tau={args.tau}) vs PL(alpha=8), N=10 vs N=10 ===")
print(out.round(4).to_string(index=False))
print(f"mean stoch EE-D: {out['stoch_ee_d'].mean():.4f}   mean PL8 EE-D: {out['pl8_ee_d'].mean():.4f}")
