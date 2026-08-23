"""
Compare pl_randmmr_fixed (alpha=8, lambda~Uniform(0.6,1.0)) against plain PL(alpha=8),
LaMP-1 only, strictly at N=10 samples/query on both sides (no higher-N runs mixed in,
even where they exist for a given cell).
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

print("Loading run directories...")
run_dirs = list_run_dirs()
all_rows_df = pd.DataFrame(build_query_metric_rows(run_dirs))
all_rows_df = select_full_coverage_runs(all_rows_df)
all_rows_df = all_rows_df[all_rows_df["seed"] == 42]

# ceiling/gold normalization source: use every N=10 row available (best precision,
# but capped at 10 -- per instruction, pretend no method ever had more than 10 samples)
capped = all_rows_df[(all_rows_df["pl_samples"].isna()) | (all_rows_df["pl_samples"] <= 10)]
ceiling_source_df = capped.copy()

raw_df = capped[
    (capped["lamp_num"] == 1)
    & capped["generator_name"].isin(GENERATORS)
    & capped["ranker"].isin(RANKERS)
]

pl8 = raw_df[(raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"] == 8) & (raw_df["pl_samples"] == 10)]
fixed = raw_df[
    (raw_df["rerank_method"] == "pl_randmmr_fixed")
    & (raw_df["pl_alpha"] == 8)
    & (raw_df["pl_randmmr_lambda_low"] == 0.6)
    & (raw_df["pl_randmmr_lambda_high"] == 1.0)
    & (raw_df["pl_samples"] == 10)
]

pooled = pd.concat([raw_df[raw_df["rerank_method"] == "deterministic"], pl8, fixed], ignore_index=True)
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
        rf = qn[(qn["generator_name"] == gen) & (qn["ranker"] == ranker) & (qn["rerank_method"] == "pl_randmmr_fixed")]
        if rp.empty or rf.empty:
            continue
        p = welch(rf["expected_utility_norm"].dropna(), rp["expected_utility_norm"].dropna())
        rows.append({
            "generator": gen, "ranker": ranker,
            "pl8_ee_d": rp["ee_disparity"].mean(), "pl8_eu_norm": rp["expected_utility_norm"].mean(),
            "pl8_n": rp["n_lists"].mean() if "n_lists" in rp.columns else None,
            "fixed_ee_d": rf["ee_disparity"].mean(), "fixed_eu_norm": rf["expected_utility_norm"].mean(),
            "fixed_n": rf["n_lists"].mean() if "n_lists" in rf.columns else None,
            "delta_eu": rf["expected_utility_norm"].mean() - rp["expected_utility_norm"].mean(),
            "p": p,
        })

out = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(out.round(4).to_string(index=False))
