"""
Task-robustness (leave-one-task-out) sensitivity analysis for the report's three
core pooled statistics. Answers: could a single noisy LaMP task be carrying (or
masking) any headline coefficient?

Statistics tested (pooled over all four generator x ranker cells, with
generator/ranker/task dummies as nuisance controls - no generator-scaling claims):

  1. TRADEOFF   coef(EE-D)  in EU_norm ~ EE-D + ILD + controls        (all rows)
  2. DIVERSITY  coef(ILD)   in EU_norm ~ ILD + controls               (MMR rows: EE-D fixed)
  3. PL-RESIDUAL coef(is_pl) in EU_norm ~ is_pl + ILD + controls      (PL+MMR rows)

For each: the full-sample estimate, seven leave-one-task-out re-fits, and seven
single-task-only fits. Flags any LOTO fit whose sign differs from the full
sample or whose significance status (p<0.05) flips.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from analysis import normalize_query_rows, fit_ols, select_best_precision

pd.set_option("display.width", 160)

print("Loading run data...")
qn = normalize_query_rows(select_best_precision(pd.DataFrame(build_query_metric_rows(list_run_dirs()))))
print(f"{len(qn)} normalized query rows\n")

qn = qn.copy()
qn["is_base"] = (qn["generator_name"] == "flanT5Base").astype(float)
qn["is_contriever"] = (qn["ranker"] == "contriever").astype(float)
qn["is_pl"] = (qn["rerank_method"] == "pl").astype(float)


def fit(sub, target_feature, extra_feats):
    sub = sub.dropna(subset=[c for c in {target_feature, *extra_feats, "expected_utility_norm"}
                             if c in sub.columns]).copy()
    feats = [target_feature] + list(extra_feats)
    for t in sorted(sub["lamp_num"].dropna().unique())[1:]:  # drop-one coding
        col = f"task_{int(t)}"
        sub[col] = (sub["lamp_num"] == t).astype(float)
        feats.append(col)
    m = fit_ols(sub, feats, "expected_utility_norm")
    return m["coef"][target_feature], m["p_value"][target_feature], m["n"]


STATISTICS = [
    ("TRADEOFF: coef(EE-D), all rows",
     "ee_disparity_norm", ["avg_ild_jaccard_norm", "is_base", "is_contriever"],
     qn["rerank_method"].notna()),
    ("DIVERSITY: coef(ILD), MMR rows only (EE-D fixed)",
     "avg_ild_jaccard_norm", ["is_base", "is_contriever"],
     qn["rerank_method"] == "mmr"),
    ("PL-RESIDUAL: coef(is_pl), PL+MMR rows (ILD controlled)",
     "is_pl", ["avg_ild_jaccard_norm", "is_base", "is_contriever"],
     qn["rerank_method"].isin(["pl", "mmr"])),
]

all_tasks = sorted(qn["lamp_num"].dropna().unique())

for title, feature, controls, mask in STATISTICS:
    print("=" * 90)
    print(title)
    print("=" * 90)
    base_df = qn[mask]
    full_c, full_p, full_n = fit(base_df, feature, controls)
    full_sig = full_p < 0.05
    print(f"FULL SAMPLE: coef = {full_c:+.4f} (p={full_p:.3g}, n={full_n})\n")

    print("Leave-one-task-out:")
    rows = []
    for t in all_tasks:
        c, p, n = fit(base_df[base_df["lamp_num"] != t], feature, controls)
        flags = []
        if np.sign(c) != np.sign(full_c):
            flags.append("SIGN FLIP")
        if (p < 0.05) != full_sig:
            flags.append("significance flips")
        rows.append({"dropped_task": f"LaMP-{int(t)}", "coef": c, "p": p, "n": n,
                     "flag": ", ".join(flags) if flags else ""})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nEach task alone:")
    rows = []
    for t in all_tasks:
        try:
            c, p, n = fit(base_df[base_df["lamp_num"] == t], feature, controls)
            rows.append({"task": f"LaMP-{int(t)}", "coef": c, "p": p, "n": n})
        except ValueError:
            rows.append({"task": f"LaMP-{int(t)}", "coef": np.nan, "p": np.nan, "n": 0})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print()
