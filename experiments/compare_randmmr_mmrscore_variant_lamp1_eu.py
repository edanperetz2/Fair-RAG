"""
EU comparison for the pl_randmmr_mmrscore variant vs. Table 2's existing
PL(alpha=2) and original-RandMMR numbers, LaMP-1 only, all 4 (generator,
ranker) cells. Reuses the already-recomputed N=10 PL/original-RandMMR
per-query rows in report/tables/rq3_n10_recomputed_query_rows.csv (verified
earlier to reproduce Table 2 exactly) rather than redoing that work; computes
the equivalent per-query rows for the new variant's freshly-generated N=10
runs the same way (framework.metrics.compute_ee across all 10 samples,
EU/ILD averaged over the same 10), then normalizes everything together with
analysis.with_gold_normalization.normalize_query_rows_with_gold so all three
methods share one ceiling pool, exactly as rq3_randmmr_vs_pl_n10.py does.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from framework.retrieval import load_retrieval_results
from framework.metrics import compute_ee
from analysis import select_best_precision
from analysis.loading import select_full_coverage_runs
from analysis.with_gold_normalization import normalize_query_rows_with_gold

TOP_K = 5
CELLS = [
    ("flanT5Base", "bm25"),
    ("flanT5Base", "contriever"),
    ("flanT5Small", "bm25"),
    ("flanT5Small", "contriever"),
]

# --- EE-D at N=100, already computed (compare_randmmr_mmrscore_variant_lamp1.py) ---
eed_n100 = pd.read_csv("report/tables/paper_repro/randmmr_mmrscore_variant_lamp1_eed_n100.csv")


def relevance_mapping_path(lamp_num, generator_name):
    return os.path.join(ROOT, "data", f"lamp_utility_labels_{generator_name}", f"{lamp_num}_relevance_mapping.tsv")


def load_jsonl(fp):
    rows = []
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_retrieval_cache = {}
def get_retrieval_results(generator_name, ranker, lamp_num):
    key = (generator_name, ranker, lamp_num)
    if key not in _retrieval_cache:
        _retrieval_cache[key] = load_retrieval_results(generator_name=generator_name, ranker=ranker, lamp_num=lamp_num)
    return _retrieval_cache[key]


def recompute_run_rows(run_dir):
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    cfg = manifest.get("config", {})
    generator_name = cfg.get("generation", {}).get("generator_name")
    ranker = cfg.get("retrieval", {}).get("ranker")
    lamp_num = cfg.get("dataset", {}).get("lamp_num")
    method = cfg.get("rerank", {}).get("method")
    pl_alpha = cfg.get("rerank", {}).get("pl_alpha")
    lam_lo = cfg.get("rerank", {}).get("pl_randmmr_lambda_low")
    lam_hi = cfg.get("rerank", {}).get("pl_randmmr_lambda_high")

    retrieval_results = get_retrieval_results(generator_name, ranker, lamp_num)
    rel_fp = relevance_mapping_path(lamp_num, generator_name)

    lists_by_qid = {}
    for row in load_jsonl(os.path.join(run_dir, "retrieval_lists.jsonl")):
        lists_by_qid.setdefault(row["qid"], []).append(row)
    metrics_by_list_id = {
        row["list_id"]: row for row in load_jsonl(os.path.join(run_dir, "per_list_metrics.jsonl"))
    }

    out_rows = []
    for qid, rows in lists_by_qid.items():
        if qid not in retrieval_results:
            continue
        subset = sorted(rows, key=lambda r: r.get("sample_idx", 0))
        det_indices_per_list = [r["det_indices"] for r in subset]
        ee_res = compute_ee(
            qid=qid, det_indices_per_list=det_indices_per_list,
            retrieval_results_for_qid=retrieval_results.get(qid, []),
            rel_mapping_fp=rel_fp, top_k=TOP_K,
        )
        eu_scores, ild_scores = [], []
        for r in subset:
            m = metrics_by_list_id.get(r["list_id"])
            if m is None:
                continue
            eu_scores.append(m["eu_score"])
            ild_scores.append(m.get("ild_jaccard"))
        if not eu_scores:
            continue
        out_rows.append({
            "qid": qid, "lamp_num": int(lamp_num), "generator_name": generator_name, "ranker": ranker,
            "rerank_method": method, "pl_alpha": pl_alpha,
            "pl_randmmr_lambda_low": lam_lo, "pl_randmmr_lambda_high": lam_hi,
            "expected_utility": float(np.mean(eu_scores)),
            "max_utility": float(np.max(eu_scores)), "min_utility": float(np.min(eu_scores)),
            "avg_ild_jaccard": float(np.mean([v for v in ild_scores if v is not None])) if any(v is not None for v in ild_scores) else None,
            "ee_disparity": ee_res["ee_disparity"], "ee_relevance": ee_res["ee_relevance"],
            "ee_difference": ee_res["ee_difference"], "n_lists": len(subset),
        })
    return out_rows


print("Loading all run dirs (for ceiling pool, now including the new mmrscore runs)...")
run_dirs = list_run_dirs()
all_rows_df = pd.DataFrame(build_query_metric_rows(run_dirs))
all_rows_df = select_full_coverage_runs(all_rows_df)
all_rows_df = all_rows_df[all_rows_df["seed"] == 42]

ceiling_source_df = select_best_precision(all_rows_df.copy())

raw_df = all_rows_df[
    all_rows_df["generator_name"].isin([g for g, r in CELLS]) & all_rows_df["ranker"].isin([r for g, r in CELLS])
]
raw_df = select_best_precision(raw_df)

# --- existing PL(alpha=2) + original RandMMR N=10 rows, filtered to LaMP-1 ---
# dtype=str on qid is essential: pandas otherwise infers qid as numeric and
# strips leading zeros ("010" -> "10"), which silently breaks every ceiling
# lookup against the JSONL-sourced (string, zero-padded) qids below and makes
# every row fall back to normalize_query_rows_with_gold's denom-missing ->
# 1.0 default.
existing_n10 = pd.read_csv("report/tables/rq3_n10_recomputed_query_rows.csv", dtype={"qid": str})
existing_n10 = existing_n10[existing_n10["lamp_num"] == 1]
pl_rows = existing_n10[(existing_n10["rerank_method"] == "pl") & (existing_n10["pl_alpha"] == 2)]
orig_randmmr_rows = existing_n10[
    (existing_n10["rerank_method"] == "pl_randmmr") & (existing_n10["pl_alpha"] == 4)
    & (existing_n10["pl_randmmr_lambda_low"] == 0.8) & (existing_n10["pl_randmmr_lambda_high"] == 1.0)
]
det_rows = raw_df[(raw_df["lamp_num"] == 1) & (raw_df["rerank_method"] == "deterministic")].copy()

# --- new variant rows, freshly computed from the just-generated run dirs ---
new_run_dirs = raw_df[
    (raw_df["lamp_num"] == 1) & (raw_df["rerank_method"] == "pl_randmmr_mmrscore")
    & (raw_df["pl_alpha"] == 4) & (raw_df["pl_randmmr_lambda_low"] == 0.8) & (raw_df["pl_randmmr_lambda_high"] == 1.0)
]["run_dir"].unique().tolist()
print(f"{len(new_run_dirs)} new pl_randmmr_mmrscore run dirs found for LaMP-1")
new_rows = []
for rd in new_run_dirs:
    new_rows.extend(recompute_run_rows(rd))
new_df = pd.DataFrame(new_rows)

pooled = pd.concat([det_rows, pl_rows, orig_randmmr_rows, new_df], ignore_index=True)
cells_triples = [(1, g, r) for g, r in CELLS]
qn = normalize_query_rows_with_gold(pooled, ceiling_source_df, cells=cells_triples)
qn["expected_utility_norm"] = pd.to_numeric(qn["expected_utility_norm"], errors="coerce")


def welch(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) >= 2 and len(b) >= 2 and a.nunique() > 1:
        _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    return np.nan


rows = []
for gen, ranker in CELLS:
    pl_cell = qn[(qn.generator_name == gen) & (qn.ranker == ranker) & (qn.rerank_method == "pl")]
    orig_cell = qn[(qn.generator_name == gen) & (qn.ranker == ranker) & (qn.rerank_method == "pl_randmmr")]
    new_cell = qn[(qn.generator_name == gen) & (qn.ranker == ranker) & (qn.rerank_method == "pl_randmmr_mmrscore")]
    eed_row = eed_n100[(eed_n100.generator == gen) & (eed_n100.ranker == ranker)].iloc[0]

    pl_eu = pl_cell["expected_utility_norm"].mean()
    orig_eu = orig_cell["expected_utility_norm"].mean()
    new_eu = new_cell["expected_utility_norm"].mean()

    rows.append({
        "generator": gen, "ranker": ranker,
        "pl_ee_d_n100": eed_row["pl_ee_d_n100"], "pl_eu_norm_n10": pl_eu,
        "randmmr_orig_ee_d_n100": eed_row["randmmr_orig_ee_d_n100"], "randmmr_orig_eu_norm_n10": orig_eu,
        "randmmr_mmrscore_ee_d_n100": eed_row["randmmr_mmrscore_ee_d_n100"], "randmmr_mmrscore_eu_norm_n10": new_eu,
        "delta_eu_orig_vs_pl": orig_eu - pl_eu,
        "delta_eu_mmrscore_vs_pl": new_eu - pl_eu,
        "delta_eu_mmrscore_vs_orig": new_eu - orig_eu,
        "p_mmrscore_vs_orig": welch(new_cell["expected_utility_norm"], orig_cell["expected_utility_norm"]),
        "n_new": len(new_cell), "n_orig": len(orig_cell), "n_pl": len(pl_cell),
    })

out = pd.DataFrame(rows)
out_path = "report/tables/paper_repro/randmmr_mmrscore_variant_lamp1_comparison.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}\n")
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
