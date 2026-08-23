"""
Recompute the paper's Table 3 (EE-D-interval utility gain over baseline) using
PER-QUERY EE-D estimated at N=100 reranking samples instead of the N=10 the
paper actually uses -- since binning is done per-query (which bin a query's
samples fall into), this needs a per-query EE-D recompute, not just a cell
mean swap. Utility (normalized_eu) is left completely untouched, still loaded
from the real N=10 generation runs on disk. No LLM calls -- EE-D only depends
on reranking output.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = "/Users/asimk/Code/Fair-RAG-1"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from analysis import select_best_precision
from analysis.loading import select_full_coverage_runs
from analysis.binning import pool_delta_by_bin, bin_label_for_value
from analysis.with_gold_normalization import build_with_gold_ceiling

sys.path.insert(0, os.path.join(ROOT, "experiments"))
from table4_with_gold import load_qid_eu, CELLS, RANKERS  # noqa: E402

from framework.config import DatasetConfig, RunConfig, RetrievalConfig, RerankConfig, GenerationConfig
from framework.dataset import make_dataset
from framework.retrieval import load_retrieval_results
from framework.reranking import generate_pl_lists
from framework.metrics import compute_ee

PL_ALPHAS = [1.0, 2.0, 4.0, 8.0]
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
BIN_LABELS = ["[.0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1.0]"]
TOP_K = 5
SEED = 42
N = 100

# Table 3 is scoped to LaMP-1 and LaMP-4 only (the two tasks with a published baseline)
SCOPE_CELLS = [(ln, gen) for (ln, gen) in CELLS if ln in (1, 4)]

_dataset_cache = {}


def get_dataset(generator_name, lamp_num):
    key = (generator_name, lamp_num)
    if key not in _dataset_cache:
        cfg = RunConfig(
            dataset=DatasetConfig(lamp_num=lamp_num, lamp_split_type="user", num_queries=None),
            retrieval=RetrievalConfig(ranker="bm25", top_k=TOP_K),
            rerank=RerankConfig(method="pl"),
            generation=GenerationConfig(generator_name=generator_name),
        )
        _dataset_cache[key] = make_dataset(cfg)
    return _dataset_cache[key]


def per_query_ee_d_n100(pl_alpha, generator_name, ranker, lamp_num):
    """Returns {qid: ee_disparity} at N=100, no LLM calls."""
    dataset = get_dataset(generator_name, lamp_num)
    retrieval_results = load_retrieval_results(generator_name, ranker, lamp_num)
    rel_fp = dataset.relevance_mapping_path()

    out = {}
    for qid, question, target, all_profiles in dataset.iter_queries():
        ret_for_qid = retrieval_results.get(qid, [])
        if not ret_for_qid:
            continue
        lists = generate_pl_lists(
            retrieval_results_for_qid=ret_for_qid, ranker=ranker,
            pl_alpha=pl_alpha, pl_samples=N, top_k=TOP_K, seed=SEED, qid=qid,
        )
        det_indices_per_list = [rl.det_indices for rl in lists]
        ee_res = compute_ee(
            qid=qid, det_indices_per_list=det_indices_per_list,
            retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K,
        )
        if ee_res["ee_disparity"] is not None:
            out[qid] = ee_res["ee_disparity"]
    return out


def load_qid_mean_eu(run_dir):
    fp = os.path.join(run_dir, "per_list_metrics.jsonl")
    out = defaultdict(list)
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = __import__("json").loads(line)
            out[row["qid"]].append(row["eu_score"])
    return {qid: sum(v) / len(v) for qid, v in out.items()}


def load_ee_disparity(run_dir):
    fp = os.path.join(run_dir, "ee_metrics.jsonl")
    out = {}
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = __import__("json").loads(line)
            out[row["qid"]] = row["ee_disparity"]
    return out


run_dirs = list_run_dirs()
raw_df = pd.DataFrame(build_query_metric_rows(run_dirs))
raw_df = select_full_coverage_runs(raw_df)
raw_df = select_best_precision(raw_df)
raw_df = raw_df[raw_df["seed"] == 42]

all_rows = []
for lamp_num, gen in SCOPE_CELLS:
    for ranker in RANKERS:
        print(f"lamp{lamp_num} {gen}/{ranker} ...")
        det_sub = raw_df[
            (raw_df["lamp_num"] == lamp_num) & (raw_df["generator_name"] == gen)
            & (raw_df["ranker"] == ranker) & (raw_df["rerank_method"] == "deterministic")
        ]
        det_run_dirs = det_sub["run_dir"].drop_duplicates().tolist()
        assert len(det_run_dirs) == 1, f"expected 1 det run for lamp{lamp_num}/{gen}/{ranker}"
        det_eu = load_qid_eu(det_run_dirs[0], max_only=True)
        det_eed = load_ee_disparity(det_run_dirs[0])  # always 1.0, N-independent

        per_alpha_mean_eu = {}
        per_alpha_eed_n100 = {}
        for alpha in PL_ALPHAS:
            pl_sub = raw_df[
                (raw_df["lamp_num"] == lamp_num) & (raw_df["generator_name"] == gen)
                & (raw_df["ranker"] == ranker) & (raw_df["rerank_method"] == "pl")
                & (raw_df["pl_alpha"] == alpha)
            ]
            pl_run_dirs = pl_sub["run_dir"].drop_duplicates().tolist()
            assert len(pl_run_dirs) == 1, f"expected 1 pl(a={alpha}) run for lamp{lamp_num}/{gen}/{ranker}"
            per_alpha_mean_eu[alpha] = load_qid_mean_eu(pl_run_dirs[0])  # UNCHANGED, real N=10 EU
            per_alpha_eed_n100[alpha] = per_query_ee_d_n100(alpha, gen, ranker, lamp_num)  # NEW, N=100

        ceiling = build_with_gold_ceiling(raw_df, lamp_num, gen, ranker)

        for qid in det_eu:
            if qid not in ceiling:
                continue
            denom = ceiling[qid]
            norm = det_eu[qid] / denom if denom > 0 else 1.0
            all_rows.append({
                "lamp_num": lamp_num, "generator": gen, "ranker": ranker,
                "rerank_method": "deterministic", "qid": qid,
                "ee_disparity": det_eed.get(qid, 1.0), "normalized_eu": norm,
            })

        for alpha in PL_ALPHAS:
            for qid, mean_u in per_alpha_mean_eu[alpha].items():
                if qid not in ceiling:
                    continue
                eed = per_alpha_eed_n100[alpha].get(qid, None)
                if eed is None:
                    continue
                denom = ceiling[qid]
                norm = mean_u / denom if denom > 0 else 1.0
                all_rows.append({
                    "lamp_num": lamp_num, "generator": gen, "ranker": ranker,
                    "rerank_method": "pl", "qid": qid, "pl_alpha": alpha,
                    "ee_disparity": eed, "normalized_eu": norm,
                })

long_df = pd.DataFrame(all_rows)
long_df = long_df.dropna(subset=["ee_disparity", "normalized_eu"])

result = pool_delta_by_bin(
    long_df, metric_col="ee_disparity", bin_edges=BIN_EDGES, bin_labels=BIN_LABELS,
    baseline_method="deterministic", comparison_methods=["pl"],
    method_col="rerank_method", value_col="normalized_eu",
    group_by=["lamp_num", "ranker", "generator"], bin_baseline=False,
)

sig_rows = []
for (lamp_num, ranker, gen), part in long_df.groupby(["lamp_num", "ranker", "generator"]):
    baseline_vals = part.loc[part.rerank_method == "deterministic", "normalized_eu"].dropna()
    for bin_label in BIN_LABELS:
        comp_vals = part.loc[
            (part.rerank_method == "pl") & (part.ee_disparity.apply(
                lambda v: bin_label_for_value(v, BIN_EDGES, BIN_LABELS)
            ) == bin_label), "normalized_eu",
        ].dropna()
        if len(comp_vals) >= 2 and len(baseline_vals) >= 2:
            _, p = scipy_stats.ttest_ind(comp_vals, baseline_vals, equal_var=False)
        else:
            p = None
        sig_rows.append({"lamp_num": lamp_num, "ranker": ranker, "generator": gen, "bin": bin_label, "p_value": p, "n": len(comp_vals)})
sig_df = pd.DataFrame(sig_rows)

result = result.merge(sig_df, on=["lamp_num", "ranker", "generator", "bin"], how="left")
result["sig"] = result["p_value"].apply(lambda p: p is not None and p < 0.05)

out_fp = "/private/tmp/claude-501/-Users-asimk-Code-Fair-RAG-1/2ca6f218-27f0-49c7-8fcd-155c4cdafcd3/scratchpad/table3_eed_interval_n100.csv"
result.to_csv(out_fp, index=False)

GEN_ORDER = {"flanT5Base": 0, "flanT5Small": 1}
RANK_ORDER = {"bm25": 0, "contriever": 1}
GEN_SHORT = {"flanT5Base": "Base", "flanT5Small": "Small"}
RANKER_SHORT = {"bm25": "BM25", "contriever": "Contr."}

for lamp_num in sorted(set(c[0] for c in SCOPE_CELLS)):
    print(f"\n=== LaMP-{lamp_num} (N=100 EE-D binning) ===")
    rows = sorted(
        [(r, g) for (ln, g) in SCOPE_CELLS if ln == lamp_num for r in RANKERS],
        key=lambda rg: (GEN_ORDER[rg[1]], RANK_ORDER[rg[0]]),
    )
    for ranker, gen in rows:
        sub = result[(result.lamp_num == lamp_num) & (result.ranker == ranker) & (result.generator == gen)]
        sub = sub.set_index("bin").reindex(BIN_LABELS)
        vals = []
        for bl in BIN_LABELS:
            d = sub.loc[bl, "delta"]
            s = sub.loc[bl, "sig"]
            n = sub.loc[bl, "n"]
            if pd.isna(d):
                vals.append("--")
            else:
                mark = "*" if s else ""
                vals.append(f"{d:+.2f}{mark}(n={int(n)})")
        print(f"{RANKER_SHORT[ranker]}+{GEN_SHORT[gen]:<6}: {'  '.join(vals)}")

print(f"\nSaved: {out_fp}")
