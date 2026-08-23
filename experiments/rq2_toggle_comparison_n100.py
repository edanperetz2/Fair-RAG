"""
EE-D-only (N=100, no LLM) comparison across RandMMR's three design toggles, for
the new Section 5.2 rewrite. Scope: LaMP-1 and LaMP-4 (matching Table 2's scope,
the only tasks with a real EU comparison).

Methods compared, all at rank-1 alpha=8 (except the old-buggy "all-ranks-PL"
reference, which is alpha=4/PL(2)-matched -- already in the paper's Table 2 at
N=100, not recomputed here):
  - PL(alpha=8): plain PL, every rank resampled
  - fixed_steady: rank 1 = PL(8), ranks 2..k = deterministic MMR argmax, ONE
    fixed lambda=0.85 for every rank/sample ("deterministic lambda" toggle)
  - fixed_ranged: rank 1 = PL(8), ranks 2..k = deterministic MMR argmax, a
    FRESH lambda ~ Uniform(0.6,1.0) drawn per rank ("uniformly sampled lambda"
    toggle)
Also recomputes PL(alpha=2) at N=100 for the same 8 cells, for direct reference
alongside the paper's existing old-buggy-RandMMR-vs-PL(2) comparison.
"""
import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from framework.config import DatasetConfig, RunConfig, RetrievalConfig, RerankConfig, GenerationConfig
from framework.dataset import make_dataset
from framework.retrieval import load_retrieval_results
from framework.reranking import generate_pl_lists
from experiments._fast_randmmr_fixed_reranking import generate_pl_randmmr_fixed_lists_fast as generate_pl_randmmr_fixed_lists
from framework.metrics import compute_ee

TOP_K = 5
SEED = 42
N = 100
GENERATORS = ["flanT5Base", "flanT5Small"]
RANKERS = ["bm25", "contriever"]
LAMP_NUMS = [1, 4]

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


def mean_ee_d_pl(pl_alpha, generator_name, ranker, lamp_num):
    dataset = get_dataset(generator_name, lamp_num)
    retrieval_results = load_retrieval_results(generator_name, ranker, lamp_num)
    rel_fp = dataset.relevance_mapping_path()
    vals = []
    for qid, question, target, all_profiles in dataset.iter_queries():
        ret_for_qid = retrieval_results.get(qid, [])
        if not ret_for_qid:
            continue
        lists = generate_pl_lists(
            retrieval_results_for_qid=ret_for_qid, ranker=ranker,
            pl_alpha=pl_alpha, pl_samples=N, top_k=TOP_K, seed=SEED, qid=qid,
        )
        det_indices_per_list = [rl.det_indices for rl in lists]
        ee_res = compute_ee(qid=qid, det_indices_per_list=det_indices_per_list,
                             retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K)
        if ee_res["ee_disparity"] is not None:
            vals.append(ee_res["ee_disparity"])
    return float(np.mean(vals))


def mean_ee_d_fixed(pl_alpha, lambda_low, lambda_high, generator_name, ranker, lamp_num):
    dataset = get_dataset(generator_name, lamp_num)
    retrieval_results = load_retrieval_results(generator_name, ranker, lamp_num)
    rel_fp = dataset.relevance_mapping_path()
    vals = []
    for qid, question, target, all_profiles in dataset.iter_queries():
        ret_for_qid = retrieval_results.get(qid, [])
        if not ret_for_qid:
            continue
        pids_in_order = [p[0] for p in ret_for_qid]
        profiles_in_order = dataset.find_profiles_by_pids(qid, pids_in_order)
        lists = generate_pl_randmmr_fixed_lists(
            retrieval_results_for_qid=ret_for_qid, profiles_for_qid=profiles_in_order,
            ranker=ranker, pl_alpha=pl_alpha, lambda_low=lambda_low, lambda_high=lambda_high,
            pl_samples=N, top_k=TOP_K, seed=SEED, qid=qid,
        )
        det_indices_per_list = [rl.det_indices for rl in lists]
        ee_res = compute_ee(qid=qid, det_indices_per_list=det_indices_per_list,
                             retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K)
        if ee_res["ee_disparity"] is not None:
            vals.append(ee_res["ee_disparity"])
    return float(np.mean(vals))


rows = []
for lamp_num in LAMP_NUMS:
    for gen in GENERATORS:
        for ranker in RANKERS:
            pl2 = mean_ee_d_pl(2, gen, ranker, lamp_num)
            pl8 = mean_ee_d_pl(8, gen, ranker, lamp_num)
            steady = mean_ee_d_fixed(8, 0.85, 0.85, gen, ranker, lamp_num)
            ranged = mean_ee_d_fixed(8, 0.6, 1.0, gen, ranker, lamp_num)
            row = {
                "lamp_num": lamp_num, "generator": gen, "ranker": ranker,
                "pl2_ee_d": pl2, "pl8_ee_d": pl8,
                "fixed_steady_ee_d": steady, "fixed_ranged_ee_d": ranged,
            }
            rows.append(row)
            print(f"lamp{lamp_num} {gen}/{ranker}: PL2={pl2:.4f} PL8={pl8:.4f} "
                  f"fixed_steady(a8,l0.85)={steady:.4f} fixed_ranged(a8,l0.6-1.0)={ranged:.4f}")

out = pd.DataFrame(rows)
out_fp = "/private/tmp/claude-501/-Users-asimk-Code-Fair-RAG-1/2ca6f218-27f0-49c7-8fcd-155c4cdafcd3/scratchpad/rq2_toggle_comparison_n100.csv"
out.to_csv(out_fp, index=False)
print(f"\nSaved: {out_fp}")
print("\n=== Means across 8 cells ===")
print(out[["pl2_ee_d", "pl8_ee_d", "fixed_steady_ee_d", "fixed_ranged_ee_d"]].mean().round(4))
