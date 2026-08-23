"""
EE-D-only (N=100, no LLM) for two new/changed Table 3 rows, same single cell as
the rest of Table 3 (LaMP-1, Flan-T5-Base/BM25):
  - "all k" mechanism (pl_randmmr, mechanism A), alpha=4, lambda FIXED at 0.9
    (steady, replacing/adding to the existing lambda~U(0.8,1.0) row).
  - "rank 1 only" mechanism (pl_randmmr_fixed, mechanism B), alpha=8, lambda
    FIXED at 0.8 (replacing the existing 0.85-fixed row).
"""
import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from framework.config import DatasetConfig, RunConfig, RetrievalConfig, RerankConfig, GenerationConfig
from framework.dataset import make_dataset
from framework.retrieval import load_retrieval_results
from experiments._fast_randmmr_reranking import generate_pl_randmmr_lists_fast
from experiments._fast_randmmr_fixed_reranking import generate_pl_randmmr_fixed_lists_fast
from framework.metrics import compute_ee

TOP_K = 5
SEED = 42
N = 100
GENERATOR = "flanT5Base"
RANKER = "bm25"
LAMP_NUM = 1

cfg = RunConfig(
    dataset=DatasetConfig(lamp_num=LAMP_NUM, lamp_split_type="user", num_queries=None),
    retrieval=RetrievalConfig(ranker=RANKER, top_k=TOP_K),
    rerank=RerankConfig(method="pl"),
    generation=GenerationConfig(generator_name=GENERATOR),
)
dataset = make_dataset(cfg)
retrieval_results = load_retrieval_results(GENERATOR, RANKER, LAMP_NUM)
rel_fp = dataset.relevance_mapping_path()


def mean_ee_d_all_k(pl_alpha, lambda_fixed):
    vals = []
    for qid, question, target, all_profiles in dataset.iter_queries():
        ret_for_qid = retrieval_results.get(qid, [])
        if not ret_for_qid:
            continue
        pids_in_order = [p[0] for p in ret_for_qid]
        profiles_in_order = dataset.find_profiles_by_pids(qid, pids_in_order)
        lists = generate_pl_randmmr_lists_fast(
            retrieval_results_for_qid=ret_for_qid, profiles_for_qid=profiles_in_order,
            ranker=RANKER, pl_alpha=pl_alpha, lambda_low=lambda_fixed, lambda_high=lambda_fixed,
            pl_samples=N, top_k=TOP_K, seed=SEED, qid=qid,
        )
        det_indices_per_list = [rl.det_indices for rl in lists]
        ee_res = compute_ee(qid=qid, det_indices_per_list=det_indices_per_list,
                             retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K)
        if ee_res["ee_disparity"] is not None:
            vals.append(ee_res["ee_disparity"])
    return float(np.mean(vals))


def mean_ee_d_rank1_only(pl_alpha, lambda_fixed):
    vals = []
    for qid, question, target, all_profiles in dataset.iter_queries():
        ret_for_qid = retrieval_results.get(qid, [])
        if not ret_for_qid:
            continue
        pids_in_order = [p[0] for p in ret_for_qid]
        profiles_in_order = dataset.find_profiles_by_pids(qid, pids_in_order)
        lists = generate_pl_randmmr_fixed_lists_fast(
            retrieval_results_for_qid=ret_for_qid, profiles_for_qid=profiles_in_order,
            ranker=RANKER, pl_alpha=pl_alpha, lambda_low=lambda_fixed, lambda_high=lambda_fixed,
            pl_samples=N, top_k=TOP_K, seed=SEED, qid=qid,
        )
        det_indices_per_list = [rl.det_indices for rl in lists]
        ee_res = compute_ee(qid=qid, det_indices_per_list=det_indices_per_list,
                             retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K)
        if ee_res["ee_disparity"] is not None:
            vals.append(ee_res["ee_disparity"])
    return float(np.mean(vals))


all_k_fixed09 = mean_ee_d_all_k(pl_alpha=4, lambda_fixed=0.9)
print(f"all_k, alpha=4, lambda=0.9 fixed: EE-D={all_k_fixed09:.4f}")

rank1_fixed08 = mean_ee_d_rank1_only(pl_alpha=8, lambda_fixed=0.8)
print(f"rank1_only, alpha=8, lambda=0.8 fixed: EE-D={rank1_fixed08:.4f}")
