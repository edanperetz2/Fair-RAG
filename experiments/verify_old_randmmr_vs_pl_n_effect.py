"""
Disentangle two confounds behind "old RandMMR beat PL on EE-D":
(1) N effect: does the SAME policy's measured EE-D shift between N=10 and N=20?
(2) Mechanism effect: does the OLD buggy pl_randmmr (fully stochastic tail, fresh
    Gumbel noise every rank) actually beat PL(same alpha) on EE-D even at matched N?
No LLM generation needed -- EE-D only depends on reranking output.
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
from framework.reranking import generate_pl_lists, generate_pl_randmmr_lists
from framework.metrics import compute_ee

LAMP_NUM = 1
GENERATOR = "flanT5Small"
RANKER = "bm25"
TOP_K = 5
SEED = 42
N_QUERIES = 60

cfg = RunConfig(
    dataset=DatasetConfig(lamp_num=LAMP_NUM, lamp_split_type="user", num_queries=None),
    retrieval=RetrievalConfig(ranker=RANKER, top_k=TOP_K),
    rerank=RerankConfig(method="pl"),
    generation=GenerationConfig(generator_name=GENERATOR),
)
dataset = make_dataset(cfg)
retrieval_results = load_retrieval_results(GENERATOR, RANKER, LAMP_NUM)
rel_fp = dataset.relevance_mapping_path()

configs_to_test = {
    "PL(a=4) N=10": ("pl", dict(pl_alpha=4), 10),
    "PL(a=4) N=20": ("pl", dict(pl_alpha=4), 20),
    "OLD pl_randmmr(a=4,l=0.8-1.0) N=10": ("pl_randmmr", dict(pl_alpha=4, lambda_low=0.8, lambda_high=1.0), 10),
    "OLD pl_randmmr(a=4,l=0.8-1.0) N=20": ("pl_randmmr", dict(pl_alpha=4, lambda_low=0.8, lambda_high=1.0), 20),
}

ee_d_per_method = {name: [] for name in configs_to_test}

n_done = 0
for qid, question, target, all_profiles in dataset.iter_queries():
    if n_done >= N_QUERIES:
        break
    ret_for_qid = retrieval_results.get(qid, [])
    if not ret_for_qid:
        continue
    pids_in_order = [p[0] for p in ret_for_qid]
    profiles_in_order = dataset.find_profiles_by_pids(qid, pids_in_order)

    for name, (method, kwargs, n_samples) in configs_to_test.items():
        if method == "pl":
            lists = generate_pl_lists(
                retrieval_results_for_qid=ret_for_qid, ranker=RANKER,
                pl_samples=n_samples, top_k=TOP_K, seed=SEED, qid=qid, **kwargs,
            )
        else:
            lists = generate_pl_randmmr_lists(
                retrieval_results_for_qid=ret_for_qid, profiles_for_qid=profiles_in_order,
                ranker=RANKER, pl_samples=n_samples, top_k=TOP_K, seed=SEED, qid=qid, **kwargs,
            )
        det_indices_per_list = [rl.det_indices for rl in lists]
        ee_res = compute_ee(
            qid=qid, det_indices_per_list=det_indices_per_list,
            retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K,
        )
        if ee_res["ee_disparity"] is not None:
            ee_d_per_method[name].append(ee_res["ee_disparity"])

    n_done += 1

print(f"\n=== N-effect vs mechanism-effect, LaMP-{LAMP_NUM}, {GENERATOR}/{RANKER}, {n_done} queries ===")
for name, vals in ee_d_per_method.items():
    arr = np.array(vals)
    print(f"{name:38s}  mean EE-D = {arr.mean():.4f}   (n_queries={len(arr)})")
