"""
Verify the RandMMR-fixed vs. PL EE-D asymmetry (RandMMR(a=8) << PL(8), but
RandMMR(a=4) >> PL(4)) at N=100 samples/query, WITHOUT any LLM generation --
EE-D only depends on which documents land in which positions across the
sampled lists, not on generated text, so this can be checked directly from
the reranking + EE computation, skipping the (expensive) generation step
entirely.
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
from framework.reranking import generate_pl_lists, generate_pl_randmmr_fixed_lists
from framework.metrics import compute_ee

LAMP_NUM = 1
GENERATOR = "flanT5Small"
RANKER = "bm25"
TOP_K = 5
N_SAMPLES = 100
SEED = 42
N_QUERIES = 60  # subset for speed; EE-D is a per-query mean so this is a fair estimate

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
    "PL(a=4)": ("pl", dict(pl_alpha=4)),
    "PL(a=8)": ("pl", dict(pl_alpha=8)),
    "RandMMR-fixed(a=4,l=0.85)": ("pl_randmmr_fixed", dict(pl_alpha=4, lambda_low=0.85, lambda_high=0.85)),
    "RandMMR-fixed(a=8,l=0.85)": ("pl_randmmr_fixed", dict(pl_alpha=8, lambda_low=0.85, lambda_high=0.85)),
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

    for name, (method, kwargs) in configs_to_test.items():
        if method == "pl":
            lists = generate_pl_lists(
                retrieval_results_for_qid=ret_for_qid, ranker=RANKER,
                pl_samples=N_SAMPLES, top_k=TOP_K, seed=SEED, qid=qid, **kwargs,
            )
        else:
            lists = generate_pl_randmmr_fixed_lists(
                retrieval_results_for_qid=ret_for_qid, profiles_for_qid=profiles_in_order,
                ranker=RANKER, pl_samples=N_SAMPLES, top_k=TOP_K, seed=SEED, qid=qid, **kwargs,
            )
        det_indices_per_list = [rl.det_indices for rl in lists]
        ee_res = compute_ee(
            qid=qid, det_indices_per_list=det_indices_per_list,
            retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K,
        )
        if ee_res["ee_disparity"] is not None:
            ee_d_per_method[name].append(ee_res["ee_disparity"])

    n_done += 1

print(f"\n=== EE-D at N={N_SAMPLES}, LaMP-{LAMP_NUM}, {GENERATOR}/{RANKER}, {n_done} queries ===")
for name, vals in ee_d_per_method.items():
    arr = np.array(vals)
    print(f"{name:30s}  mean EE-D = {arr.mean():.4f}   (n_queries={len(arr)})")
