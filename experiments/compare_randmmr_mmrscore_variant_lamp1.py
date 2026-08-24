"""
Compares generate_pl_randmmr_lists (rel^alpha * (1-(1-lambda)*max_sim), the
formula that actually produced Table 2/5) against generate_pl_randmmr_mmrscore_lists
(alpha applied to the literal MMR score lambda*rel - (1-lambda)*max_sim as a
whole, then normalized/exponentiated the same way a raw retrieval score is) --
an order-of-operations variant, same alpha/lambda/similarity/tokenization.

EE-D only, at N=100 samples/query, matching how Table 2's EE-D column is
computed (experiments/recompute_paper_eed_at_n100.py). No LLM calls. LaMP-1
only, all 4 (generator, ranker) cells, matched PL alpha=2 (same as Table 2 --
already confirmed via experiments/rematch_alpha_at_n100.py that alpha=2 is the
closest match in every one of the 28 cells, so no need to re-run alpha
selection here).
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
from framework.reranking import generate_pl_lists, generate_pl_randmmr_lists, generate_pl_randmmr_mmrscore_lists
from framework.metrics import compute_ee

TOP_K = 5
SEED = 42
N = 100
CHOSEN_PL_ALPHA = 2  # confirmed via rematch_alpha_at_n100.py: alpha=2 in all 28 cells

CELLS = [
    ("flanT5Base", "bm25"),
    ("flanT5Base", "contriever"),
    ("flanT5Small", "bm25"),
    ("flanT5Small", "contriever"),
]

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


def mean_ee_d(method, kwargs, generator_name, ranker, lamp_num):
    dataset = get_dataset(generator_name, lamp_num)
    retrieval_results = load_retrieval_results(generator_name, ranker, lamp_num)
    rel_fp = dataset.relevance_mapping_path()

    vals = []
    for qid, question, target, all_profiles in dataset.iter_queries():
        ret_for_qid = retrieval_results.get(qid, [])
        if not ret_for_qid:
            continue
        if method == "pl":
            lists = generate_pl_lists(
                retrieval_results_for_qid=ret_for_qid, ranker=ranker,
                pl_samples=N, top_k=TOP_K, seed=SEED, qid=qid, **kwargs,
            )
        else:
            pids_in_order = [p[0] for p in ret_for_qid]
            profiles_in_order = dataset.find_profiles_by_pids(qid, pids_in_order)
            gen_fn = generate_pl_randmmr_lists if method == "pl_randmmr" else generate_pl_randmmr_mmrscore_lists
            lists = gen_fn(
                retrieval_results_for_qid=ret_for_qid, profiles_for_qid=profiles_in_order,
                ranker=ranker, pl_samples=N, top_k=TOP_K, seed=SEED, qid=qid, **kwargs,
            )
        det_indices_per_list = [rl.det_indices for rl in lists]
        ee_res = compute_ee(
            qid=qid, det_indices_per_list=det_indices_per_list,
            retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K,
        )
        if ee_res["ee_disparity"] is not None:
            vals.append(ee_res["ee_disparity"])
    return float(np.mean(vals)), len(vals)


rows = []
for generator, ranker in CELLS:
    print(f"lamp1 {generator}/{ranker} ...", flush=True)
    pl_ee_d, pl_n = mean_ee_d("pl", dict(pl_alpha=CHOSEN_PL_ALPHA), generator, ranker, 1)
    orig_ee_d, orig_n = mean_ee_d(
        "pl_randmmr", dict(pl_alpha=4, lambda_low=0.8, lambda_high=1.0), generator, ranker, 1,
    )
    new_ee_d, new_n = mean_ee_d(
        "pl_randmmr_mmrscore", dict(pl_alpha=4, lambda_low=0.8, lambda_high=1.0), generator, ranker, 1,
    )
    rows.append({
        "generator": generator, "ranker": ranker,
        "pl_ee_d_n100": pl_ee_d,
        "randmmr_orig_ee_d_n100": orig_ee_d,
        "randmmr_mmrscore_ee_d_n100": new_ee_d,
        "n_queries": pl_n,
    })
    print(f"  PL(a={CHOSEN_PL_ALPHA})={pl_ee_d:.4f}  RandMMR(orig)={orig_ee_d:.4f}  "
          f"RandMMR(mmrscore)={new_ee_d:.4f}  n={pl_n}")

out_df = pd.DataFrame(rows)
out_csv_path = os.path.join(ROOT, "report/tables/paper_repro/randmmr_mmrscore_variant_lamp1_eed_n100.csv")
out_df.to_csv(out_csv_path, index=False)
print(f"\nSaved: {out_csv_path}")
print(out_df.round(4).to_string(index=False))
