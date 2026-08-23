"""
Redo the paper's per-cell "closest-EE-D PL alpha" match using N=100 EE-D
estimates (instead of the N=10 estimates the paper actually used), for all 28
cells, to see whether any cell's matched alpha shifts once EE-D is estimated at
higher precision. No LLM calls -- EE-D only depends on reranking output.
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
from framework.metrics import compute_ee

TOP_K = 5
SEED = 42
N = 100
PL_ALPHAS = [1, 2, 4, 8]
CLOSE_ENOUGH = 0.02

paper_csv = pd.read_csv("report/tables/rq3_randmmr_vs_pl_n10_all_cells.csv")
n100_csv = pd.read_csv("report/tables/exploratory_randmmr_vs_pl_eed_n100_recompute.csv")

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


def pl_mean_ee_d(pl_alpha, generator_name, ranker, lamp_num):
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
        ee_res = compute_ee(
            qid=qid, det_indices_per_list=det_indices_per_list,
            retrieval_results_for_qid=ret_for_qid, rel_mapping_fp=rel_fp, top_k=TOP_K,
        )
        if ee_res["ee_disparity"] is not None:
            vals.append(ee_res["ee_disparity"])
    return float(np.mean(vals))


def choose_alpha(cell_pl_means, rand_ee_d):
    diffs = {a: abs(v - rand_ee_d) for a, v in cell_pl_means.items()}
    closest_alpha = min(diffs, key=diffs.get)
    if diffs[closest_alpha] <= CLOSE_ENOUGH:
        return closest_alpha
    above = {a: v for a, v in cell_pl_means.items() if v >= rand_ee_d}
    if above:
        return min(above, key=above.get)
    return closest_alpha


rows = []
for i, row in paper_csv.iterrows():
    lamp_num = int(row["lamp_num"])
    generator = row["generator"]
    ranker = row["ranker"]
    old_alpha = int(row["chosen_pl_alpha"])

    rand_ee_d_n100 = n100_csv[
        (n100_csv["lamp_num"] == lamp_num) & (n100_csv["generator"] == generator) & (n100_csv["ranker"] == ranker)
    ]["randmmr_ee_d_n100"].iloc[0]

    pl_means_n100 = {}
    for a in PL_ALPHAS:
        pl_means_n100[a] = pl_mean_ee_d(a, generator, ranker, lamp_num)

    new_alpha = choose_alpha(pl_means_n100, rand_ee_d_n100)

    row_out = {
        "lamp_num": lamp_num, "generator": generator, "ranker": ranker,
        "old_alpha_n10": old_alpha, "new_alpha_n100": new_alpha,
        "changed": old_alpha != new_alpha,
        "randmmr_ee_d_n100": rand_ee_d_n100,
        **{f"pl_a{a}_ee_d_n100": v for a, v in pl_means_n100.items()},
    }
    rows.append(row_out)
    flag = " <<<< CHANGED" if old_alpha != new_alpha else ""
    print(f"[{i+1}/28] lamp{lamp_num} {generator}/{ranker}: old_alpha={old_alpha} new_alpha={new_alpha}{flag}  "
          f"(rand={rand_ee_d_n100:.4f}, " + ", ".join(f"a{a}={v:.4f}" for a, v in pl_means_n100.items()) + ")")

out_df = pd.DataFrame(rows)
out_path = "/private/tmp/claude-501/-Users-asimk-Code-Fair-RAG-1/2ca6f218-27f0-49c7-8fcd-155c4cdafcd3/scratchpad/rematch_alpha_n100.csv"
out_df.to_csv(out_path, index=False)
print(f"\n{out_df['changed'].sum()}/28 cells changed matched alpha at N=100")
print(f"Saved to {out_path}")
