"""
Recompute EE-D at N=100 (decoupled from EU, which stays exactly as already
reported in report/tables/rq3_randmmr_vs_pl_n10_all_cells.csv, i.e. from the
real N=10 LLM-generation runs) for every one of the 28 cells currently in the
paper's Table 2 / Appendix B (tab:hybrid / tab:hybrid-full): old buggy
pl_randmmr(alpha=4, lambda~Uniform(0.8,1.0)) vs. its matched plain PL(chosen_pl_alpha),
per (lamp_num, generator, ranker). No LLM calls -- EE-D only depends on reranking
output, so this is purely retrieval + reranking + expeval.

This does NOT modify the paper or any of its tables -- it's a standalone check to
see how much the EE-D numbers currently in the paper would shift under a
higher-precision (N=100) estimate.
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
from experiments._fast_randmmr_reranking import generate_pl_randmmr_lists_fast as generate_pl_randmmr_lists

TOP_K = 5
SEED = 42
N = 100

paper_csv = pd.read_csv("report/tables/rq3_randmmr_vs_pl_n10_all_cells.csv")

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
            lists = generate_pl_randmmr_lists(
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
for i, row in paper_csv.iterrows():
    lamp_num = int(row["lamp_num"])
    generator = row["generator"]
    ranker = row["ranker"]
    chosen_alpha = int(row["chosen_pl_alpha"])

    pl_ee_d_100, pl_nq = mean_ee_d("pl", dict(pl_alpha=chosen_alpha), generator, ranker, lamp_num)
    rmmr_ee_d_100, rmmr_nq = mean_ee_d(
        "pl_randmmr", dict(pl_alpha=4, lambda_low=0.8, lambda_high=1.0), generator, ranker, lamp_num,
    )

    out_row = {
        "lamp_num": lamp_num, "generator": generator, "ranker": ranker,
        "chosen_pl_alpha": chosen_alpha,
        "pl_ee_d_paper_n10": row["pl_ee_d"], "pl_ee_d_n100": pl_ee_d_100,
        "randmmr_ee_d_paper_n10": row["randmmr_ee_d"], "randmmr_ee_d_n100": rmmr_ee_d_100,
        "randmmr_better_paper_n10": bool(row["randmmr_ee_d"] <= row["pl_ee_d"]),
        "randmmr_better_n100": bool(rmmr_ee_d_100 <= pl_ee_d_100),
        "n_queries": pl_nq,
    }
    rows.append(out_row)
    print(f"[{i+1}/28] lamp{lamp_num} {generator}/{ranker} a={chosen_alpha}: "
          f"PL N10={row['pl_ee_d']:.4f} N100={pl_ee_d_100:.4f} | "
          f"RandMMR N10={row['randmmr_ee_d']:.4f} N100={rmmr_ee_d_100:.4f}")

out_df = pd.DataFrame(rows)
out_csv_path = "/private/tmp/claude-501/-Users-asimk-Code-Fair-RAG-1/2ca6f218-27f0-49c7-8fcd-155c4cdafcd3/scratchpad/paper_eed_n100_recompute.csv"
out_df.to_csv(out_csv_path, index=False)

n_flipped = (out_df["randmmr_better_paper_n10"] != out_df["randmmr_better_n100"]).sum()
n_rmmr_better_n10 = out_df["randmmr_better_paper_n10"].sum()
n_rmmr_better_n100 = out_df["randmmr_better_n100"].sum()

print(f"\n=== Summary ===")
print(f"RandMMR better (lower EE-D) at N=10 (paper):  {n_rmmr_better_n10}/28")
print(f"RandMMR better (lower EE-D) at N=100:         {n_rmmr_better_n100}/28")
print(f"Cells where win/lose flips between N=10 and N=100: {n_flipped}/28")
print(f"Mean PL EE-D:      paper(N10)={out_df['pl_ee_d_paper_n10'].mean():.4f}  N100={out_df['pl_ee_d_n100'].mean():.4f}")
print(f"Mean RandMMR EE-D: paper(N10)={out_df['randmmr_ee_d_paper_n10'].mean():.4f}  N100={out_df['randmmr_ee_d_n100'].mean():.4f}")
print(f"Mean (RandMMR - PL) EE-D: paper(N10)={(out_df['randmmr_ee_d_paper_n10']-out_df['pl_ee_d_paper_n10']).mean():.4f}  N100={(out_df['randmmr_ee_d_n100']-out_df['pl_ee_d_n100']).mean():.4f}")
print(f"\nFull results saved to {out_csv_path}")
