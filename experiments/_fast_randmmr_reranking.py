"""
Drop-in-equivalent, faster reimplementation of framework.reranking's
generate_pl_randmmr_lists for large-N EE-D-only verification. Precomputes a
pairwise Jaccard similarity matrix once per query (like generate_pl_xquad_lists
already does) instead of recomputing jaccard_similarity() on every rank of
every sample -- that redundant recomputation is what made the naive N=100 sweep
too slow. Preserves the EXACT same RNG call order/seeding as the original
(one rng.uniform() per sample for sample_lambda, one np.random.gumbel() per
rank), so output is statistically identical to framework.reranking's version,
just computed with vectorized numpy instead of a jaccard_similarity() call per
candidate per rank per sample.
"""
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from framework.retrieval import normalize_scores_for_pl
from framework.config import list_id_for_pl_randmmr
from framework.metrics import jaccard_similarity
from framework.reranking import RetrievalList, _profile_to_tokens


def generate_pl_randmmr_lists_fast(
    retrieval_results_for_qid: List,
    profiles_for_qid: List[Dict],
    ranker: str,
    pl_alpha: int,
    lambda_low: float,
    lambda_high: float,
    pl_samples: int,
    top_k: int,
    seed: int,
    qid: str,
) -> List[RetrievalList]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    pids = [p[0] for p in retrieval_results_for_qid]
    raw_scores = np.array([float(p[1]) for p in retrieval_results_for_qid], dtype=np.float64)

    normed = normalize_scores_for_pl(raw_scores, ranker)
    base_scores = normed ** pl_alpha
    log_base = np.log(np.maximum(base_scores, 1e-12))

    pid_to_tokens: Dict[str, frozenset] = {}
    for pid, prof in zip(pids, profiles_for_qid):
        pid_to_tokens[pid] = _profile_to_tokens(prof)

    n_docs = len(pids)
    sim_matrix = np.zeros((n_docs, n_docs), dtype=np.float64)
    for a in range(n_docs):
        for b in range(a + 1, n_docs):
            s = jaccard_similarity(pid_to_tokens[pids[a]], pid_to_tokens[pids[b]])
            sim_matrix[a, b] = s
            sim_matrix[b, a] = s

    cutoff = min(top_k, n_docs)
    result: List[RetrievalList] = []

    for sample_idx in range(pl_samples):
        sample_lambda = float(rng.uniform(lambda_low, lambda_high))
        selected_indices: List[int] = []
        remaining = list(range(n_docs))

        for rank in range(cutoff):
            remaining_arr = np.array(remaining)
            if rank == 0 or not selected_indices:
                adj_log = log_base[remaining_arr]
            else:
                sel_arr = np.array(selected_indices)
                max_sim = sim_matrix[np.ix_(remaining_arr, sel_arr)].max(axis=1)
                diversity_penalty = np.log(
                    np.maximum(1.0 - (1.0 - sample_lambda) * max_sim, 1e-12)
                )
                adj_log = log_base[remaining_arr] + diversity_penalty

            gumbel_noise = np.random.gumbel(size=len(remaining))
            perturbed = adj_log + gumbel_noise
            chosen_k = int(np.argmax(perturbed))
            chosen_i = remaining[chosen_k]

            selected_indices.append(chosen_i)
            remaining.pop(chosen_k)

        doc_ids = [pids[i] for i in selected_indices]
        result.append(RetrievalList(
            qid=qid,
            list_id=list_id_for_pl_randmmr(qid, sample_idx),
            doc_ids=doc_ids,
            det_indices=selected_indices,
            method="pl_randmmr",
            param=f"alpha={pl_alpha},lambda={sample_lambda:.4f}",
            sample_idx=sample_idx,
        ))

    return result
