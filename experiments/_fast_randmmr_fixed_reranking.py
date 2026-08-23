"""
Drop-in-equivalent, faster reimplementation of framework.reranking's
generate_pl_randmmr_fixed_lists, for large-N EE-D-only verification.
Precomputes a pairwise Jaccard similarity matrix once per query instead of
recomputing jaccard_similarity() on every rank of every sample. Preserves the
exact same RNG call order/seeding as the original (one np.random.gumbel() for
rank 1, one rng.uniform() per rank for ranks 2..k), so output is bit-identical
to framework.reranking's version.
"""
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from framework.retrieval import normalize_scores_for_pl
from framework.config import list_id_for_pl_randmmr_fixed
from framework.metrics import jaccard_similarity
from framework.reranking import RetrievalList, _profile_to_tokens


def generate_pl_randmmr_fixed_lists_fast(
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

    mn, mx = raw_scores.min(), raw_scores.max()
    rel_scores = (raw_scores - mn) / (mx - mn) if mx > mn else np.ones_like(raw_scores)

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
        selected_indices: List[int] = []
        remaining = list(range(n_docs))

        gumbel_noise = np.random.gumbel(size=len(remaining))
        perturbed = log_base[remaining] + gumbel_noise
        chosen_k = int(np.argmax(perturbed))
        chosen_i = remaining[chosen_k]
        selected_indices.append(chosen_i)
        remaining.pop(chosen_k)

        rank_lambdas: List[float] = []
        for _ in range(cutoff - 1):
            if not remaining:
                break
            rank_lambda = float(rng.uniform(lambda_low, lambda_high))
            rank_lambdas.append(rank_lambda)

            remaining_arr = np.array(remaining)
            sel_arr = np.array(selected_indices)
            max_sim = sim_matrix[np.ix_(remaining_arr, sel_arr)].max(axis=1)
            mmr_scores = rank_lambda * rel_scores[remaining_arr] - (1.0 - rank_lambda) * max_sim
            best_k = int(np.argmax(mmr_scores))
            best_idx = remaining[best_k]

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        doc_ids = [pids[i] for i in selected_indices]
        result.append(RetrievalList(
            qid=qid,
            list_id=list_id_for_pl_randmmr_fixed(qid, sample_idx),
            doc_ids=doc_ids,
            det_indices=selected_indices,
            method="pl_randmmr_fixed",
            param=f"alpha={pl_alpha},lambdas={'|'.join(f'{l:.4f}' for l in rank_lambdas)}",
            sample_idx=sample_idx,
        ))

    return result
