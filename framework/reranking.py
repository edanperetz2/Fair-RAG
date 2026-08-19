"""
Re-ranking methods.

Plackett-Luce (stochastic)
--------------------------
Generates ``N`` randomised rankings via Gumbel-max trick.
Each sample becomes one ``RetrievalList`` with a unique ``list_id``.

MMR — Maximal Marginal Relevance (deterministic)
-------------------------------------------------
Produces exactly **one** ``RetrievalList`` per query.
Uses Jaccard similarity on document token sets for the diversity term.

Algorithm:
    score(d) = λ · rel(d, q) − (1 − λ) · max_{d' ∈ S} sim(d, d')
where
    rel(d, q) — base retrieval score normalised to [0, 1]
    S         — already selected documents
    λ         — trade-off (default 0.55; higher = more relevance-focused)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from perturbation import plackettluce as pl_mod
from framework.retrieval import normalize_scores_for_pl
from framework.config import list_id_for_pl, list_id_for_mmr, list_id_for_deterministic, list_id_for_pl_mmr, list_id_for_pl_randmmr, list_id_for_pl_xquad
from framework.metrics import profile_to_text, jaccard_similarity


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RetrievalList:
    """
    One concrete ranked list of documents for a single query.

    Attributes
    ----------
    qid          : query identifier
    list_id      : unique list identifier (encodes method + sample index)
    doc_ids      : ordered PIDs in list order (length == top_k for MMR/det;
                   full cutoff length for PL samples)
    det_indices  : positions into the original deterministic ranking
                   (same order as doc_ids); used to rebuild TREC top-files for EE
    method       : "pl" | "mmr" | "det"
    param        : human-readable param string e.g. "alpha=2" or "lambda=0.55"
    sample_idx   : index within the method's sample set (0 for deterministic/MMR)
    """
    qid: str
    list_id: str
    doc_ids: List[str]
    det_indices: List[int]
    method: str
    param: str
    sample_idx: int = 0


# ---------------------------------------------------------------------------
# Plackett-Luce
# ---------------------------------------------------------------------------

def generate_pl_lists(
    retrieval_results_for_qid: List,   # [(pid, score), ...]
    ranker: str,
    pl_alpha: int,
    pl_samples: int,
    top_k: int,
    seed: int,
    qid: str,
) -> List[RetrievalList]:
    """
    Generate ``pl_samples`` stochastic rankings via Plackett-Luce sampling.

    Parameters
    ----------
    retrieval_results_for_qid : list of (pid, score) in deterministic rank order
    ranker                    : e.g. "bm25"; used for score normalisation
    pl_alpha                  : temperature exponent; higher = more deterministic
    pl_samples                : number of samples N
    top_k                     : cutoff (list length)
    seed                      : numpy random seed
    qid                       : query id

    Returns
    -------
    List of ``RetrievalList``, one per sample.
    """
    np.random.seed(seed)

    pids = [p[0] for p in retrieval_results_for_qid]
    raw_scores = np.array([float(p[1]) for p in retrieval_results_for_qid], dtype=np.float64)

    normed = normalize_scores_for_pl(raw_scores, ranker)
    scores = normed ** pl_alpha
    cutoff = min(top_k, len(pids))

    rankings, *_ = pl_mod.gumbel_sample_rankings(
        scores, pl_samples, cutoff=cutoff, doc_prob=False
    )

    result: List[RetrievalList] = []
    for i, ranking in enumerate(rankings):
        det_idx = ranking.tolist()          # indices into deterministic ranking
        doc_ids = [pids[j] for j in det_idx]
        result.append(RetrievalList(
            qid=qid,
            list_id=list_id_for_pl(qid, i),
            doc_ids=doc_ids,
            det_indices=det_idx,
            method="pl",
            param=f"alpha={pl_alpha}",
            sample_idx=i,
        ))
    return result


# ---------------------------------------------------------------------------
# MMR helpers
# ---------------------------------------------------------------------------

def _profile_to_tokens(profile: dict) -> frozenset:
    """Extract a token set from a profile dict (all string fields except id/date)."""
    return frozenset(profile_to_text(profile).lower().split())


# ---------------------------------------------------------------------------
# MMR re-ranker
# ---------------------------------------------------------------------------

def generate_mmr_list(
    retrieval_results_for_qid: List,   # [(pid, score), ...]
    profiles_for_qid: List[Dict],      # full profile dicts, same order as retrieval_results
    top_k: int,
    mmr_lambda: float,
    qid: str,
) -> RetrievalList:
    """
    Deterministic MMR re-ranking.

    Parameters
    ----------
    retrieval_results_for_qid : list of (pid, score) in deterministic rank order
    profiles_for_qid          : profile dicts in the same order as retrieval_results
    top_k                     : desired output list length
    mmr_lambda                : relevance weight (0.55 default)
    qid                       : query id

    Returns
    -------
    Single ``RetrievalList`` with ``method="mmr"``.
    """
    pids = [p[0] for p in retrieval_results_for_qid]
    raw_scores = np.array([float(p[1]) for p in retrieval_results_for_qid], dtype=np.float64)

    # Normalise relevance scores to [0, 1]
    mn, mx = raw_scores.min(), raw_scores.max()
    if mx > mn:
        rel_scores = (raw_scores - mn) / (mx - mn)
    else:
        rel_scores = np.ones_like(raw_scores)

    # Build token sets for each document
    pid_to_tokens: Dict[str, frozenset] = {}
    for pid, prof in zip(pids, profiles_for_qid):
        pid_to_tokens[pid] = _profile_to_tokens(prof)

    # Greedy MMR selection
    selected_indices: List[int] = []
    remaining = list(range(len(pids)))
    n_select = min(top_k, len(pids))

    for _ in range(n_select):
        best_idx: Optional[int] = None
        best_score = -np.inf

        for i in remaining:
            rel = rel_scores[i]
            if selected_indices:
                max_sim = max(
                    jaccard_similarity(pid_to_tokens[pids[i]], pid_to_tokens[pids[j]])
                    for j in selected_indices
                )
            else:
                max_sim = 0.0

            mmr_score = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    doc_ids = [pids[i] for i in selected_indices]
    return RetrievalList(
        qid=qid,
        list_id=list_id_for_mmr(qid),
        doc_ids=doc_ids,
        det_indices=selected_indices,
        method="mmr",
        param=f"lambda={mmr_lambda}",
        sample_idx=0,
    )


# ---------------------------------------------------------------------------
# PL-MMR hybrid (stochastic + diversity)
# ---------------------------------------------------------------------------

def generate_pl_mmr_lists(
    retrieval_results_for_qid: List,   # [(pid, score), ...]
    profiles_for_qid: List[Dict],      # profile dicts, same order as retrieval_results
    ranker: str,
    pl_alpha: int,
    pl_mmr_lambda: float,
    pl_samples: int,
    top_k: int,
    seed: int,
    qid: str,
) -> List[RetrievalList]:
    """
    Hybrid PL-MMR re-ranking.

    Rank 1 is drawn via pure Plackett-Luce (Gumbel-max on log-scores).
    Ranks 2..k are drawn sequentially with a diversity penalty applied in
    log-score space before Gumbel sampling:

        log_adj(d) = log(base_score(d)) + log(1 − (1 − λ) · max_sim(d, S))

    where S is the set of already-selected documents and max_sim is the
    maximum Jaccard similarity between d and any member of S.

    At max_sim=0 (completely different doc) the score is unchanged.
    At max_sim=1 (identical doc) it is penalised by log(λ).

    Parameters
    ----------
    retrieval_results_for_qid : list of (pid, score) in deterministic rank order
    profiles_for_qid          : profile dicts in the same order
    ranker                    : e.g. "splade"; used for score normalisation
    pl_alpha                  : temperature exponent; higher = more deterministic
    pl_mmr_lambda             : diversity trade-off (0 = max diversity, 1 = pure PL)
    pl_samples                : number of stochastic samples
    top_k                     : list length cutoff
    seed                      : numpy random seed
    qid                       : query id

    Returns
    -------
    List of ``RetrievalList``, one per sample, with ``method="pl_mmr"``.
    """
    np.random.seed(seed)

    pids = [p[0] for p in retrieval_results_for_qid]
    raw_scores = np.array([float(p[1]) for p in retrieval_results_for_qid], dtype=np.float64)

    normed = normalize_scores_for_pl(raw_scores, ranker)
    base_scores = normed ** pl_alpha          # shape (n_docs,)
    log_base = np.log(np.maximum(base_scores, 1e-12))

    # Build token sets for diversity
    pid_to_tokens: Dict[str, frozenset] = {}
    for pid, prof in zip(pids, profiles_for_qid):
        pid_to_tokens[pid] = _profile_to_tokens(prof)

    n_docs = len(pids)
    cutoff = min(top_k, n_docs)
    result: List[RetrievalList] = []

    for sample_idx in range(pl_samples):
        selected_indices: List[int] = []
        remaining = list(range(n_docs))

        for rank in range(cutoff):
            # Build adjusted log-scores for remaining docs
            adj_log = np.empty(len(remaining))
            for k_r, i in enumerate(remaining):
                if rank == 0 or not selected_indices:
                    # First rank: pure PL, no diversity adjustment
                    adj_log[k_r] = log_base[i]
                else:
                    max_sim = max(
                        jaccard_similarity(pid_to_tokens[pids[i]], pid_to_tokens[pids[j]])
                        for j in selected_indices
                    )
                    diversity_penalty = np.log(
                        max(1.0 - (1.0 - pl_mmr_lambda) * max_sim, 1e-12)
                    )
                    adj_log[k_r] = log_base[i] + diversity_penalty

            # Gumbel-max trick: sample one document
            gumbel_noise = np.random.gumbel(size=len(remaining))
            perturbed = adj_log + gumbel_noise
            chosen_k = int(np.argmax(perturbed))
            chosen_i = remaining[chosen_k]

            selected_indices.append(chosen_i)
            remaining.pop(chosen_k)

        doc_ids = [pids[i] for i in selected_indices]
        result.append(RetrievalList(
            qid=qid,
            list_id=list_id_for_pl_mmr(qid, sample_idx),
            doc_ids=doc_ids,
            det_indices=selected_indices,
            method="pl_mmr",
            param=f"alpha={pl_alpha},lambda={pl_mmr_lambda}",
            sample_idx=sample_idx,
        ))

    return result


# ---------------------------------------------------------------------------
# PL-then-random-MMR hybrid: rank 1 via pure PL(alpha), ranks 2..k via the same
# sequential diversity-penalized Gumbel-max sampling as generate_pl_mmr_lists, but
# with a FRESH lambda drawn ~ Uniform(lambda_low, lambda_high) independently for
# each of the pl_samples lists, rather than one fixed lambda shared by all samples.
# ---------------------------------------------------------------------------

def generate_pl_randmmr_lists(
    retrieval_results_for_qid: List,   # [(pid, score), ...]
    profiles_for_qid: List[Dict],      # profile dicts, same order as retrieval_results
    ranker: str,
    pl_alpha: int,
    lambda_low: float,
    lambda_high: float,
    pl_samples: int,
    top_k: int,
    seed: int,
    qid: str,
) -> List[RetrievalList]:
    """
    Rank 1 is drawn via pure Plackett-Luce (Gumbel-max on log-scores), exactly as in
    ``generate_pl_mmr_lists``. Ranks 2..k use the same diversity-penalized sequential
    Gumbel-max sampling, but each of the ``pl_samples`` independent lists draws its
    own lambda ~ Uniform(lambda_low, lambda_high) rather than sharing one fixed value
    - so within a query, different samples explore different points along the
    relevance/diversity trade-off instead of repeating the same one.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # keep Gumbel draws below reproducible/consistent with siblings

    pids = [p[0] for p in retrieval_results_for_qid]
    raw_scores = np.array([float(p[1]) for p in retrieval_results_for_qid], dtype=np.float64)

    normed = normalize_scores_for_pl(raw_scores, ranker)
    base_scores = normed ** pl_alpha
    log_base = np.log(np.maximum(base_scores, 1e-12))

    pid_to_tokens: Dict[str, frozenset] = {}
    for pid, prof in zip(pids, profiles_for_qid):
        pid_to_tokens[pid] = _profile_to_tokens(prof)

    n_docs = len(pids)
    cutoff = min(top_k, n_docs)
    result: List[RetrievalList] = []

    for sample_idx in range(pl_samples):
        sample_lambda = float(rng.uniform(lambda_low, lambda_high))
        selected_indices: List[int] = []
        remaining = list(range(n_docs))

        for rank in range(cutoff):
            adj_log = np.empty(len(remaining))
            for k_r, i in enumerate(remaining):
                if rank == 0 or not selected_indices:
                    adj_log[k_r] = log_base[i]
                else:
                    max_sim = max(
                        jaccard_similarity(pid_to_tokens[pids[i]], pid_to_tokens[pids[j]])
                        for j in selected_indices
                    )
                    diversity_penalty = np.log(
                        max(1.0 - (1.0 - sample_lambda) * max_sim, 1e-12)
                    )
                    adj_log[k_r] = log_base[i] + diversity_penalty

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


# ---------------------------------------------------------------------------
# PL-xQuAD hybrid (stochastic) — "RandxQuad"
# ---------------------------------------------------------------------------

def generate_pl_xquad_lists(
    retrieval_results_for_qid: List,   # [(pid, score), ...]
    profiles_for_qid: List[Dict],      # profile dicts, same order as retrieval_results
    ranker: str,
    pl_alpha: int,
    lambda_low: float,
    lambda_high: float,
    pl_samples: int,
    top_k: int,
    seed: int,
    qid: str,
) -> List[RetrievalList]:
    """
    Rank 1 is drawn via pure Plackett-Luce (Gumbel-max on log-scores), exactly as in
    ``generate_pl_randmmr_lists``. Ranks 2..k use xQuAD-style (Santos et al. 2010)
    aspect-novelty sampling instead of MMR's pairwise-similarity penalty, with lambda
    drawn fresh per sample from Uniform(lambda_low, lambda_high).

    xQuAD adaptation: without explicit query-aspect/subtopic data (this is a
    content-based, not search-log-based, setting), each candidate document doubles as
    its own pseudo-aspect, weighted by its own relevance. For candidate i given
    already-selected set S:
        novelty(i | S) = sum_a  rel(a) * sim(i, a) * prod_{s in S} (1 - sim(s, a))
        score(i | S)   = (1 - lambda) * rel(i) + lambda * novelty(i | S)
    where rel is the alpha-shaped relevance distribution (normalized to sum to 1) and
    sim is Jaccard similarity of profile token sets. score(i|S) is then log-transformed
    for the same Gumbel-max sampling step used everywhere else in this module.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # keep Gumbel draws below reproducible/consistent with siblings

    pids = [p[0] for p in retrieval_results_for_qid]
    raw_scores = np.array([float(p[1]) for p in retrieval_results_for_qid], dtype=np.float64)

    normed = normalize_scores_for_pl(raw_scores, ranker)
    base_scores = normed ** pl_alpha
    log_base = np.log(np.maximum(base_scores, 1e-12))
    rel_prob = base_scores / base_scores.sum()

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
        # coverage_complement[a] = prod_{s in S} (1 - sim(s, a)); starts uncovered (1.0)
        coverage_complement = np.ones(n_docs, dtype=np.float64)

        for rank in range(cutoff):
            if rank == 0 or not selected_indices:
                adj_log = log_base[remaining]
            else:
                weighted = rel_prob * coverage_complement
                novelty_all = sim_matrix @ weighted  # novelty(i|S) for every candidate i
                xquad_score = (1.0 - sample_lambda) * rel_prob + sample_lambda * novelty_all
                adj_log = np.log(np.maximum(xquad_score[remaining], 1e-12))

            gumbel_noise = np.random.gumbel(size=len(remaining))
            perturbed = adj_log + gumbel_noise
            chosen_k = int(np.argmax(perturbed))
            chosen_i = remaining[chosen_k]

            selected_indices.append(chosen_i)
            remaining.pop(chosen_k)
            coverage_complement *= (1.0 - sim_matrix[chosen_i])

        doc_ids = [pids[i] for i in selected_indices]
        result.append(RetrievalList(
            qid=qid,
            list_id=list_id_for_pl_xquad(qid, sample_idx),
            doc_ids=doc_ids,
            det_indices=selected_indices,
            method="pl_xquad",
            param=f"alpha={pl_alpha},lambda={sample_lambda:.4f}",
            sample_idx=sample_idx,
        ))

    return result


# ---------------------------------------------------------------------------
# Deterministic (passthrough) — kept for completeness and EE baseline
# ---------------------------------------------------------------------------

def generate_deterministic_list(
    retrieval_results_for_qid: List,
    top_k: int,
    qid: str,
) -> RetrievalList:
    """Return the plain deterministic base ranking (no re-ranking)."""
    pids = [p[0] for p in retrieval_results_for_qid]
    cutoff = min(top_k, len(pids))
    det_idx = list(range(cutoff))
    doc_ids = pids[:cutoff]
    return RetrievalList(
        qid=qid,
        list_id=list_id_for_deterministic(qid),
        doc_ids=doc_ids,
        det_indices=det_idx,
        method="det",
        param="",
        sample_idx=0,
    )
