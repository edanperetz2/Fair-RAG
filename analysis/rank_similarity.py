"""
Rank-similarity metrics for individual reranked lists ("how much did reranking move
this list away from the base retriever's ordering?"), built on
`framework.cross_run_analysis.build_list_metric_rows`'s `det_indices` field.

`det_indices` (from `retrieval_lists.jsonl`) already gives, for each position in a
reranked list, that item's rank (0-indexed) in the base retriever's full candidate
ordering for the query - e.g. `[72, 10, 6, 42, 29]` for a PL sample means the item
placed first was originally ranked 73rd. That sequence alone is enough to score how
reordered a list is relative to the original ranking, without needing to look up doc
IDs from a sibling deterministic run.
"""

from __future__ import annotations

from typing import Optional, Sequence

from scipy import stats as scipy_stats


def kendall_tau_from_det_indices(det_indices: Optional[Sequence[int]]) -> Optional[float]:
    """
    Kendall rank correlation between a list and the base ranking, computed directly from
    `det_indices`. Naively correlating position 0..k-1 against the det_indices sequence
    (an earlier version of this function) is WRONG whenever the list substitutes an item:
    e.g. det_indices=[0,1,3,4,5] (base-rank-2's item dropped, base-rank-5's item pulled
    in) is a monotonically increasing sequence, so the naive version scores it tau=1.0 -
    "unchanged" - even though a real substitution happened. The fix: treat every item in
    the union of {0,...,k-1} (the base ranking's own top-k, the implicit reference) and
    `det_indices` (this list's actual items) as a comparable pair. An item present in the
    reference top-k but absent from this list is assigned rank k (tied "fell out of the
    top-k, no more precise position known") rather than being dropped from the
    comparison - this is what makes a dropped item register as a real demotion instead
    of the comparison silently skipping it (kendalltau's tie handling, i.e. tau-b,
    accounts for the shared placeholder rank of multiple dropped items). Verified: an
    exact match still gives tau=1.0; a single substitution like the example above gives
    tau=0.6, not 1.0; PL grabbing several far-down, mutually-increasing items now scores
    strongly negative instead of falsely appearing "perfect."
    """
    if not det_indices or len(det_indices) < 2:
        return None
    k = len(det_indices)
    pos_in_list = {idx: pos for pos, idx in enumerate(det_indices)}
    universe = sorted(set(range(k)) | set(det_indices))
    reference_rank = universe
    list_rank = [pos_in_list.get(item, k) for item in universe]
    result = scipy_stats.kendalltau(reference_rank, list_rank)
    tau = getattr(result, "correlation", None)
    if tau is None:
        tau = getattr(result, "statistic", None)
    return float(tau) if tau is not None and tau == tau else None  # tau==tau filters NaN


def rbo_from_det_indices(det_indices: Optional[Sequence[int]], p: float = 0.9) -> Optional[float]:
    """
    Truncated, depth-weighted rank-biased overlap between a list and the base ranking,
    computed directly from `det_indices`. At each depth d (1..k), the base ranking's
    top-d is by definition the item set {0, ..., d-1}; overlap_d is the fraction of the
    list's own top-d whose det_index falls under d. RBO = (1-p)/(1-p^k) * sum_d
    p^(d-1) * overlap_d - a plain truncated RBO (no extrapolation past depth k, which
    is fine here since k is the full list length, e.g. top_k=5). RBO=1.0 means untouched
    order; lower means more of the list's early positions hold items pulled from further
    down the original ranking.
    """
    if not det_indices:
        return None
    k = len(det_indices)
    if k == 0:
        return None
    weights = [p ** d for d in range(k)]
    norm = sum(weights)
    total = 0.0
    for d in range(1, k + 1):
        prefix = det_indices[:d]
        overlap_d = sum(1 for idx in prefix if idx < d) / d
        total += weights[d - 1] * overlap_d
    return total / norm


def add_rank_similarity(df, det_indices_col: str = "det_indices", p: float = 0.9):
    """Add `kendall_tau` and `rbo` columns computed from `det_indices_col`."""
    df = df.copy()
    df["kendall_tau"] = df[det_indices_col].apply(kendall_tau_from_det_indices)
    df["rbo"] = df[det_indices_col].apply(lambda x: rbo_from_det_indices(x, p=p))
    return df
