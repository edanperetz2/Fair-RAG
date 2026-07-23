"""Rerank-setting label formatting, shared across all analysis notebooks."""

from __future__ import annotations

from typing import Any, Mapping


def format_rerank_label(row: Mapping[str, Any], *, ranker_prefix: bool = False) -> str:
    """
    Format a rerank setting as a short display label, e.g. "MMR λ=0.55", "PL α=4",
    "BM25+PL α=4", "BM25 (det)".

    `row` needs "rerank_method" and, depending on method, "pl_alpha"/"mmr_lambda"; also
    "ranker" if `ranker_prefix=True`.
    """
    method = row.get("rerank_method")
    if method == "deterministic":
        base = "det"
    elif method == "mmr":
        lam = row.get("mmr_lambda")
        base = f"MMR λ={lam}" if lam is not None else "MMR"
    elif method == "pl":
        alpha = row.get("pl_alpha")
        base = f"PL α={alpha}" if alpha is not None else "PL"
    elif method == "pl_mmr":
        alpha = row.get("pl_alpha")
        lam = row.get("mmr_lambda")
        parts = []
        if alpha is not None:
            parts.append(f"α={alpha}")
        if lam is not None:
            parts.append(f"λ={lam}")
        base = ("PL-MMR " + ", ".join(parts)) if parts else "PL-MMR"
    else:
        base = str(method)

    if not ranker_prefix:
        return base

    ranker = row.get("ranker")
    if not ranker:
        return base
    ranker_label = ranker.upper() if ranker == "bm25" else str(ranker).capitalize()
    if method == "deterministic":
        return f"{ranker_label} ({base})"
    return f"{ranker_label}+{base}"
