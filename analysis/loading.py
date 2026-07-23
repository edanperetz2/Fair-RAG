"""
Completed-run discovery and measurement-input loaders, shared across all analysis
notebooks. Builds on top of framework.cross_run_analysis (the existing "load
completed runs into rows" layer) rather than duplicating it.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Set

from framework.cross_run_analysis import build_macro_comparison_rows, list_run_dirs

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _load_manifest(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _is_completed_manifest(manifest: dict) -> bool:
    status = str(manifest.get("status") or "").lower()
    expected = manifest.get("expected_queries")
    done = manifest.get("n_queries_completed")
    return status in {"completed", "complete", "finished"} or (
        expected is not None and done is not None and done >= expected
    )


def find_completed_run_dirs(
    *,
    runs_root: Optional[str] = None,
    dataset_type: Optional[str] = "lamp",
    lamp_split_type: Optional[str] = "user",
    generator_name: Optional[str] = None,
    top_k: Optional[int] = None,
    rankers: Optional[Iterable[str]] = None,
    lamp_num: Optional[int] = None,
    rerank_method: Optional[str] = None,
) -> List[str]:
    """
    Return sorted run directories whose manifest reports completion and matches every
    given (non-None) filter. Manifest-only (doesn't load query/macro artifacts) - cheap
    enough to call before the heavier build_macro_comparison_rows/build_query_metric_rows.
    """
    rankers_set: Optional[Set[str]] = set(rankers) if rankers is not None else None
    matched = []
    for run_dir in list_run_dirs(runs_root):
        manifest = _load_manifest(run_dir)
        if not _is_completed_manifest(manifest):
            continue

        config = manifest.get("config", {})
        dataset = config.get("dataset", {})
        generation = config.get("generation", {})
        retrieval = config.get("retrieval", {})
        rerank = config.get("rerank", {})

        if dataset_type is not None and dataset.get("dataset_type") != dataset_type:
            continue
        if lamp_split_type is not None and dataset.get("lamp_split_type") != lamp_split_type:
            continue
        if lamp_num is not None and dataset.get("lamp_num") != lamp_num:
            continue
        if generator_name is not None and generation.get("generator_name") != generator_name:
            continue
        if top_k is not None and retrieval.get("top_k") != top_k:
            continue
        if rankers_set is not None and retrieval.get("ranker") not in rankers_set:
            continue
        if rerank_method is not None and rerank.get("method") != rerank_method:
            continue

        matched.append(run_dir)

    return sorted(matched)


def existing_setting_ids(run_dirs: Optional[Iterable[str]] = None) -> Set[str]:
    """Return the set of setting_ids already present among completed runs."""
    if run_dirs is None:
        run_dirs = list_run_dirs()
    rows = build_macro_comparison_rows(run_dirs)
    return {r.get("setting_id") for r in rows if r.get("setting_id")}


def load_relevance_mapping(lamp_num: int, generator_name: str):
    """Load the qid/pid/relevance_label TSV for one LaMP task's utility labels."""
    import pandas as pd

    path = os.path.join(ROOT, "data", f"lamp_utility_labels_{generator_name}", f"{lamp_num}_relevance_mapping.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", dtype={"qid": str, "pid": str, "relevance_label": int})


def load_retrieval_scores(ranker: str, lamp_num: int, generator_name: str) -> Dict[str, Dict[str, float]]:
    """Load precomputed retrieval scores, as {qid: {pid: score}}."""
    path = os.path.join(ROOT, "retrieval", "retrieval_results", generator_name, ranker, f"{lamp_num}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    score_map: Dict[str, Dict[str, float]] = {}
    for qid, items in data.items():
        q_scores: Dict[str, float] = {}
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pid, score = item[0], item[1]
                q_scores[str(pid)] = float(score)
        score_map[str(qid)] = q_scores
    return score_map
