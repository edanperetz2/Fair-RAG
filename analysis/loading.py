"""
Completed-run discovery and measurement-input loaders, shared across all analysis
notebooks. Builds on top of framework.cross_run_analysis (the existing "load
completed runs into rows" layer) rather than duplicating it.
"""

from __future__ import annotations

import json
import os
import re
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


def select_best_precision(df):
    """
    Some (lamp_num, pl_alpha) pairs exist at more than one pl_samples (N) value -
    e.g. this repo's pl_alpha_sweep_n30 experiment re-ran some alphas at N=30 while
    others only ever got N=10. Mixing both precisions for the same condition in
    one analysis would silently double-count that condition and blend two different
    noise levels. Keep only the highest-pl_samples row per experiment condition;
    non-pl_samples rows (deterministic/mmr, which don't have a pl_samples axis)
    pass through untouched.

    The condition key must include every axis the experiment grid varies over.
    Grouping by (lamp_num, pl_alpha) alone - as this function originally did when
    the analysis had a single (generator, ranker) cell - silently deleted the
    other three cells' N=10 PL runs for any (task, alpha) that Small/BM25 had
    re-run at N=30.

    Applies to "pl" (keyed by pl_alpha), "pl_randmmr", and "pl_xquad" (both keyed by
    their own lambda_low/high), since re-running any of these methods at a different
    pl_samples count - e.g. an initial N=10 probe followed by a higher-precision
    N=100 run - hits the exact same double-counting risk as "pl" does.
    """
    import pandas as pd

    out_parts = [df[~df["rerank_method"].isin(["pl", "pl_randmmr", "pl_xquad"])]]
    for method, key_cols in (
        ("pl", ("generator_name", "ranker", "lamp_num", "pl_alpha")),
        ("pl_randmmr", ("generator_name", "ranker", "lamp_num", "pl_alpha",
                         "pl_randmmr_lambda_low", "pl_randmmr_lambda_high")),
        ("pl_xquad", ("generator_name", "ranker", "lamp_num", "pl_alpha",
                       "pl_xquad_lambda_low", "pl_xquad_lambda_high")),
    ):
        sub = df[df["rerank_method"] == method]
        if sub.empty:
            continue
        cols = [c for c in key_cols if c in sub.columns]
        best_samples = sub.groupby(cols)["pl_samples"].transform("max")
        out_parts.append(sub[sub["pl_samples"] == best_samples])
    return pd.concat(out_parts, ignore_index=True)


_COVERAGE_SUFFIX_RE = re.compile(r"__nq(?:all|[0-9]+)(?:__off[0-9]+)?__seed\d+$")
_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{8}_\d{6}_")
_FULL_PASS_RE = re.compile(r"__nqall__seed\d+$")


def select_full_coverage_runs(df):
    """
    flanT5Base was originally rolled out as two separate passes per (ranker, lamp_num,
    rerank_method, param) condition - an nq=100 first pass plus a query_offset=100
    scale-up pass covering the rest - before later being re-run as a single
    full-coverage pass (query_offset=0, covering every query for that task in one
    run, named with a bare "__nqall__seed{N}" suffix, no "__off" and no "__nq100").
    Where a full single-pass run exists for a condition, its query set is a strict
    superset of the old two-pass split's, so keeping both would double-count every
    query in that condition. Keep only the full-pass row(s) for any condition that
    has one; conditions with no full-pass run (e.g. flanT5Small, which never got a
    single-pass re-run) are left untouched.
    """
    import pandas as pd

    def condition_key(run_dir: str) -> str:
        name = _TIMESTAMP_PREFIX_RE.sub("", os.path.basename(run_dir))
        return _COVERAGE_SUFFIX_RE.sub("", name)

    cond_key = df["run_dir"].map(condition_key)
    is_full_pass = df["run_dir"].map(lambda d: bool(_FULL_PASS_RE.search(os.path.basename(d))))
    full_pass_conditions = set(cond_key[is_full_pass])
    keep = is_full_pass | ~cond_key.isin(full_pass_conditions)
    return df[keep].reset_index(drop=True)


def select_consistent_precision(
    df,
    *,
    group_cols: Iterable[str],
    setting_cols: Iterable[str],
    samples_col: str = "pl_samples",
    method_col: str = "rerank_method",
    methods: Iterable[str] = ("pl",),
    min_samples: int = 10,
):
    """
    Within each `group_cols` group (e.g. one lamp_num), pool `methods` rows at a single
    pl_samples (N) precision shared by every `setting_cols` combination present in that
    group (e.g. every (generator_name, ranker, pl_alpha) triple) - the highest N for which
    ALL of them have data, falling back to `min_samples` if no higher N is universal.

    Unlike `select_best_precision` (which maximizes N independently per condition and can
    therefore mix, say, N=30 for one (generator, ranker, alpha) triple with N=10 for
    another within the same pooled comparison), this keeps every condition inside one
    group at the same sample count, so no condition gets more independent draws - and
    hence a tighter empirical distribution - than any other it's being compared against.
    Rows for methods not in `methods` pass through untouched.
    """
    import pandas as pd

    group_cols = list(group_cols)
    setting_cols = list(setting_cols)
    methods_set = set(methods)

    out_parts = [df[~df[method_col].isin(methods_set)]]
    for method in methods_set:
        sub = df[df[method_col] == method]
        if sub.empty:
            continue
        for group_key, g in sub.groupby(group_cols, dropna=False):
            available = g.groupby(setting_cols, dropna=False)[samples_col].apply(
                lambda s: set(int(v) for v in s.dropna())
            )
            if available.empty:
                continue
            candidate_ns = sorted(set().union(*available.to_numpy()), reverse=True)
            chosen = min_samples
            for n in candidate_ns:
                if n >= min_samples and all(n in s for s in available.to_numpy()):
                    chosen = n
                    break
            out_parts.append(g[g[samples_col] == chosen])
    return pd.concat(out_parts, ignore_index=True)


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
