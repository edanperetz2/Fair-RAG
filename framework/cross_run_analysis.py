"""Cross-run loading helpers for plotting and comparing completed experiment runs."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)


def experiment_runs_dir(base_dir: Optional[str] = None) -> str:
    if base_dir is None:
        base_dir = os.path.join(ROOT, "experiment_runs")
    return base_dir


def list_run_dirs(base_dir: Optional[str] = None) -> List[str]:
    """
    Find every run directory under base_dir, at any depth (a run directory is
    identified by containing manifest.json directly). Runs may live flat
    (experiment_runs/{run_id}) or nested (experiment_runs/{generator}/lamp{N}/
    {ranker}/{run_id}) - both are supported transparently since a run dir is
    always a leaf with no subdirectories, so pruning descent at the first match
    is safe either way.
    """
    base_path = experiment_runs_dir(base_dir)
    if not os.path.isdir(base_path):
        return []
    run_dirs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(base_path):
        if "manifest.json" in filenames:
            run_dirs.append(dirpath)
            dirnames[:] = []
    return sorted(run_dirs)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run_artifacts(run_dir: str) -> Dict[str, Any]:
    manifest = _load_json(os.path.join(run_dir, "manifest.json"))
    macro_path = os.path.join(run_dir, "macro_summary.json")
    summary_path = os.path.join(run_dir, "summary.json")
    query_path = os.path.join(run_dir, "query_summary.jsonl")
    progress_path = os.path.join(run_dir, "progress_reports.jsonl")
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "macro_summary": _load_json(macro_path) if os.path.exists(macro_path) else {},
        "summary": _load_json(summary_path) if os.path.exists(summary_path) else {},
        "query_rows": _load_jsonl(query_path),
        "progress_rows": _load_jsonl(progress_path),
    }


def build_macro_comparison_rows(run_dirs: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        artifacts = load_run_artifacts(run_dir)
        manifest = artifacts["manifest"]
        macro = dict(artifacts["macro_summary"])
        macro.update(
            {
                "run_id": manifest.get("run_id"),
                "setting_id": manifest.get("setting_id"),
                "seed": manifest.get("seed"),
                "status": manifest.get("status"),
                "dataset_type": manifest.get("config", {}).get("dataset", {}).get("dataset_type"),
                "lamp_num": manifest.get("config", {}).get("dataset", {}).get("lamp_num"),
                "lamp_split_type": manifest.get("config", {}).get("dataset", {}).get("lamp_split_type"),
                "generator_name": manifest.get("config", {}).get("generation", {}).get("generator_name"),
                "ranker": manifest.get("config", {}).get("retrieval", {}).get("ranker"),
                "rerank_method": manifest.get("config", {}).get("rerank", {}).get("method"),
                "pl_alpha": manifest.get("config", {}).get("rerank", {}).get("pl_alpha"),
                "pl_samples": manifest.get("config", {}).get("rerank", {}).get("pl_samples"),
                "mmr_lambda": manifest.get("config", {}).get("rerank", {}).get("mmr_lambda"),
                "top_k": manifest.get("config", {}).get("retrieval", {}).get("top_k"),
                "num_queries_config": manifest.get("config", {}).get("dataset", {}).get("num_queries"),
                "run_dir": run_dir,
            }
        )
        rows.append(macro)
    return rows


def build_query_metric_rows(run_dirs: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        artifacts = load_run_artifacts(run_dir)
        manifest = artifacts["manifest"]
        for row in artifacts["query_rows"]:
            out = dict(row)
            out.update(
                {
                    "run_id": manifest.get("run_id"),
                    "setting_id": manifest.get("setting_id"),
                    "seed": manifest.get("seed"),
                    "dataset_type": manifest.get("config", {}).get("dataset", {}).get("dataset_type"),
                    "lamp_num": manifest.get("config", {}).get("dataset", {}).get("lamp_num"),
                    "lamp_split_type": manifest.get("config", {}).get("dataset", {}).get("lamp_split_type"),
                    "generator_name": manifest.get("config", {}).get("generation", {}).get("generator_name"),
                    "ranker": manifest.get("config", {}).get("retrieval", {}).get("ranker"),
                    "rerank_method": manifest.get("config", {}).get("rerank", {}).get("method"),
                    "pl_alpha": manifest.get("config", {}).get("rerank", {}).get("pl_alpha"),
                    "pl_samples": manifest.get("config", {}).get("rerank", {}).get("pl_samples"),
                    "mmr_lambda": manifest.get("config", {}).get("rerank", {}).get("mmr_lambda"),
                    "top_k": manifest.get("config", {}).get("retrieval", {}).get("top_k"),
                    "num_queries_config": manifest.get("config", {}).get("dataset", {}).get("num_queries"),
                    "run_dir": run_dir,
                }
            )
            rows.append(out)
    return rows


def maybe_to_dataframe(rows: List[Dict[str, Any]]):
    try:
        import pandas as pd
    except Exception:
        return rows
    return pd.DataFrame(rows)
