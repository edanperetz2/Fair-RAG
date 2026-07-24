"""Sequential batch execution for multiple Fair-RAG run settings."""

from __future__ import annotations

import copy
import datetime
import json
import os
import sys
from typing import Dict, List, Optional

from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from framework.config import RunConfig, setting_id
from framework.dataset import make_dataset
from framework.runner import ExperimentRunner, _log, units_per_query
from framework.cross_run_analysis import build_macro_comparison_rows
from framework.artifacts import ArtifactStore, RunRegistry


def _estimate_total_units(configs: List[RunConfig]) -> Optional[int]:
    """
    Best-effort total (qid, list_id) work-unit count across every config in a batch,
    for a global progress bar. Returns None (unknown total) rather than a misleading
    partial count if any config's dataset can't be loaded (e.g. missing data files).
    """
    total = 0
    try:
        for cfg in configs:
            dataset = make_dataset(cfg)
            total += dataset.total_queries() * units_per_query(cfg.rerank)
    except Exception:
        return None
    return total


class BatchExperimentRunner:
    """Run multiple experiment configurations sequentially and persist a batch summary."""

    def __init__(
        self,
        configs: List[RunConfig],
        batch_id: Optional[str] = None,
        reuse_policy: str = "smart",
    ) -> None:
        self.configs = configs
        self.batch_id = batch_id or datetime.datetime.now().strftime("batch_%Y%m%d_%H%M%S")
        self.reuse_policy = reuse_policy
        self.registry = RunRegistry()

    def run_all(self) -> Dict[str, object]:
        run_dirs: List[str] = []
        comparisons: List[Dict[str, object]] = []
        decisions: List[Dict[str, object]] = []

        total_units = _estimate_total_units(self.configs)
        if total_units is not None:
            _log(f"[Batch] Estimated total work: {total_units} (qid, list_id) units across {len(self.configs)} settings")
        else:
            _log(f"[Batch] Could not estimate total work upfront (some config's data may be missing); {len(self.configs)} settings queued")

        global_pbar = tqdm(total=total_units, desc="Batch progress", unit="unit", leave=True)

        def _on_unit_complete() -> None:
            global_pbar.update(1)

        for index, cfg in enumerate(self.configs, start=1):
            cfg_to_run, decision = self._resolve_config_action(cfg)
            sid = decision["setting_id"]

            if decision["action"] == "skip_completed":
                _log(
                    f"[Batch] Skipping run {index}/{len(self.configs)}: {sid} "
                    f"(reusing completed {decision['run_id']})"
                )
                already_done = len(ArtifactStore(decision["run_dir"]).get_completed_answer_units())
                global_pbar.update(already_done)
                run_dirs.append(decision["run_dir"])
                decisions.append(decision)
                self._write_batch_summary(
                    self._build_batch_summary(run_dirs=run_dirs, decisions=decisions)
                )
                continue

            verb = "Resuming" if decision["action"] == "resume_existing" else "Starting"
            _log(f"[Batch] {verb} run {index}/{len(self.configs)}: {sid}")
            if decision["action"] == "resume_existing":
                already_done = len(ArtifactStore(decision["run_dir"]).get_completed_answer_units())
                global_pbar.update(already_done)
            try:
                store = ExperimentRunner(cfg_to_run).run(on_unit_complete=_on_unit_complete)
            except Exception as exc:
                # One setting's failure (e.g. a transient OS-level error) must not
                # take down an entire multi-hour batch. The failed setting's manifest
                # is left mid-run (status="running"), so simply rerunning this same
                # batch later will pick it up again via resume_existing - nothing is
                # lost, just deferred.
                _log(f"[Batch] Run {index}/{len(self.configs)} FAILED: {sid} - {type(exc).__name__}: {exc}")
                decision["action"] = "failed"
                decision["error"] = f"{type(exc).__name__}: {exc}"
                decisions.append(decision)
                self._write_batch_summary(
                    self._build_batch_summary(run_dirs=run_dirs, decisions=decisions)
                )
                continue
            run_dirs.append(store.run_dir)
            decision["run_dir"] = store.run_dir
            decisions.append(decision)
            self._write_batch_summary(
                self._build_batch_summary(run_dirs=run_dirs, decisions=decisions)
            )

        global_pbar.close()

        comparisons = build_macro_comparison_rows(run_dirs)
        batch_summary = self._build_batch_summary(run_dirs=run_dirs, decisions=decisions)
        batch_summary["runs"] = comparisons
        summary_fp = self._write_batch_summary(batch_summary)
        _log(f"[Batch] Summary written to {summary_fp}")
        return batch_summary

    def _resolve_config_action(self, cfg: RunConfig) -> tuple[RunConfig, Dict[str, object]]:
        sid = setting_id(cfg)
        decision: Dict[str, object] = {
            "setting_id": sid,
            "action": "start_fresh",
            "run_id": cfg.run_id,
            "run_dir": None,
            "reuse_policy": self.reuse_policy,
        }

        if cfg.run_id is not None:
            decision["action"] = "explicit_run_id"
            return cfg, decision

        if self.reuse_policy == "fresh":
            return cfg, decision

        completed = self.registry.find_latest_completed(cfg, setting_id=sid)
        if completed is not None and self.reuse_policy == "smart":
            manifest = completed["manifest"]
            decision.update(
                {
                    "action": "skip_completed",
                    "run_id": manifest.get("run_id"),
                    "run_dir": completed["run_dir"],
                }
            )
            return cfg, decision

        resumable = self.registry.find_latest_resumable(cfg, setting_id=sid)
        if resumable is not None:
            manifest = resumable["manifest"]
            resumed_cfg = copy.deepcopy(cfg)
            resumed_cfg.resume = True
            resumed_cfg.run_id = manifest.get("run_id")
            decision.update(
                {
                    "action": "resume_existing",
                    "run_id": manifest.get("run_id"),
                    "run_dir": resumable["run_dir"],
                }
            )
            return resumed_cfg, decision

        return cfg, decision

    def _build_batch_summary(
        self,
        run_dirs: List[str],
        decisions: List[Dict[str, object]],
    ) -> Dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "reuse_policy": self.reuse_policy,
            "run_dirs": run_dirs,
            "decisions": decisions,
        }

    def _write_batch_summary(self, batch_summary: Dict[str, object]) -> str:
        batch_dir = os.path.join(ROOT, "experiment_runs", "batches", self.batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        fp = os.path.join(batch_dir, "batch_summary.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(batch_summary, fh, indent=2, ensure_ascii=False)
        return fp
