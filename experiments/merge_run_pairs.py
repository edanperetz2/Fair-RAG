"""
Merges a "baseline" (query_offset=0) run directory with its "scale-up"
(query_offset>0) counterpart - same (generator, ranker, lamp_num, rerank
setting) - into a single directory covering the full query range, with
correctly recomputed aggregate stats.

Why this exists: framework.cross_run_analysis.build_macro_comparison_rows
returns one row per run directory, each row carrying that directory's own
pre-computed macro_summary.json (a plain mean over just that directory's
queries). Two separate directories for one logical setting (100 queries vs.
the remaining N) would give two unequally-weighted rows instead of one
correctly-weighted one - silently wrong for any macro-level analysis unless
merged like this. Per-query analysis (build_query_metric_rows) was already
correct either way since it pools raw rows by simple concatenation.

Usage:
    python experiments/merge_run_pairs.py --scan                  # list candidate pairs
    python experiments/merge_run_pairs.py --apply                 # merge all candidate pairs
    python experiments/merge_run_pairs.py --apply --only flanT5Base   # restrict by generator
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

from framework.cross_run_analysis import list_run_dirs
from framework.artifacts import ArtifactStore, comparable_config_dict
from framework.config import setting_id as compute_setting_id, RunConfig, DatasetConfig, RetrievalConfig, RerankConfig, GenerationConfig, MetricsConfig

JSONL_FILES = [
    "retrieval_lists.jsonl", "ee_metrics.jsonl", "llm_answers.jsonl",
    "per_list_metrics.jsonl", "query_summary.jsonl", "progress_reports.jsonl",
]


def _load_manifest(run_dir):
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _group_key(manifest):
    """Everything that defines a logical setting EXCEPT num_queries/query_offset."""
    cfg = comparable_config_dict(manifest.get("config", {}))
    ds = dict(cfg["dataset"])
    ds.pop("num_queries", None)
    ds.pop("query_offset", None)
    cfg["dataset"] = ds
    return json.dumps(cfg, sort_keys=True)


def discover_pairs(run_dirs):
    """Return list of (baseline_dir, scaleup_dir, manifest_a, manifest_b) tuples to merge."""
    groups = defaultdict(list)
    for run_dir in run_dirs:
        manifest = _load_manifest(run_dir)
        groups[_group_key(manifest)].append((run_dir, manifest))

    pairs = []
    for key, entries in groups.items():
        if len(entries) == 1:
            continue
        if len(entries) > 2:
            dirs = [e[0] for e in entries]
            raise RuntimeError(f"Expected at most 2 runs per setting, found {len(entries)}: {dirs}")
        (dir_a, man_a), (dir_b, man_b) = entries
        off_a = man_a["config"]["dataset"].get("query_offset", 0) or 0
        off_b = man_b["config"]["dataset"].get("query_offset", 0) or 0
        if off_a == off_b:
            raise RuntimeError(f"Two runs with the same query_offset={off_a}, not a baseline+scaleup pair: {dir_a}, {dir_b}")
        baseline, scaleup = (dir_a, dir_b) if off_a == 0 else (dir_b, dir_a)
        man_base, man_scale = (man_a, man_b) if off_a == 0 else (man_b, man_a)
        pairs.append((baseline, scaleup, man_base, man_scale))
    return pairs


def _make_run_config_for_setting_id(manifest):
    cfg = manifest["config"]
    d = dict(cfg["dataset"])
    d["num_queries"] = None
    d["query_offset"] = 0
    return RunConfig(
        dataset=DatasetConfig(**d),
        retrieval=RetrievalConfig(**cfg["retrieval"]),
        rerank=RerankConfig(**cfg["rerank"]),
        generation=GenerationConfig(**cfg["generation"]),
        metrics=MetricsConfig(**cfg.get("metrics", {})),
    )


def nested_target_base_dir(manifest):
    ds = manifest["config"]["dataset"]
    gen = manifest["config"]["generation"]["generator_name"]
    ranker = manifest["config"]["retrieval"]["ranker"]
    return os.path.join(ROOT, "experiment_runs", gen, f"lamp{ds['lamp_num']}", ranker)


def merge_pair(baseline_dir, scaleup_dir, man_base, man_scale, target_base_dir, dry_run=True):
    new_cfg = _make_run_config_for_setting_id(man_base)
    new_setting_id = compute_setting_id(new_cfg)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_run_id = f"{timestamp}_{new_setting_id}"
    target_dir = os.path.join(target_base_dir, new_run_id)

    n_base = man_base.get("n_queries_completed", 0)
    n_scale = man_scale.get("n_queries_completed", 0)
    n_total = n_base + n_scale

    if dry_run:
        return {
            "baseline_dir": baseline_dir, "scaleup_dir": scaleup_dir,
            "target_dir": target_dir, "n_base": n_base, "n_scale": n_scale,
            "n_total": n_total, "setting_id": new_setting_id,
        }

    os.makedirs(target_dir, exist_ok=True)

    for fname in JSONL_FILES:
        fp_base = os.path.join(baseline_dir, fname)
        fp_scale = os.path.join(scaleup_dir, fname)
        fp_out = os.path.join(target_dir, fname)
        with open(fp_out, "w", encoding="utf-8") as out:
            for src in (fp_base, fp_scale):
                if os.path.exists(src):
                    with open(src, encoding="utf-8") as fh:
                        shutil.copyfileobj(fh, out)

    started_at = min(man_base.get("started_at") or "9999", man_scale.get("started_at") or "9999")
    manifest = {
        "run_id": new_run_id,
        "setting_id": new_setting_id,
        "config": new_cfg.to_dict(),
        "status": "completed",
        "seed": man_base.get("seed"),
        "report_every_queries": man_base.get("report_every_queries"),
        "started_at": started_at,
        "resumed_at": None,
        "expected_queries": n_total,
        "n_queries_completed": n_total,
        "last_completed_qid": man_scale.get("last_completed_qid") or man_base.get("last_completed_qid"),
        "last_seen_query_index": n_total,
        "merged_from": {
            "baseline_run_id": man_base.get("run_id"),
            "scaleup_run_id": man_scale.get("run_id"),
        },
    }
    with open(os.path.join(target_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # Verify row counts before trusting this merge.
    qs_path = os.path.join(target_dir, "query_summary.jsonl")
    merged_rows = sum(1 for _ in open(qs_path, encoding="utf-8")) if os.path.exists(qs_path) else 0
    base_rows = sum(1 for _ in open(os.path.join(baseline_dir, "query_summary.jsonl"), encoding="utf-8"))
    scale_rows = sum(1 for _ in open(os.path.join(scaleup_dir, "query_summary.jsonl"), encoding="utf-8"))
    if merged_rows != base_rows + scale_rows:
        raise RuntimeError(
            f"Row count mismatch for {target_dir}: merged={merged_rows}, "
            f"base={base_rows}, scale={scale_rows} (expected merged == base+scale)"
        )
    base_qids = {json.loads(l)["qid"] for l in open(os.path.join(baseline_dir, "query_summary.jsonl"), encoding="utf-8")}
    scale_qids = {json.loads(l)["qid"] for l in open(os.path.join(scaleup_dir, "query_summary.jsonl"), encoding="utf-8")}
    overlap = base_qids & scale_qids
    if overlap:
        raise RuntimeError(f"Duplicate qids between baseline and scaleup for {target_dir}: {overlap}")

    store = ArtifactStore(target_dir)
    store.write_summary()
    store.write_macro_summary()

    return {
        "baseline_dir": baseline_dir, "scaleup_dir": scaleup_dir,
        "target_dir": target_dir, "n_base": n_base, "n_scale": n_scale,
        "n_total": n_total, "merged_rows": merged_rows, "setting_id": new_setting_id,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually perform the merge (default: dry-run scan only)")
    parser.add_argument("--only", default=None, help="restrict to run dirs whose path contains this substring (e.g. a generator name)")
    parser.add_argument("--delete-sources", action="store_true", help="git rm/remove source dirs after a verified merge")
    args = parser.parse_args()

    run_dirs = list_run_dirs()
    if args.only:
        run_dirs = [d for d in run_dirs if args.only in d]

    pairs = discover_pairs(run_dirs)
    print(f"Found {len(pairs)} baseline+scaleup pairs to merge among {len(run_dirs)} scanned run dirs.")

    results = []
    for baseline_dir, scaleup_dir, man_base, man_scale in pairs:
        target_base_dir = nested_target_base_dir(man_base)
        result = merge_pair(baseline_dir, scaleup_dir, man_base, man_scale, target_base_dir, dry_run=not args.apply)
        results.append(result)
        tag = "MERGED" if args.apply else "would merge"
        print(f"[{tag}] {result['setting_id']}: {result['n_base']}+{result['n_scale']}={result['n_total']} -> {result['target_dir']}")

    if args.apply and args.delete_sources:
        for r, (baseline_dir, scaleup_dir, _, _) in zip(results, pairs):
            shutil.rmtree(baseline_dir)
            shutil.rmtree(scaleup_dir)
        print(f"Removed {len(results) * 2} source directories.")

    print(f"\n{'Applied' if args.apply else 'Dry run'}: {len(results)} pairs processed.")
