"""
Regenerates Table 4 (deterministic-baseline EU, Appendix A) using the source
paper's actual with-gold normalization formula (Appendix C.3 / legacy/normalize_eu.py):

    normalized_eu(qid) = det_eu(qid) / max(gold_ceiling(qid), model_ceiling(qid))

Ceiling computation is delegated to analysis.with_gold_normalization.build_with_gold_ceiling
(the shared implementation used by every other with-gold table/figure in the
paper) rather than duplicated here - see that module's docstring for the exact
gold_ceiling/model_ceiling definitions, including why model_ceiling now draws on
every rerank method run for a cell (PL, MMR, RandMMR, deterministic), not just PL.
"""
import json
import os
import sys
from collections import defaultdict

import pandas as pd

ROOT = "/Users/asimk/Code/Fair-RAG-1"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from analysis import select_best_precision
from analysis.loading import select_full_coverage_runs
from analysis.with_gold_normalization import build_with_gold_ceiling

CELLS = [
    (1, "flanT5Small"), (1, "flanT5Base"),
    (4, "flanT5Small"), (4, "flanT5Base"),
]
RANKERS = ["bm25", "contriever"]

RANKER_SHORT = {"bm25": "BM25", "contriever": "Contr."}
GEN_SHORT = {"flanT5Base": "Base", "flanT5Small": "Small"}

PAPER_VALUES = {
    (1, "bm25", "flanT5Small"): 0.308,
    (1, "contriever", "flanT5Small"): 0.286,
    (1, "bm25", "flanT5Base"): 0.670,
    (1, "contriever", "flanT5Base"): 0.637,
    (4, "bm25", "flanT5Small"): 0.217,
    (4, "contriever", "flanT5Small"): 0.254,
    (4, "bm25", "flanT5Base"): 0.223,
    (4, "contriever", "flanT5Base"): 0.268,
}


def load_qid_eu(run_dir, max_only=False):
    """qid -> list of eu_score across all lists in this run (per_list_metrics.jsonl)."""
    fp = os.path.join(run_dir, "per_list_metrics.jsonl")
    out = defaultdict(list)
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["qid"]].append(row["eu_score"])
    if max_only:
        return {qid: max(vals) for qid, vals in out.items()}
    return out


def main(only_cells=None):
    cells = only_cells if only_cells is not None else CELLS
    run_dirs = list_run_dirs()
    raw_df = pd.DataFrame(build_query_metric_rows(run_dirs))
    raw_df = select_full_coverage_runs(raw_df)
    raw_df = select_best_precision(raw_df)
    # A handful of settings have a stray seed=43 duplicate run alongside the
    # canonical seed=42 one (select_best_precision doesn't dedupe across seeds);
    # seed=42 is the project standard everywhere else, so pin to it explicitly.
    raw_df = raw_df[raw_df["seed"] == 42]

    rows = []
    for lamp_num, gen in cells:
        for ranker in RANKERS:
            # deterministic run: per-qid raw EU
            det_sub = raw_df[
                (raw_df["lamp_num"] == lamp_num) & (raw_df["generator_name"] == gen)
                & (raw_df["ranker"] == ranker) & (raw_df["rerank_method"] == "deterministic")
            ]
            det_run_dirs = det_sub["run_dir"].drop_duplicates().tolist()
            assert len(det_run_dirs) == 1, f"expected 1 det run for lamp{lamp_num}/{gen}/{ranker}, found {len(det_run_dirs)}"
            det_eu = load_qid_eu(det_run_dirs[0], max_only=True)  # single list per qid -> max == the value

            ceiling = build_with_gold_ceiling(raw_df, lamp_num, gen, ranker)

            common_qids = set(det_eu) & set(ceiling)
            missing = set(det_eu) - common_qids
            if missing:
                print(f"  WARNING lamp{lamp_num}/{gen}/{ranker}: {len(missing)} det qids lack ceiling data, excluded")

            normed = []
            for qid in common_qids:
                denom = ceiling[qid]
                normed.append(det_eu[qid] / denom if denom > 0 else 1.0)
            ours = sum(normed) / len(normed)

            paper = PAPER_VALUES[(lamp_num, ranker, gen)]
            delta_pct = (ours - paper) / paper * 100
            rows.append({
                "lamp_num": lamp_num, "ranker": ranker, "generator": gen,
                "n_qids": len(common_qids), "paper": paper, "ours": ours, "delta_pct": delta_pct,
            })
            print(f"lamp{lamp_num} {RANKER_SHORT[ranker]}+{GEN_SHORT[gen]}: "
                  f"n={len(common_qids)} paper={paper:.3f} ours={ours:.3f} delta={delta_pct:+.0f}%")

    out_df = pd.DataFrame(rows)
    out_fp = os.path.join(ROOT, "report", "tables", "paper_repro", "table4_with_gold.csv")
    out_df.to_csv(out_fp, index=False)
    print(f"\nSaved: {out_fp}")

    print("\nLaTeX table body:")
    for lamp_num, gen in cells:
        for ranker in RANKERS:
            r = out_df[(out_df.lamp_num == lamp_num) & (out_df.ranker == ranker) & (out_df.generator == gen)].iloc[0]
            sign = "+" if r.delta_pct >= 0 else "$-$"
            print(f"LaMP-{lamp_num} & {RANKER_SHORT[ranker]}+{GEN_SHORT[gen]:<6} & {r.paper:.3f} & {r.ours:.3f} & {sign}{abs(r.delta_pct):.0f}\\% \\\\")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lamp", type=int, nargs="*", default=None, help="restrict to these lamp_nums")
    args = parser.parse_args()
    restrict = None
    if args.lamp:
        restrict = [(ln, gen) for (ln, gen) in CELLS if ln in args.lamp]
    main(only_cells=restrict)
