"""
Replicates the source paper's Table 2 methodology exactly (Kim & Diaz 2025,
arXiv:2409.11598, Section 5 / Table 2): pool every PL alpha's (qid, alpha)
observations together, bin each by its own EE-D into five fixed disparity
intervals ([0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1.0]), and report each bin's
mean normalized utility minus the single scalar deterministic-baseline
utility (unpaired against the *pooled* baseline, not bin-matched -- the
source paper's baseline is one deterministic run with one EE-D value near
1.0, so it can't itself populate every bin; analysis.binning.pool_delta_by_bin's
bin_baseline=False mode exists for exactly this).

Utility is normalized with the source paper's own with-gold formula (Appendix
C.3 / legacy/normalize_eu.py): normalized_eu(qid) = eu(qid) /
max(gold_ceiling(qid), model_ceiling(qid)), via
analysis.with_gold_normalization.build_with_gold_ceiling - the same shared
ceiling implementation every other with-gold table/figure in the paper uses.

Significance: unpaired Student's t-test per bin (comparison-bin observations
vs. the full pooled baseline set), matching the source paper's own caption
("unpaired Student's t-test, p<0.05").

Only LaMP-1 and LaMP-4 are in scope: those are the two tasks the source
paper's Table 2 covers, and the two tasks this project's Appendix A already
compares against a published baseline.
"""
import os
import sys
from collections import defaultdict

import pandas as pd
from scipy import stats as scipy_stats

ROOT = "/Users/asimk/Code/Fair-RAG-1"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from analysis import select_best_precision
from analysis.loading import select_full_coverage_runs
from analysis.binning import pool_delta_by_bin, bin_label_for_value
from analysis.with_gold_normalization import build_with_gold_ceiling

sys.path.insert(0, os.path.join(ROOT, "experiments"))
from table4_with_gold import load_qid_eu, CELLS, RANKERS  # noqa: E402

PL_ALPHAS = [1.0, 2.0, 4.0, 8.0]
GEN_SHORT = {"flanT5Base": "Base", "flanT5Small": "Small"}
RANKER_SHORT = {"bm25": "BM25", "contriever": "Contr."}
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
BIN_LABELS = ["[.0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1.0]"]

# Source paper's Table 2, transcribed directly (docs/../FairRAG.pdf, Table 2),
# restricted to the 4 (ranker,generator) rows this project also has data for.
PAPER_TABLE2 = {
    (1, "bm25", "flanT5Small"):       [-0.12, -0.13, -0.18, -0.02, -0.15],
    (1, "bm25", "flanT5Base"):        [-0.20, -0.04, -0.08, -0.05, -0.02],
    (1, "contriever", "flanT5Small"): [-0.08, -0.29, -0.06, +0.03, -0.14],
    (1, "contriever", "flanT5Base"):  [-0.16, +0.05, -0.06, +0.03, +0.00],
    (4, "bm25", "flanT5Small"):       [-0.06, +0.00, +0.02, +0.01, +0.00],
    (4, "bm25", "flanT5Base"):        [-0.06, +0.00, +0.03, +0.01, +0.02],
    (4, "contriever", "flanT5Small"): [-0.09, -0.02, +0.00, +0.01, +0.00],
    (4, "contriever", "flanT5Base"):  [-0.10, -0.02, +0.01, +0.00, +0.01],
}
# Which of the paper's 5 cells were flagged significant (*), same order.
PAPER_TABLE2_SIG = {
    (1, "bm25", "flanT5Small"):       [False, False, False, False, False],
    (1, "bm25", "flanT5Base"):        [True,  False, False, False, False],
    (1, "contriever", "flanT5Small"): [False, False, False, False, False],
    (1, "contriever", "flanT5Base"):  [True,  False, False, False, False],
    (4, "bm25", "flanT5Small"):       [True,  False, False, False, False],
    (4, "bm25", "flanT5Base"):        [True,  False, False, False, False],
    (4, "contriever", "flanT5Small"): [True,  True,  False, False, False],
    (4, "contriever", "flanT5Base"):  [True,  False, False, False, False],
}


def load_ee_disparity(run_dir):
    """qid -> ee_disparity (one value per qid, computed jointly over that
    setting's sampled rankings)."""
    fp = os.path.join(run_dir, "ee_metrics.jsonl")
    out = {}
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = __import__("json").loads(line)
            out[row["qid"]] = row["ee_disparity"]
    return out


def load_qid_mean_eu(run_dir):
    fp = os.path.join(run_dir, "per_list_metrics.jsonl")
    out = defaultdict(list)
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = __import__("json").loads(line)
            out[row["qid"]].append(row["eu_score"])
    return {qid: sum(v) / len(v) for qid, v in out.items()}


def main(only_cells=None):
    run_dirs = list_run_dirs()
    raw_df = pd.DataFrame(build_query_metric_rows(run_dirs))
    raw_df = select_full_coverage_runs(raw_df)
    raw_df = select_best_precision(raw_df)
    # A handful of settings have a stray seed=43 duplicate run alongside the
    # canonical seed=42 one (select_best_precision doesn't dedupe across seeds);
    # seed=42 is the project standard everywhere else, so pin to it explicitly.
    raw_df = raw_df[raw_df["seed"] == 42]

    cells = only_cells if only_cells is not None else CELLS

    all_rows = []
    for lamp_num, gen in cells:
        for ranker in RANKERS:
            det_sub = raw_df[
                (raw_df["lamp_num"] == lamp_num) & (raw_df["generator_name"] == gen)
                & (raw_df["ranker"] == ranker) & (raw_df["rerank_method"] == "deterministic")
            ]
            det_run_dirs = det_sub["run_dir"].drop_duplicates().tolist()
            assert len(det_run_dirs) == 1, f"expected 1 det run for lamp{lamp_num}/{gen}/{ranker}"
            det_eu = load_qid_eu(det_run_dirs[0], max_only=True)
            det_eed = load_ee_disparity(det_run_dirs[0])

            per_alpha_mean = {}
            per_alpha_eed = {}
            for alpha in PL_ALPHAS:
                pl_sub = raw_df[
                    (raw_df["lamp_num"] == lamp_num) & (raw_df["generator_name"] == gen)
                    & (raw_df["ranker"] == ranker) & (raw_df["rerank_method"] == "pl")
                    & (raw_df["pl_alpha"] == alpha)
                ]
                pl_run_dirs = pl_sub["run_dir"].drop_duplicates().tolist()
                assert len(pl_run_dirs) == 1, f"expected 1 pl(a={alpha}) run for lamp{lamp_num}/{gen}/{ranker}"
                per_alpha_mean[alpha] = load_qid_mean_eu(pl_run_dirs[0])
                per_alpha_eed[alpha] = load_ee_disparity(pl_run_dirs[0])

            ceiling = build_with_gold_ceiling(raw_df, lamp_num, gen, ranker)

            # deterministic (baseline) rows
            for qid in det_eu:
                if qid not in ceiling:
                    continue
                denom = ceiling[qid]
                norm = det_eu[qid] / denom if denom > 0 else 1.0
                all_rows.append({
                    "lamp_num": lamp_num, "generator": gen, "ranker": ranker,
                    "rerank_method": "deterministic", "qid": qid,
                    "ee_disparity": det_eed.get(qid, 1.0), "normalized_eu": norm,
                })

            # pl rows, one per (qid, alpha)
            for alpha in PL_ALPHAS:
                for qid, mean_u in per_alpha_mean[alpha].items():
                    if qid not in ceiling:
                        continue
                    denom = ceiling[qid]
                    norm = mean_u / denom if denom > 0 else 1.0
                    all_rows.append({
                        "lamp_num": lamp_num, "generator": gen, "ranker": ranker,
                        "rerank_method": "pl", "qid": qid, "pl_alpha": alpha,
                        "ee_disparity": per_alpha_eed[alpha].get(qid, None),
                        "normalized_eu": norm,
                    })

    long_df = pd.DataFrame(all_rows)
    long_df = long_df.dropna(subset=["ee_disparity", "normalized_eu"])

    result = pool_delta_by_bin(
        long_df, metric_col="ee_disparity", bin_edges=BIN_EDGES, bin_labels=BIN_LABELS,
        baseline_method="deterministic", comparison_methods=["pl"],
        method_col="rerank_method", value_col="normalized_eu",
        group_by=["lamp_num", "ranker", "generator"], bin_baseline=False,
    )

    # significance: unpaired t-test, this bin's pl obs vs. the FULL pooled baseline set
    sig_rows = []
    for (lamp_num, ranker, gen), part in long_df.groupby(["lamp_num", "ranker", "generator"]):
        baseline_vals = part.loc[part.rerank_method == "deterministic", "normalized_eu"].dropna()
        for bin_label in BIN_LABELS:
            comp_vals = part.loc[
                (part.rerank_method == "pl") & (part.ee_disparity.apply(
                    lambda v: bin_label_for_value(v, BIN_EDGES, BIN_LABELS)
                ) == bin_label), "normalized_eu",
            ].dropna()
            if len(comp_vals) >= 2 and len(baseline_vals) >= 2:
                _, p = scipy_stats.ttest_ind(comp_vals, baseline_vals, equal_var=False)
            else:
                p = None
            sig_rows.append({"lamp_num": lamp_num, "ranker": ranker, "generator": gen, "bin": bin_label, "p_value": p})
    sig_df = pd.DataFrame(sig_rows)

    result = result.merge(sig_df, on=["lamp_num", "ranker", "generator", "bin"], how="left")
    result["sig"] = result["p_value"].apply(lambda p: p is not None and p < 0.05)

    out_fp = os.path.join(ROOT, "report", "tables", "paper_repro", "table2_eed_interval_ours.csv")
    result.to_csv(out_fp, index=False)
    print(f"Saved: {out_fp}\n")

    GEN_ORDER = {"flanT5Base": 0, "flanT5Small": 1}
    RANK_ORDER = {"bm25": 0, "contriever": 1}
    for lamp_num in sorted(set(c[0] for c in cells)):
        print(f"\n=== LaMP-{lamp_num} ===")
        rows = sorted(
            [(r, g) for (ln, g) in cells if ln == lamp_num for r in RANKERS],
            key=lambda rg: (GEN_ORDER[rg[1]], RANK_ORDER[rg[0]]),
        )
        for ranker, gen in rows:
            sub = result[(result.lamp_num == lamp_num) & (result.ranker == ranker) & (result.generator == gen)]
            sub = sub.set_index("bin").reindex(BIN_LABELS)
            ours_vals = []
            for bl in BIN_LABELS:
                d = sub.loc[bl, "delta"]
                s = sub.loc[bl, "sig"]
                if pd.isna(d):
                    ours_vals.append("--")
                else:
                    mark = "*" if s else ""
                    ours_vals.append(f"{d:+.2f}{mark}")
            paper_vals = PAPER_TABLE2.get((lamp_num, ranker, gen))
            paper_sig = PAPER_TABLE2_SIG.get((lamp_num, ranker, gen))
            paper_str = ["--"] * 5
            if paper_vals:
                paper_str = [f"{v:+.2f}{'*' if s else ''}" for v, s in zip(paper_vals, paper_sig)]
            print(f"{RANKER_SHORT[ranker]}+{GEN_SHORT[gen]:<6} Ours : {'  '.join(ours_vals)}")
            print(f"{'':<16} Paper: {'  '.join(paper_str)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lamp", type=int, nargs="*", default=None, help="restrict to these lamp_nums")
    args = parser.parse_args()
    cells = None
    if args.lamp:
        cells = [(ln, gen) for (ln, gen) in CELLS if ln in args.lamp]
    main(only_cells=cells)
