"""
RQ2 diversity-mediation test (report/paper/fair_rag_diversity_paper.tex, Table
"tab:diversity" + the new ILD value-range table). Reproduces both operationalizations
described in the paper's Methodology Notes from scratch, against the currently completed
experiment_runs/, with one methodological upgrade over how the original headline numbers
were apparently produced (no saved script for those was found in the repo - see the
2026-08-13 reconciliation discussion): PL sampling precision (pl_samples/N) is pooled
consistently within each task rather than independently maximized per condition, so no
(generator, ranker, alpha) triple contributes more independent draws - and hence a
tighter empirical distribution - than any other it is compared against inside the same
pooled group. Concretely: per lamp_num, every (generator, ranker, pl_alpha) triple is
pooled at the highest pl_samples value ALL of them share; if that's only 10, N=10 is used
even where some triples have deeper runs.

Two comparisons, matching the paper's text exactly:
  (A) parameter-selected: the single PL alpha / MMR lambda whose own output is most
      concentrated in the low-diversity region (highest self-membership fraction below the
      group's own bottom-quartile/median ILD_norm cutoff); compare ALL of that setting's
      lists' EU_norm to deterministic.
  (B) outcome-selected: every individual PL/MMR list (any setting), regardless of which
      method/parameter produced it, whose own measured ILD_norm falls below the group's
      bottom-quartile/median cutoff; compare those lists' EU_norm to deterministic.

Both at two cutoffs (bottom 25% / bottom 50%), across three granularities (generator x
ranker, generator-pooled, ranker-pooled), for 7 tasks = 56 cells each -> 224 (cell x
approach x cutoff) rows total. MMR lambda=1.0 excluded throughout (mathematically
degenerate - reproduces the deterministic ranking exactly, see Methodology Notes).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from analysis import select_consistent_precision, normalize_query_rows
from analysis.loading import select_full_coverage_runs

TABLE_DIR = os.path.join(ROOT, "report", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]
PL_ALPHAS = [1.0, 2.0, 4.0, 8.0]
MMR_LAMBDAS = [0.15, 0.3, 0.45, 0.55, 0.7, 0.85]  # excludes the degenerate 1.0

GRANULARITIES = [
    ("gen_x_ranker", ["generator_name", "ranker"]),
    ("generator_pooled", ["generator_name"]),
    ("ranker_pooled", ["ranker"]),
]
CUTOFFS = [(0.25, "bottom25"), (0.5, "bottom50")]


def welch(sample, baseline):
    if len(sample) >= 2 and len(baseline) >= 2 and sample.nunique() > 1:
        _, p = scipy_stats.ttest_ind(sample, baseline, equal_var=False)
        return float(p)
    return np.nan


def sig_stars(p):
    if pd.isna(p):
        return ""
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


print("Loading run data...")
run_dirs = list_run_dirs()
raw_df = pd.DataFrame(build_query_metric_rows(run_dirs))
raw_df = select_full_coverage_runs(raw_df)
raw_df = raw_df[raw_df["generator_name"].isin(GENERATORS) & raw_df["ranker"].isin(RANKERS)]
raw_df = raw_df[
    (raw_df["rerank_method"] == "deterministic")
    | ((raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"].isin(PL_ALPHAS)))
    | ((raw_df["rerank_method"] == "mmr") & (raw_df["mmr_lambda"].isin(MMR_LAMBDAS)))
].copy()
raw_df = select_consistent_precision(
    raw_df, group_cols=["lamp_num"], setting_cols=["generator_name", "ranker", "pl_alpha"],
)

qn = normalize_query_rows(raw_df)
qn["expected_utility_norm"] = pd.to_numeric(qn["expected_utility_norm"], errors="coerce")
qn["avg_ild_jaccard_norm"] = pd.to_numeric(qn["avg_ild_jaccard_norm"], errors="coerce")
qn["_setting"] = np.where(
    qn["rerank_method"] == "pl", "pl_a" + qn["pl_alpha"].astype("Int64").astype(str),
    np.where(qn["rerank_method"] == "mmr", "mmr_l" + qn["mmr_lambda"].astype(str), qn["rerank_method"]),
)
TASKS = sorted(qn["lamp_num"].dropna().unique())
print(f"{len(qn)} query rows across {len(TASKS)} tasks")


def slice_mask(df, keys, slice_vals):
    mask = np.ones(len(df), dtype=bool)
    for k, v in zip(keys, slice_vals):
        mask &= (df[k] == v).to_numpy()
    return mask


def analyze_cell(det_vals, divers_df, quantile):
    """One (task, granularity-slice, cutoff) cell: returns dict with cutoff value plus
    approach-A and approach-B results, or None if there isn't enough data."""
    divers_df = divers_df.dropna(subset=["avg_ild_jaccard_norm", "expected_utility_norm"])
    if det_vals.empty or divers_df.empty:
        return None
    cutoff = float(divers_df["avg_ild_jaccard_norm"].quantile(quantile))
    baseline_mean = float(det_vals.mean())

    # Approach B: outcome-selected.
    below = divers_df[divers_df["avg_ild_jaccard_norm"] <= cutoff]
    b_eu = below["expected_utility_norm"].dropna()
    if b_eu.empty:
        approach_b = None
    else:
        p = welch(b_eu, det_vals)
        approach_b = {
            "cutoff_ild_norm": cutoff, "n": int(len(b_eu)),
            "delta_eu_norm": float(b_eu.mean() - baseline_mean), "p_value": p, "sig": sig_stars(p),
        }

    # Approach A: parameter-selected (highest self-membership fraction below cutoff).
    membership = (
        divers_df.groupby("_setting")["avg_ild_jaccard_norm"]
        .apply(lambda s: float((s <= cutoff).mean()))
    )
    counts = divers_df.groupby("_setting").size()
    membership = membership[counts >= 5]  # ignore settings with too few lists to be meaningful
    if membership.empty:
        approach_a = None
    else:
        best_setting = membership.idxmax()
        setting_rows = divers_df[divers_df["_setting"] == best_setting]
        a_eu = setting_rows["expected_utility_norm"].dropna()
        p = welch(a_eu, det_vals)
        approach_a = {
            "cutoff_ild_norm": cutoff, "selected_setting": best_setting,
            "self_membership_frac": float(membership[best_setting]), "n": int(len(a_eu)),
            "delta_eu_norm": float(a_eu.mean() - baseline_mean), "p_value": p, "sig": sig_stars(p),
        }

    return {"baseline_eu_norm": baseline_mean, "n_baseline": int(len(det_vals)),
            "approach_a": approach_a, "approach_b": approach_b}


def build():
    rows = []
    for task in TASKS:
        task_df = qn[qn["lamp_num"] == task]
        det_all = task_df[task_df["rerank_method"] == "deterministic"]
        divers_all = task_df[task_df["rerank_method"].isin(["pl", "mmr"])]
        for gran_name, keys in GRANULARITIES:
            for slice_vals, det_slice in det_all.groupby(keys, dropna=False):
                if not isinstance(slice_vals, tuple):
                    slice_vals = (slice_vals,)
                divers_slice = divers_all[slice_mask(divers_all, keys, slice_vals)]
                det_vals = det_slice["expected_utility_norm"].dropna()
                for quantile, cutoff_name in CUTOFFS:
                    result = analyze_cell(det_vals, divers_slice, quantile)
                    if result is None:
                        continue
                    for approach_name in ("approach_a", "approach_b"):
                        detail = result[approach_name]
                        if detail is None:
                            continue
                        row = {
                            "lamp_num": int(task), "granularity": gran_name, "cutoff": cutoff_name,
                            "approach": approach_name, "baseline_eu_norm": result["baseline_eu_norm"],
                            "n_baseline": result["n_baseline"],
                        }
                        row.update(dict(zip(keys, slice_vals)))
                        row.update(detail)
                        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, "table2_diversity_mediation_full.csv"), index=False)
    print(f"saved table: table2_diversity_mediation_full.csv ({len(out)} rows)")
    return out


def summarize(out):
    print("\n=== Summary (matches tab:diversity format) ===")
    summary_rows = []
    for quantile, cutoff_name in CUTOFFS:
        for approach_name, approach_label in [("approach_a", "A (parameter)"), ("approach_b", "B (outcome)")]:
            sub = out[(out["cutoff"] == cutoff_name) & (out["approach"] == approach_name)]
            n_total = len(sub)
            sig = sub[sub["sig"] != ""]
            n_pos = int((sig["delta_eu_norm"] > 0).sum())
            n_neg = int((sig["delta_eu_norm"] < 0).sum())
            summary_rows.append({
                "cutoff": cutoff_name, "approach": approach_label, "n_cells": n_total,
                "n_significant": len(sig), "n_pos": n_pos, "n_neg": n_neg,
            })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(TABLE_DIR, "table2_diversity_mediation_summary.csv"), index=False)

    sig_b25 = out[(out["cutoff"] == "bottom25") & (out["approach"] == "approach_b") & (out["sig"] != "")]
    sig_b25 = sig_b25.sort_values(["lamp_num", "delta_eu_norm"], ascending=[True, False])
    cols = ["lamp_num", "granularity", "generator_name", "ranker", "cutoff_ild_norm",
            "baseline_eu_norm", "n", "delta_eu_norm", "p_value", "sig"]
    cols = [c for c in cols if c in sig_b25.columns]
    print("\n=== Approach B, bottom-25% cutoff: significant cells (value-range detail) ===")
    print(sig_b25[cols].round(4).to_string(index=False))
    sig_b25[cols].to_csv(os.path.join(TABLE_DIR, "diversity_value_ranges_significant.csv"), index=False)
    print(f"\nsaved table: diversity_value_ranges_significant.csv ({len(sig_b25)} rows)")
    return summary, sig_b25


QUARTILE_LABELS = ["Q1 (least diverse)", "Q2", "Q3", "Q4 (most diverse)"]


def build_task_pooled_quartiles():
    """A complementary, coarser view for the paper's headline value-range table: one row
    per task, pooling PL+MMR lists across ALL FOUR (generator, ranker) cells at once (not
    one of the 56-cell design's 3 granularities), split into quartiles by that task's own
    ILD_norm distribution. Trades cell-level resolution for one clean, directly-readable
    table showing the actual ILD_norm value defining each quartile and its pooled EU_norm
    delta vs. deterministic, Welch's t-test significance."""
    rows = []
    for task in TASKS:
        task_df = qn[qn["lamp_num"] == task]
        det_vals = task_df[task_df["rerank_method"] == "deterministic"]["expected_utility_norm"].dropna()
        divers_df = task_df[task_df["rerank_method"].isin(["pl", "mmr"])].dropna(
            subset=["avg_ild_jaccard_norm", "expected_utility_norm"]
        )
        if det_vals.empty or divers_df.empty:
            continue
        baseline = float(det_vals.mean())
        edges = divers_df["avg_ild_jaccard_norm"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
        edges = list(np.unique(edges))
        if len(edges) < 2:
            continue
        cut_edges = edges.copy()
        cut_edges[0] -= 1e-9
        cut_edges[-1] += 1e-9
        n_bins = len(cut_edges) - 1
        labels = QUARTILE_LABELS if n_bins == 4 else [f"bin {i + 1}/{n_bins}" for i in range(n_bins)]
        divers_df = divers_df.copy()
        divers_df["_bin"] = pd.cut(divers_df["avg_ild_jaccard_norm"], bins=cut_edges, labels=labels, include_lowest=True)

        row = {"lamp_num": int(task), "baseline_eu_norm": baseline, "n_baseline": int(len(det_vals))}
        for i, label in enumerate(labels):
            lo, hi = edges[i], edges[i + 1]
            bin_vals = divers_df.loc[divers_df["_bin"] == label, "expected_utility_norm"].dropna()
            if bin_vals.empty:
                row[f"{label}_lo"] = None
                row[f"{label}_hi"] = None
                row[f"{label}_delta"] = None
                row[f"{label}_n"] = 0
                row[f"{label}_sig"] = ""
                continue
            p = welch(bin_vals, det_vals)
            row[f"{label}_lo"] = lo
            row[f"{label}_hi"] = hi
            row[f"{label}_delta"] = float(bin_vals.mean() - baseline)
            row[f"{label}_n"] = int(len(bin_vals))
            row[f"{label}_sig"] = sig_stars(p)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, "diversity_value_ranges_task_pooled.csv"), index=False)
    print(f"\nsaved table: diversity_value_ranges_task_pooled.csv ({len(out)} rows)")

    print("\n=== Task-pooled ILD quartiles (all 4 generator x ranker cells combined) ===")
    for _, r in out.iterrows():
        print(f"\nLaMP-{r['lamp_num']} (baseline EU_norm={r['baseline_eu_norm']:.3f}, n={r['n_baseline']})")
        disp = []
        for label in QUARTILE_LABELS:
            if pd.isna(r.get(f"{label}_lo")):
                continue
            disp.append({
                "quartile": label,
                "ILD_norm range": f"[{r[f'{label}_lo']:.3f}, {r[f'{label}_hi']:.3f}]",
                "n": int(r[f"{label}_n"]),
                "delta EU_norm": f"{r[f'{label}_delta']:+.3f}{r[f'{label}_sig']}",
            })
        print(pd.DataFrame(disp).to_string(index=False))

    print("\n% ---- LaTeX table body ----")
    for _, r in out.iterrows():
        cells = [f"LaMP-{int(r['lamp_num'])}", f"{r['baseline_eu_norm']:.3f}"]
        for label in QUARTILE_LABELS:
            if pd.isna(r.get(f"{label}_lo")):
                cells.append("n/a")
                continue
            cells.append(f"[{r[f'{label}_lo']:.2f},{r[f'{label}_hi']:.2f}] {r[f'{label}_delta']:+.3f}{r[f'{label}_sig']}")
        print(" & ".join(cells) + r" \\")
    return out


if __name__ == "__main__":
    out = build()
    summarize(out)
    build_task_pooled_quartiles()
