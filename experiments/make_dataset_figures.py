"""
Per-dataset (per-LaMP-task) MMR-vs-PL utility comparison figures, fully disaggregated
by (generator, ranker) cell - one figure per cell per metric (4 cells x 3 metrics = 12
figures), each with one panel per LaMP task.

Complements report/figures/fig7_eu_by_diversity_level.png, which pools all 7 LaMP
tasks together within each (generator, ranker) cell. Here the facet is inverted: one
panel per LaMP task. Cells are kept fully separate (not pooled, unlike an earlier
version of this script) because pl_samples isn't uniform across cells - flanT5Small/
BM25 was upgraded to pl_samples=30 on LaMP-1-4 in a later sweep while every other cell
stayed at pl_samples=10 (see docs/PROJECT_STATUS.md section 8) - pooling cells would
have silently let that higher-precision cell dominate CI widths within a shared bin.

Two families of x-axis, both at per-LIST granularity (per_list_metrics.jsonl joined
with retrieval_lists.jsonl via framework.build_list_metric_rows - finer than the
per-query rows the rest of report/figures/ uses):

  1. Diversity (ILD-Jaccard of that one list) - same metric as fig7, just re-faceted.
     Continuous (tens of thousands of distinct values per cell) - binned into deciles.
  2. "How much this list reordered the base ranking" - Kendall tau and RBO between the
     list's item order and the base retriever's full candidate ordering, computed
     directly from retrieval_lists.jsonl's det_indices (see analysis/rank_similarity.py).
     RBO (96 distinct values) is binned into deciles like ILD. Kendall tau is NOT
     binned: with top_k=5, only 11 tau values are mathematically achievable (5! = 120
     permutations collapse to 11 correlation values), so it's plotted at its raw
     values directly - quantile-binning it would just re-bucket the same 11 points,
     not add resolution.

Outputs (both .pdf and .png) into report/figures/, one file per (metric, cell):
  per_task_eu_by_ild__{generator}_{ranker}.{png,pdf}
  per_task_eu_by_kendalltau__{generator}_{ranker}.{png,pdf}
  per_task_eu_by_rbo__{generator}_{ranker}.{png,pdf}
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_list_metric_rows, list_run_dirs
from analysis import select_best_precision, safe_div, add_rank_similarity, bootstrap_ci

OUT_DIR = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_STYLE = {"mmr": ("#2A9D8F", "o", "MMR (diversity only)"),
                "pl": ("#E76F51", "^", "PL (stochastic fair)")}
GEN_LABEL = {"flanT5Small": "flan-T5-small (77M)", "flanT5Base": "flan-T5-base (250M)"}
RANKER_LABEL = {"bm25": "BM25", "contriever": "Contriever"}
CELLS = [("flanT5Small", "bm25"), ("flanT5Small", "contriever"),
         ("flanT5Base", "bm25"), ("flanT5Base", "contriever")]
EU_UPPER_BOUND = 4.0  # LaMP-3's MAE metric; flipped to higher-is-better before normalizing.
MIN_ROWS, ROUNDS, SEED, N_BINS = 5, 2000, 42, 10


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {name}")


def normalize_list_rows(df, group_cols=("lamp_num", "generator_name", "qid")):
    """Per-query max-normalization at list granularity, mirroring
    analysis.normalization.normalize_query_rows (same LaMP-3 MAE flip, same pooling
    across ranker/rerank_method within a query) but applied directly to each list's own
    eu_score/ild_jaccard rather than a run's query-level aggregate. Normalization still
    pools across ranker/generator (the denominator needs the best EU seen for that
    query anywhere) even though the figures below plot cells separately - normalizing
    per-cell would make the same query's EU incomparable across the 4 cells' figures."""
    parts = []
    for key, g in df.groupby(list(group_cols), dropna=False):
        lamp_num = g["lamp_num"].iloc[0]
        is_lower_better = pd.notna(lamp_num) and int(lamp_num) == 3
        part = g.copy()
        eu_vals = (EU_UPPER_BOUND - part["eu_score"]) if is_lower_better else part["eu_score"]
        eu_denom = eu_vals.dropna().max() if eu_vals.notna().any() else None
        part["eu_score_norm"] = eu_vals.apply(lambda x: safe_div(x, eu_denom))
        ild_denom = part["ild_jaccard"].dropna().max() if "ild_jaccard" in part.columns and part["ild_jaccard"].notna().any() else None
        part["ild_jaccard_norm"] = part["ild_jaccard"].apply(lambda x: safe_div(x, ild_denom)) if "ild_jaccard" in part.columns else None
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _det_reference(task_df, ax):
    det = task_df[task_df["rerank_method"] == "deterministic"]
    if not det.empty:
        det_eu = det["eu_score_norm"].mean()
        ax.axhline(det_eu, color="black", linewidth=1.0, linestyle="--",
                   label="deterministic (original list)")


def plot_binned_metric(cell_df, metric_col, x_label, out_name, title, n_bins=N_BINS):
    """One panel per LaMP task, quantile-binned by `metric_col` (deciles by default),
    MMR vs PL mean normalized EU per bin, within one (generator, ranker) cell."""
    tasks = sorted(cell_df["lamp_num"].dropna().unique())
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 5.6), sharey=True)
    axes_flat = axes.flat

    for ax, task in zip(axes_flat, tasks):
        task_df = cell_df[cell_df["lamp_num"] == task].dropna(subset=[metric_col, "eu_score_norm"])
        pl_mmr = task_df[task_df["rerank_method"].isin(["pl", "mmr"])].copy()
        if pl_mmr.empty:
            ax.set_title(f"LaMP-{int(task)} (no data)")
            continue
        qs = np.linspace(0, 1, n_bins + 1)
        raw_edges = pl_mmr[metric_col].quantile(qs).tolist()
        raw_edges[-1] += 1e-9
        edges = sorted(set(raw_edges))
        if len(edges) < 3:
            ax.set_title(f"LaMP-{int(task)} (metric too narrow to bin)")
            continue
        pl_mmr["bin"] = pd.cut(pl_mmr[metric_col], bins=edges, labels=False, include_lowest=True)
        n_bins_actual = int(pl_mmr["bin"].max() + 1) if pl_mmr["bin"].notna().any() else 0

        for method, (color, marker, label) in METHOD_STYLE.items():
            xs, ys, lo, hi = [], [], [], []
            for b in range(n_bins_actual):
                sub = pl_mmr[pl_mmr["bin"] == b]
                m_sub = sub[sub["rerank_method"] == method]
                other_sub = sub[sub["rerank_method"] != method]
                if len(m_sub) < MIN_ROWS or len(other_sub) < MIN_ROWS:
                    continue
                ci_lo, point, ci_hi = bootstrap_ci(m_sub["eu_score_norm"], np.mean, rounds=ROUNDS, seed=SEED)
                xs.append(b); ys.append(point); lo.append(point - ci_lo); hi.append(ci_hi - point)
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ms=5,
                             linewidth=1.6, capsize=3, label=label)

        _det_reference(task_df, ax)
        ax.set_xticks(range(n_bins_actual), [f"D{b + 1}" for b in range(n_bins_actual)])
        ax.set_title(f"LaMP-{int(task)}")

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean EU (normalized)")
    handles, labels_ = axes_flat[0].get_legend_handles_labels()
    if not handles:
        for a in axes_flat:
            handles, labels_ = a.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels_, frameon=False, fontsize=8, loc="lower center",
               ncols=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.supxlabel(x_label, fontsize=9, y=-0.02)
    save(fig, out_name)


def plot_raw_kendall_tau(cell_df, out_name, title):
    """One panel per LaMP task, x-axis = the metric's own raw achievable values (no
    quantile binning - see module docstring for why). Requires MIN_ROWS at that exact
    tau value for both methods before it's plotted."""
    tasks = sorted(cell_df["lamp_num"].dropna().unique())
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 5.6), sharey=True, sharex=True)
    axes_flat = axes.flat

    for ax, task in zip(axes_flat, tasks):
        task_df = cell_df[cell_df["lamp_num"] == task].dropna(subset=["kendall_tau", "eu_score_norm"])
        pl_mmr = task_df[task_df["rerank_method"].isin(["pl", "mmr"])]
        if pl_mmr.empty:
            ax.set_title(f"LaMP-{int(task)} (no data)")
            continue
        tau_values = sorted(pl_mmr["kendall_tau"].unique())

        for method, (color, marker, label) in METHOD_STYLE.items():
            xs, ys, lo, hi = [], [], [], []
            for tau in tau_values:
                sub = pl_mmr[np.isclose(pl_mmr["kendall_tau"], tau)]
                m_sub = sub[sub["rerank_method"] == method]
                other_sub = sub[sub["rerank_method"] != method]
                if len(m_sub) < MIN_ROWS or len(other_sub) < MIN_ROWS:
                    continue
                ci_lo, point, ci_hi = bootstrap_ci(m_sub["eu_score_norm"], np.mean, rounds=ROUNDS, seed=SEED)
                xs.append(tau); ys.append(point); lo.append(point - ci_lo); hi.append(ci_hi - point)
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ms=5,
                             linewidth=1.6, capsize=3, label=label)

        _det_reference(task_df, ax)
        ax.set_title(f"LaMP-{int(task)}")

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean EU (normalized)")
    handles, labels_ = axes_flat[0].get_legend_handles_labels()
    if not handles:
        for a in axes_flat:
            handles, labels_ = a.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels_, frameon=False, fontsize=8, loc="lower center",
               ncols=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.supxlabel("Kendall tau to base ranking (raw values; 1.0 = unchanged order)", fontsize=9, y=-0.02)
    save(fig, out_name)


print("Loading run data (per-list granularity)...")
run_dirs = list_run_dirs()
list_df = select_best_precision(pd.DataFrame(build_list_metric_rows(run_dirs)))
list_df = list_df[list_df["status"].isin(["completed", "complete", "finished"])] if "status" in list_df.columns else list_df
print(f"{len(list_df)} per-list rows")

list_df = add_rank_similarity(list_df)
list_df = normalize_list_rows(list_df)

for generator, ranker in CELLS:
    cell_df = list_df[(list_df["generator_name"] == generator) & (list_df["ranker"] == ranker)]
    cell_tag = f"{GEN_LABEL[generator]} / {RANKER_LABEL[ranker]}"
    suffix = f"{generator}_{ranker}"
    print(f"--- {cell_tag} ({len(cell_df)} rows) ---")

    plot_binned_metric(
        cell_df, "ild_jaccard_norm", "Diversity (ILD decile, D10 = most diverse)",
        f"per_task_eu_by_ild__{suffix}",
        f"{cell_tag}: utility by diversity level, per LaMP task\nMMR vs PL vs the original deterministic list",
    )
    plot_binned_metric(
        cell_df, "rbo", "Rank similarity to base ranking (RBO decile, D10 = least reranked)",
        f"per_task_eu_by_rbo__{suffix}",
        f"{cell_tag}: utility by how much a list reordered the base ranking (RBO), per LaMP task\n"
        "MMR vs PL vs the original deterministic list",
    )
    plot_raw_kendall_tau(
        cell_df,
        f"per_task_eu_by_kendalltau__{suffix}",
        f"{cell_tag}: utility by how much a list reordered the base ranking (Kendall tau), per LaMP task\n"
        "MMR vs PL vs the original deterministic list",
    )

print("done")
