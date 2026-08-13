"""
Reproduce the original paper's (arXiv:2409.11598, "Towards Fair RAG") Figure 3,
Table 1, Figure 4 (a/b/c), and Table 2, using our own already-completed run data.

Scope, deliberately narrower than the paper's 3 retrievers x 4 generators x 7 tasks
grid: we only ran BM25/Contriever (no SPLADE) and flanT5Small/flanT5Base (no
flanT5XXL/FlanUL2) - see docs/PROJECT_STATUS.md. This script uses exactly what
completed locally; it does not launch any new experiments. It also does NOT attempt
the paper's Figure 5 (expected attribution rate / attributed exposure disparity) -
those require a NLI-entailment metric pipeline (RoBERTa-large-MNLI over every
retrieved-item/generated-answer pair) that doesn't exist in this framework yet; that's
a separate follow-up, not something derivable from data already on disk.

Only "deterministic" and "pl" rows are used (MMR is this project's own addition, not
part of the paper's method, and is excluded here to stay faithful to the paper).

IMPORTANT metric-normalization note (verified against the vendored, paper-authors'
own expected_exposure/expeval.py before writing this): our raw `ee_disparity` and
`ee_relevance` query-level fields are ALREADY normalized exactly per the paper's
Appendix C bounds (metrics.py's Disparity/Relevance upperBound formulas are a direct
match: k for EE-D, m+((k-m)^2/(n-m)) or k^2/m for EE-R) - confirmed empirically:
deterministic runs have ee_disparity == 1.0 for 100% of queries. So this script uses
those raw fields directly for Figure 3/Table 1/Figure 4a, NOT this project's own
analysis/normalization.py::normalize_query_rows (which re-normalizes ee_relevance a
second time by empirical max for this project's own cross-task comparisons - a
different, non-paper convention). EU normalization has no clean theoretical bound
(same as the paper), so expected_utility_norm (per-query empirical-max normalization)
is used for Figure 4b/4c as the closest available proxy to the paper's EU-tilde - with
one caveat: the paper's max includes a stochastic oracle-retriever run to approach the
true upper bound, which we never ran (see CLAUDE.md's data pipeline; gold_retriever.py
exists but was never executed here), so our normalization denominator is a looser
"best we happened to observe" rather than a true empirical ceiling. Table 2 uses expected_utility_norm too (this project's own empirical-max normalization,
same caveat as above) - NOT raw utility. Confirmed against the old ../Fair-RAG repo's
fair_rag_experiment.ipynb (which faithfully includes a gold/oracle reference run in its
own normalization pool, per the paper's actual method) and against magnitude sanity: the
paper's LaMP-1 baseline utility of 0.308 would be below-chance raw accuracy for a binary
task, which only makes sense as "30.8% of the way to the oracle ceiling" (ẼU), not raw.
Switching from raw to expected_utility_norm moved our baseline utilities to within the
same ballpark as the paper's for every (task, model) checked (e.g. LaMP-4 BM25+Small:
0.203 vs paper's 0.217) - this was a real bug in an earlier version of this script, not
a legitimate methodology difference.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from analysis import select_best_precision, normalize_query_rows
from analysis.loading import select_full_coverage_runs

FIG_DIR = os.path.join(ROOT, "report", "figures", "paper_repro")
TABLE_DIR = os.path.join(ROOT, "report", "tables", "paper_repro")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]
CELLS = [(g, r) for g in GENERATORS for r in RANKERS]
GEN_LABEL = {"flanT5Small": "flan-T5-small", "flanT5Base": "flan-T5-base"}
RANKER_LABEL = {"bm25": "BM25", "contriever": "Contriever"}
GEN_COLOR = {"flanT5Small": "#4878CF", "flanT5Base": "#F0932B"}
RANKER_COLOR = {"bm25": "#4878CF", "contriever": "#6ACC65"}
ALPHAS = [1, 2, 4, 8]
DISPARITY_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DISPARITY_BIN_LABELS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)"]


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved figure: {name}")


def save_table(df, name):
    path = os.path.join(TABLE_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"saved table: {name}  ({len(df)} rows)")


print("Loading run data...")
run_dirs = list_run_dirs()
raw_df = pd.DataFrame(build_query_metric_rows(run_dirs))
raw_df = select_full_coverage_runs(raw_df)
raw_df = select_best_precision(raw_df)
raw_df = raw_df[raw_df["rerank_method"].isin(["deterministic", "pl"])].copy()
raw_df = raw_df[raw_df["generator_name"].isin(GENERATORS) & raw_df["ranker"].isin(RANKERS)]
qn = normalize_query_rows(raw_df)  # only used for expected_utility_norm (Fig 4b/4c)
print(f"{len(raw_df)} query rows (deterministic + pl only, our 4 cells)")
TASKS = sorted(raw_df["lamp_num"].dropna().unique())


# =====================================================================================
# Figure 3 (paper): alpha -> EE-D disparity boxplots, per task, one figure per cell.
# =====================================================================================
def fig3_alpha_vs_eed():
    for gen, ranker in CELLS:
        cell = raw_df[(raw_df["generator_name"] == gen) & (raw_df["ranker"] == ranker)
                     & (raw_df["rerank_method"] == "pl")]
        fig, axes = plt.subplots(2, 4, figsize=(13.6, 5.6), sharey=True)
        for ax, task in zip(axes.flat, TASKS):
            task_df = cell[cell["lamp_num"] == task]
            data = [task_df[task_df["pl_alpha"] == a]["ee_disparity"].dropna().to_numpy() for a in ALPHAS]
            ax.boxplot(data, tick_labels=[str(a) for a in ALPHAS], showfliers=True)
            ax.set_title(f"LaMP-{int(task)}")
            ax.set_ylim(-0.05, 1.05)
        for ax in axes[-1]:
            ax.set_xlabel("alpha")
        for ax in axes[:, 0]:
            ax.set_ylabel("EE-D (paper-normalized)")
        fig.suptitle(f"Paper Fig. 3 reproduction - {GEN_LABEL[gen]} + {RANKER_LABEL[ranker]}:\n"
                     "effect of fairness control parameter alpha on ranking disparity", y=1.03)
        fig.tight_layout()
        save_fig(fig, f"fig3_alpha_vs_eed__{gen}_{ranker}")


# =====================================================================================
# Shared helper: bin ALL individual (qid, list) rows by x_col, rounded to the nearest
# 0.2 (six anchors: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0), and take the mean of y_col within
# each group - matches the caption ("interpolating the runs for each disparity
# interval"). Confirmed by zooming into the paper's actual rendered figure at pixel
# level: EVERY panel (a, b, and c - i.e. both when x=EE-D and when x=EE-R) shows
# visible kinks landing exactly on all six 0.2-spaced gridlines, not at 5 scattered
# empirical bin-means. Since panel (b)'s x-axis (EE-R) has no reason to sit exactly at
# 1.0 for any particular condition (unlike EE-D, which is exactly 1.0 for 100% of
# deterministic runs), a "bin center = round-to-nearest-0.2" scheme is the only
# explanation that produces 6 clean grid-aligned points uniformly across all 3 panels.
# =====================================================================================
def _bin_by_interval(df, bin_col, x_col, y_col):
    d = df.dropna(subset=[bin_col, y_col]).copy()
    if d.empty:
        return pd.DataFrame(columns=[x_col, y_col, "n"])
    d["_bin"] = (d[bin_col] / 0.2).round() * 0.2
    d["_bin"] = d["_bin"].clip(0.0, 1.0).round(1)
    out = (
        d.groupby("_bin")
        .agg(**{y_col: (y_col, "mean"), "n": (y_col, "size")})
        .reset_index()
        .rename(columns={"_bin": x_col})
    )
    return out.sort_values(x_col).reset_index(drop=True)


def retriever_curve(task, ranker, x_col="ee_disparity", y_col="ee_relevance"):
    """Pool raw rows across BOTH generators for one retriever (EE-D/EE-R don't depend
    on the generator, so pooling is equivalent in expectation to 'averaging across
    generators'), then bin by disparity interval."""
    sub = qn[(qn["ranker"] == ranker) & (qn["lamp_num"] == task)]
    return _bin_by_interval(sub, bin_col=x_col, x_col=x_col, y_col=y_col)


def generator_curve(task, gen, x_col, y_col):
    """Pool raw rows across both retrievers (BM25+Contriever) for one generator - matches
    paper's 'averaging across retrievers for 4b and 4c' - then bin by disparity interval."""
    sub = qn[(qn["generator_name"] == gen) & (qn["lamp_num"] == task)]
    return _bin_by_interval(sub, bin_col=x_col, x_col=x_col, y_col=y_col)


def slope_and_auc(curve, x_col, y_col):
    x, y = curve[x_col].to_numpy(), curve[y_col].to_numpy()
    if len(x) < 2:
        return np.nan, np.nan
    slope = float(np.polyfit(x, y, 1)[0])
    order = np.argsort(x)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz_fn(y[order], x[order]))
    return slope, auc


# =====================================================================================
# Table 1 (paper): fairness-ranking-quality tradeoff slope + AUC, per retriever, per task
# (paper shows LaMP-1/LaMP-4 only; we have all 7).
# =====================================================================================
def table1_slope_auc():
    rows = []
    for ranker in RANKERS:
        for task in TASKS:
            curve = retriever_curve(task, ranker, "ee_disparity", "ee_relevance")
            slope, auc = slope_and_auc(curve, "ee_disparity", "ee_relevance")
            rows.append({"ranker": RANKER_LABEL[ranker], "lamp_num": int(task),
                         "slope": slope, "AUC": auc, "n_points": len(curve)})
    out = pd.DataFrame(rows)
    save_table(out, "table1_slope_auc")
    print(out.round(4).to_string(index=False))
    return out


# =====================================================================================
# Figure 4 (paper): a) EE-D vs EE-R per retriever; b) EE-R vs EU_norm per generator;
# c) EE-D vs EU_norm per generator. One 3-panel figure per LaMP task.
# =====================================================================================
def fig4_relationships():
    for task in TASKS:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.4))

        for ranker in RANKERS:
            curve = retriever_curve(task, ranker, "ee_disparity", "ee_relevance")
            if curve.empty:
                continue
            ax1.plot(curve["ee_disparity"], curve["ee_relevance"], marker="o", ms=4,
                    color=RANKER_COLOR[ranker], label=RANKER_LABEL[ranker])
        ax1.set_xlabel("EE-D (paper-normalized)")
        ax1.set_ylabel("EE-R (paper-normalized)")
        ax1.set_title("(a) RetDisparity vs. Ranking Quality")
        ax1.legend(frameon=False, fontsize=7.5)

        for gen in GENERATORS:
            curve = generator_curve(task, gen, "ee_relevance", "expected_utility_norm")
            if curve.empty:
                continue
            ax2.plot(curve["ee_relevance"], curve["expected_utility_norm"], marker="o", ms=4,
                    color=GEN_COLOR[gen], label=GEN_LABEL[gen])
        ax2.set_xlabel("EE-R (paper-normalized)")
        ax2.set_ylabel("EU (this project's empirical-max norm.)")
        ax2.set_title("(b) Ranking Quality vs. Utility")
        ax2.legend(frameon=False, fontsize=7.5)

        for gen in GENERATORS:
            curve = generator_curve(task, gen, "ee_disparity", "expected_utility_norm")
            if curve.empty:
                continue
            ax3.plot(curve["ee_disparity"], curve["expected_utility_norm"], marker="o", ms=4,
                    color=GEN_COLOR[gen], label=GEN_LABEL[gen])
        ax3.set_xlabel("EE-D (paper-normalized)")
        ax3.set_ylabel("EU (this project's empirical-max norm.)")
        ax3.set_title("(c) RetDisparity vs. Utility")
        ax3.legend(frameon=False, fontsize=7.5)

        fig.suptitle(f"Paper Fig. 4 reproduction - LaMP-{int(task)}", y=1.05)
        fig.tight_layout()
        save_fig(fig, f"fig4_relationships__lamp{int(task)}")


# =====================================================================================
# Table 2 (paper): per model (generator x ranker), per disparity interval: mean utility
# of PL rows falling in that interval (pooled across all 4 alphas) minus the
# deterministic baseline's mean utility, with an unpaired t-test significance star
# (p<0.05) between the two RAW per-query utility samples. RAW (non-normalized) utility,
# matching the paper's literal table values.
# =====================================================================================
def table2_disparity_intervals():
    all_rows = []
    qn_local = qn.copy()
    qn_local["expected_utility_norm"] = pd.to_numeric(qn_local["expected_utility_norm"], errors="coerce")
    for task in TASKS:
        task_df = qn_local[qn_local["lamp_num"] == task]
        for gen, ranker in CELLS:
            cell = task_df[(task_df["generator_name"] == gen) & (task_df["ranker"] == ranker)]
            det_vals = cell[cell["rerank_method"] == "deterministic"]["expected_utility_norm"].dropna()
            if det_vals.empty:
                continue
            baseline = float(det_vals.mean())
            pl = cell[cell["rerank_method"] == "pl"].dropna(subset=["ee_disparity", "expected_utility_norm"])
            row = {"lamp_num": int(task), "generator_name": gen, "ranker": ranker,
                   "model": f"{RANKER_LABEL[ranker]}+{GEN_LABEL[gen]}", "baseline_utility": baseline}
            for lo, hi, label in zip(DISPARITY_BIN_EDGES[:-1], DISPARITY_BIN_EDGES[1:], DISPARITY_BIN_LABELS):
                in_bin = pl[(pl["ee_disparity"] >= lo) & (pl["ee_disparity"] < hi + (1e-9 if hi == 1.0 else 0))]
                bin_vals = in_bin["expected_utility_norm"]
                if bin_vals.empty:
                    row[label] = None
                    row[label + "_n"] = 0
                    row[label + "_sig"] = ""
                    continue
                delta = float(bin_vals.mean() - baseline)
                if len(bin_vals) >= 2 and len(det_vals) >= 2 and bin_vals.nunique() > 1:
                    _, p = scipy_stats.ttest_ind(bin_vals, det_vals, equal_var=True)
                else:
                    p = np.nan
                row[label] = delta
                row[label + "_n"] = int(len(bin_vals))
                row[label + "_sig"] = "*" if (pd.notna(p) and p < 0.05) else ""
            all_rows.append(row)
    out = pd.DataFrame(all_rows)
    save_table(out, "table2_disparity_intervals")

    # console-friendly formatted view, one block per task (matches paper's table layout)
    for task in TASKS:
        print(f"\n=== Table 2 reproduction - LaMP-{int(task)} ===")
        sub = out[out["lamp_num"] == task]
        disp_rows = []
        for _, r in sub.iterrows():
            entry = {"Model (baseline utility)": f"{r['model']} ({r['baseline_utility']:.3f})"}
            for label in DISPARITY_BIN_LABELS:
                v = r[label]
                entry[label] = "n/a" if pd.isna(v) else f"{v:+.2f}{r[label + '_sig']}"
            disp_rows.append(entry)
        print(pd.DataFrame(disp_rows).to_string(index=False))
    return out


if __name__ == "__main__":
    fig3_alpha_vs_eed()
    t1 = table1_slope_auc()
    fig4_relationships()
    t2 = table2_disparity_intervals()
    print("\ndone")
