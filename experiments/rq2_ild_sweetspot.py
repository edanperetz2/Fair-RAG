"""
RQ2 redesign: replaces the old 56-cell significance-count table (tab:diversity) and the
per-task ILD quartile-boundary table (tab:ild-ranges) with two figures that make the same
finding legible at a glance.

Figure A (fig:ild-sweetspot): mean EU_norm by ILD_norm quintile (Q1 = least additionally
diversified but, since LaMP candidate pools are often already fairly diverse, still not a
low absolute ILD - see paper Section on Dataset/Threats), separately for PL and MMR, task-
stratified (equal-weight average across the 7 LaMP tasks so no task's query count dominates),
against the deterministic mean as a reference line. Shows where, if anywhere, diversified
lists beat the untouched baseline.

Figure B (fig:mmr-vs-pl-q1): at that same Q1 bin, per-task delta EU_norm vs. deterministic
for PL and MMR side by side - answers "for the ILD range where diversification helps, does
it help more via MMR (relevance-aware) or PL (blind randomization)?"
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
from analysis import select_consistent_precision, normalize_query_rows
from analysis.loading import select_full_coverage_runs

FIG_DIR = os.path.join(ROOT, "report", "figures")
TABLE_DIR = os.path.join(ROOT, "report", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]
PL_ALPHAS = [1.0, 2.0, 4.0, 8.0]
MMR_LAMBDAS = [0.15, 0.3, 0.45, 0.55, 0.7, 0.85]
N_BINS = 5
METHOD_STYLE = {"mmr": ("#2A9D8F", "o", "MMR"), "pl": ("#E76F51", "^", "PL")}

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
raw_df = select_consistent_precision(raw_df, group_cols=["lamp_num"], setting_cols=["generator_name", "ranker", "pl_alpha"])

qn = normalize_query_rows(raw_df)
qn["expected_utility_norm"] = pd.to_numeric(qn["expected_utility_norm"], errors="coerce")
qn["avg_ild_jaccard_norm"] = pd.to_numeric(qn["avg_ild_jaccard_norm"], errors="coerce")
TASKS = sorted(qn["lamp_num"].dropna().unique())


def welch(a, b):
    if len(a) >= 2 and len(b) >= 2 and a.nunique() > 1:
        _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    return np.nan


def sig_stars(p):
    if pd.isna(p):
        return ""
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---- per-task, per-method binning ----
bin_rows = []
q1_rows = []
for task in TASKS:
    tdf = qn[qn["lamp_num"] == task]
    det_vals = tdf[tdf["rerank_method"] == "deterministic"]["expected_utility_norm"].dropna()
    for method in ["pl", "mmr"]:
        m = tdf[tdf["rerank_method"] == method].dropna(subset=["avg_ild_jaccard_norm", "expected_utility_norm"])
        if m.empty:
            continue
        edges = m["avg_ild_jaccard_norm"].quantile(np.linspace(0, 1, N_BINS + 1)).tolist()
        edges = sorted(set(edges))
        if len(edges) < 2:
            continue
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        labels = [f"Q{i + 1}" for i in range(len(edges) - 1)]
        m = m.copy()
        m["_bin"] = pd.cut(m["avg_ild_jaccard_norm"], bins=edges, labels=labels, include_lowest=True)
        for label in labels:
            sub = m.loc[m["_bin"] == label, "expected_utility_norm"]
            if sub.empty:
                continue
            bin_rows.append({"lamp_num": int(task), "method": method, "bin": label,
                              "mean_eu_norm": sub.mean(), "n": len(sub)})
        # Q1 (least diversified) detail, with significance vs. deterministic
        q1 = m.loc[m["_bin"] == "Q1", "expected_utility_norm"]
        if not q1.empty and not det_vals.empty:
            p = welch(q1, det_vals)
            q1_rows.append({"lamp_num": int(task), "method": method, "n_q1": len(q1),
                             "eu_q1": q1.mean(), "eu_det": det_vals.mean(),
                             "delta": q1.mean() - det_vals.mean(), "p": p, "sig": sig_stars(p)})

bin_df = pd.DataFrame(bin_rows)
q1_df = pd.DataFrame(q1_rows)
q1_df.to_csv(os.path.join(TABLE_DIR, "rq2_q1_by_task_method.csv"), index=False)
print(q1_df.round(4).to_string(index=False))

det_task_means = qn[qn["rerank_method"] == "deterministic"].groupby("lamp_num")["expected_utility_norm"].mean()
det_ref = det_task_means.mean()  # equal-weight across tasks, matching the binning below
print(f"\nDeterministic task-stratified reference EU_norm: {det_ref:.4f}")

agg = bin_df.groupby(["method", "bin"])["mean_eu_norm"].mean().reset_index()
agg.to_csv(os.path.join(TABLE_DIR, "rq2_ild_quintile_curve.csv"), index=False)
print(agg.round(4).to_string(index=False))


# ---- Figure A: EU_norm by ILD quintile, PL vs MMR vs deterministic ----
# A single pooled line per method would hide how much the 7 tasks disagree with each
# other (different baselines, different variances, different query counts) - so each
# panel shows the 7 individual per-task lines (thin) behind the task-stratified mean
# (bold), and each task's own line is normalized relative to ITS OWN deterministic
# mean (delta, not raw EU_norm) so tasks with very different EU_norm scales are
# visually comparable on one axis instead of the bold line being pulled around by
# whichever task happens to have the largest raw EU_norm.
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
for ax, (method, (color, marker, label)) in zip(axes, METHOD_STYLE.items()):
    sub = bin_df[bin_df["method"] == method].copy()
    sub["delta_vs_det"] = sub["lamp_num"].map(
        lambda t: det_task_means.get(t, float("nan"))
    )
    sub["delta_vs_det"] = sub["mean_eu_norm"] - sub["delta_vs_det"]
    for task, g in sub.groupby("lamp_num"):
        g = g.sort_values("bin")
        ax.plot(g["bin"], g["delta_vs_det"], color=color, alpha=0.25, linewidth=1.0)
    mean_line = sub.groupby("bin")["delta_vs_det"].mean()
    bin_order = [f"Q{i+1}" for i in range(N_BINS)]
    mean_line = mean_line.reindex(bin_order)
    ax.plot(bin_order, mean_line.values, color=color, marker=marker, ms=6,
            linewidth=2.4, label=f"{label} (task mean)", zorder=5)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("ILD$_{norm}$ quintile\n(Q1 = least diversified)")
    ax.set_title(label, fontsize=9)
axes[0].set_ylabel("$\\Delta$EU$_{norm}$ vs.\\ own task's\ndeterministic mean")
fig.suptitle("Thin lines = individual tasks; bold = task-stratified mean", fontsize=8, y=1.03)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_ild_sweetspot.{ext}"))
plt.close(fig)
print("saved figure: fig_ild_sweetspot")


# ---- Figure B: Q1 delta vs. deterministic, PL vs MMR, per task ----
fig, ax = plt.subplots(figsize=(6.2, 2.8))
tasks_sorted = sorted(q1_df["lamp_num"].unique())
x = np.arange(len(tasks_sorted))
width = 0.35
for i, method in enumerate(["pl", "mmr"]):
    color, marker, label = METHOD_STYLE[method]
    sub = q1_df[q1_df["method"] == method].set_index("lamp_num").reindex(tasks_sorted)
    bars = ax.bar(x + (i - 0.5) * width, sub["delta"], width, color=color, label=label)
    for xi, (delta, sig) in zip(x + (i - 0.5) * width, zip(sub["delta"], sub["sig"])):
        if pd.notna(delta):
            ax.annotate(sig, (xi, delta), textcoords="offset points",
                        xytext=(0, 3 if delta >= 0 else -10), ha="center", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x, [f"LaMP-{t}" for t in tasks_sorted])
ax.set_ylabel("$\\Delta$EU$_{norm}$ vs.\ndeterministic (Q1)")
ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_mmr_vs_pl_q1.{ext}"))
plt.close(fig)
print("saved figure: fig_mmr_vs_pl_q1")
