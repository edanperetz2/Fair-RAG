"""
RQ3 redesign: broadens the RandMMR-vs-PL comparison from "LaMP-1 only, single hybrid
config" to every (task, generator, ranker) cell where the RandMMR condition alpha=4,
lambda~Unif(0.8,1.0), N=20 completed - the one RandMMR configuration run across the most
tasks (LaMP-1,2,3,4,5,7 x 4 cells = 24 conditions), so a real cross-task tally is possible.

Per cell, the plain-PL comparison partner is chosen automatically: the alpha in {1,2,4,8}
whose own mean EE-D is closest to RandMMR's, with a bias toward "same or slightly higher
disparity than RandMMR" (a conservative comparison - PL gets no fairness advantage) unless
the single closest alpha is already within 0.02 EE-D, in which case that's used regardless
of direction. Also produces a LaMP-1-only, multi-alpha (RandMMR alpha in {4,8,20}) EE-D
spread for the richer figure.
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

FIG_DIR = os.path.join(ROOT, "report", "figures")
TABLE_DIR = os.path.join(ROOT, "report", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

CELLS = [("flanT5Base", "bm25"), ("flanT5Base", "contriever"), ("flanT5Small", "bm25"), ("flanT5Small", "contriever")]
GEN_LABEL = {"flanT5Small": "flan-T5-small", "flanT5Base": "flan-T5-base"}
RANKER_LABEL = {"bm25": "BM25", "contriever": "Contriever"}
GEN_COLOR = {"flanT5Small": "#4878CF", "flanT5Base": "#F0932B"}
CLOSE_ENOUGH = 0.02
PRIMARY_ALPHA, PRIMARY_LO, PRIMARY_HI = 4.0, 0.8, 1.0  # the cross-task RandMMR condition

print("Loading run data...")
run_dirs = list_run_dirs()
raw_df = pd.DataFrame(build_query_metric_rows(run_dirs))
raw_df = select_full_coverage_runs(raw_df)
raw_df = select_best_precision(raw_df)

hybrid_all = raw_df[raw_df["rerank_method"] == "pl_randmmr"].copy()
plain_pl = raw_df[(raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"].isin([1.0, 2.0, 4.0, 8.0]))].copy()
det = raw_df[raw_df["rerank_method"] == "deterministic"].copy()

hybrid_primary = hybrid_all[
    (hybrid_all["pl_alpha"] == PRIMARY_ALPHA)
    & (hybrid_all["pl_randmmr_lambda_low"] == PRIMARY_LO) & (hybrid_all["pl_randmmr_lambda_high"] == PRIMARY_HI)
].copy()

pooled = pd.concat([det, plain_pl, hybrid_primary], ignore_index=True)
qn = normalize_query_rows(pooled)
qn["expected_utility_norm"] = pd.to_numeric(qn["expected_utility_norm"], errors="coerce")


def choose_alpha(cell_pl_means: pd.Series, rand_ee_d: float):
    """cell_pl_means: {alpha: mean_ee_d}. Returns the chosen alpha."""
    diffs = {a: abs(v - rand_ee_d) for a, v in cell_pl_means.items()}
    closest_alpha = min(diffs, key=diffs.get)
    if diffs[closest_alpha] <= CLOSE_ENOUGH:
        return closest_alpha
    above = {a: v for a, v in cell_pl_means.items() if v >= rand_ee_d}
    if above:
        return min(above, key=above.get)
    return closest_alpha


def welch(a, b):
    if len(a) >= 2 and len(b) >= 2 and a.nunique() > 1:
        _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    return np.nan


rows = []
cells = hybrid_primary[["lamp_num", "generator_name", "ranker"]].drop_duplicates()
for _, (task, gen, ranker) in cells.iterrows():
    rh = qn[(qn["lamp_num"] == task) & (qn["generator_name"] == gen) & (qn["ranker"] == ranker)
            & (qn["rerank_method"] == "pl_randmmr")]
    if rh.empty:
        continue
    rand_ee_d = rh["ee_disparity"].mean()
    pl_cell = qn[(qn["lamp_num"] == task) & (qn["generator_name"] == gen) & (qn["ranker"] == ranker)
                 & (qn["rerank_method"] == "pl")]
    pl_means = pl_cell.groupby("pl_alpha")["ee_disparity"].mean().to_dict()
    if not pl_means:
        continue
    chosen_alpha = choose_alpha(pl_means, rand_ee_d)
    pl_chosen = pl_cell[pl_cell["pl_alpha"] == chosen_alpha]
    det_cell = qn[(qn["lamp_num"] == task) & (qn["generator_name"] == gen) & (qn["ranker"] == ranker)
                  & (qn["rerank_method"] == "deterministic")]
    p = welch(rh["expected_utility_norm"].dropna(), pl_chosen["expected_utility_norm"].dropna())
    rows.append({
        "lamp_num": int(task), "generator": gen, "ranker": ranker,
        "chosen_pl_alpha": int(chosen_alpha),
        "det_ee_d": det_cell["ee_disparity"].mean(),
        "pl_ee_d": pl_means[chosen_alpha], "pl_eu_norm": pl_chosen["expected_utility_norm"].mean(),
        "randmmr_ee_d": rand_ee_d, "randmmr_eu_norm": rh["expected_utility_norm"].mean(),
        "delta_eu_norm": rh["expected_utility_norm"].mean() - pl_chosen["expected_utility_norm"].mean(),
        "p_value": p, "sig": "*" if (pd.notna(p) and p < 0.05) else "",
        "randmmr_better": rh["expected_utility_norm"].mean() > pl_chosen["expected_utility_norm"].mean(),
    })

out = pd.DataFrame(rows).sort_values(["lamp_num", "generator", "ranker"])
out.to_csv(os.path.join(TABLE_DIR, "rq3_randmmr_vs_pl_all_cells.csv"), index=False)
print(f"saved table: rq3_randmmr_vs_pl_all_cells.csv ({len(out)} rows)")
pd.set_option("display.width", 160)
print(out.round(4).to_string(index=False))

n_better = int(out["randmmr_better"].sum())
n_total = len(out)
n_sig_better = int(((out["randmmr_better"]) & (out["sig"] == "*")).sum())
n_sig_worse = int(((~out["randmmr_better"]) & (out["sig"] == "*")).sum())
print(f"\nRandMMR EU_norm > matched-PL EU_norm in {n_better}/{n_total} cells "
      f"({n_sig_better} significantly better, {n_sig_worse} significantly worse)")

# LaMP-1 / LaMP-4 detail for the main table
main = out[out["lamp_num"].isin([1, 4])]
main.to_csv(os.path.join(TABLE_DIR, "rq3_randmmr_vs_pl_lamp1_lamp4.csv"), index=False)
other = out[~out["lamp_num"].isin([1, 4])]
print(f"\nLaMP-1/LaMP-4 rows: {len(main)}; other-task rows: {len(other)}, "
      f"RandMMR better in {int(other['randmmr_better'].sum())}/{len(other)} of those")


# ---------------------------------------------------------------- richer LaMP-1 figure
def lamp1_multi_alpha_figure():
    hyb1 = hybrid_all[
        (hybrid_all["lamp_num"] == 1)
        & (((hybrid_all["pl_alpha"] == 4.0) & (hybrid_all["pl_randmmr_lambda_low"] == 0.8) & (hybrid_all["pl_randmmr_lambda_high"] == 1.0))
           | ((hybrid_all["pl_alpha"] == 8.0) & (hybrid_all["pl_randmmr_lambda_low"] == 0.7) & (hybrid_all["pl_randmmr_lambda_high"] == 0.9))
           | ((hybrid_all["pl_alpha"] == 20.0) & (hybrid_all["pl_randmmr_lambda_low"] == 0.7) & (hybrid_all["pl_randmmr_lambda_high"] == 1.0)))
    ].copy()
    pooled1 = pd.concat([
        det[det["lamp_num"] == 1], plain_pl[plain_pl["lamp_num"] == 1], hyb1,
    ], ignore_index=True)
    qn1 = normalize_query_rows(pooled1)
    qn1["expected_utility_norm"] = pd.to_numeric(qn1["expected_utility_norm"], errors="coerce")

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
    for ax, (gen, ranker) in zip(axes, CELLS):
        cell_pl = qn1[(qn1["generator_name"] == gen) & (qn1["ranker"] == ranker) & (qn1["rerank_method"] == "pl")]
        by_alpha = cell_pl.groupby("pl_alpha")[["ee_disparity", "expected_utility_norm"]].mean().sort_index(ascending=False)
        ax.plot(by_alpha["ee_disparity"], by_alpha["expected_utility_norm"], color="gray", marker="o", ms=4,
                linewidth=1.2, label="PL sweep", zorder=2)
        for a, r in by_alpha.iterrows():
            if int(a) not in (4, 8):  # only label the two alphas relevant to context; 1/2 cluster with RandMMR
                continue
            ax.annotate(f"$\\alpha$={int(a)}", (r["ee_disparity"], r["expected_utility_norm"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=6.5, color="gray")

        cell_det = qn1[(qn1["generator_name"] == gen) & (qn1["ranker"] == ranker) & (qn1["rerank_method"] == "deterministic")]
        ax.scatter(cell_det["ee_disparity"].mean(), cell_det["expected_utility_norm"].mean(),
                   color="black", marker="*", s=100, zorder=5, label="deterministic")

        cell_hy = qn1[(qn1["generator_name"] == gen) & (qn1["ranker"] == ranker) & (qn1["rerank_method"] == "pl_randmmr")]
        hy_by_alpha = cell_hy.groupby("pl_alpha")[["ee_disparity", "expected_utility_norm"]].mean().sort_index()
        ax.plot(hy_by_alpha["ee_disparity"], hy_by_alpha["expected_utility_norm"], color=GEN_COLOR[gen],
                marker="D", ms=6, linewidth=1.6, linestyle="-", label="RandMMR", zorder=6)
        # RandMMR's own alpha values crowd too closely together at this panel size for
        # per-point labels (as low as 0.02 EU_norm apart); only the endpoints are labeled,
        # the caption states the ordering (alpha=4,8,20 -> ascending EE-D) for the rest.
        endpoints = hy_by_alpha.iloc[[0, -1]] if len(hy_by_alpha) > 1 else hy_by_alpha
        for a, r in endpoints.iterrows():
            yoff = 6 if r["ee_disparity"] == hy_by_alpha["ee_disparity"].min() else -11
            ax.annotate(f"RandMMR $\\alpha$={int(a)}", (r["ee_disparity"], r["expected_utility_norm"]),
                        textcoords="offset points", xytext=(5, yoff), fontsize=6.5, color=GEN_COLOR[gen])

        ax.set_xlabel("EE-D")
        ax.set_title(f"{GEN_LABEL[gen]} / {RANKER_LABEL[ranker]}", fontsize=8.5)
    axes[0].set_ylabel("EU (normalized)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.suptitle("RandMMR (PL rank-1 + randomized-MMR tail) across its own $\\alpha$ settings vs.\\ the plain-PL sweep (LaMP-1)", fontsize=9.5, y=1.08)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"fig_randmmr_vs_pl.{ext}"))
    plt.close(fig)
    print("saved figure: fig_randmmr_vs_pl")


lamp1_multi_alpha_figure()
