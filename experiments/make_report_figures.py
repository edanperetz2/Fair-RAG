"""
Report figures for the final project report (docs/PROJECT_STATUS.md sections 7-11).

Generates the six candidate figures for the LaTeX report from the completed
experiment_runs/ data, reusing the exact same loading/normalization pipeline as
experiments/analyze_mmr_sweep.py so every number in a figure matches the printed
analysis. Outputs both .pdf (for LaTeX inclusion) and .png (for quick review)
into report/figures/.

Figures:
  1. Generator headroom  - deterministic raw EU per task, all four (gen, ranker) cells.
  2. Fairness-utility map - per-setting mean EE-D_norm vs EU_norm, one panel per cell.
  3. Lambda manipulation - lambda -> ILD per task (representative cell) + ILD range per cell.
  4. ILD sign flip       - within-MMR pooled coef(ILD) with 95% CI, per cell.
  5. Matched diversity   - PL minus MMR EU_norm delta by ILD quintile, per cell.
  6. Generator axis      - M2 EE-D coef and M3 interaction coef per generator, 95% CI.
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

from framework import build_query_metric_rows, build_macro_comparison_rows, list_run_dirs, maybe_to_dataframe
from analysis import normalize_query_rows, pool_delta_by_bin, fit_ols, select_best_precision

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

GEN_LABEL = {"flanT5Small": "flan-T5-small (77M)", "flanT5Base": "flan-T5-base (250M)"}
RANKER_LABEL = {"bm25": "BM25", "contriever": "Contriever"}
GEN_COLOR = {"flanT5Small": "#4878CF", "flanT5Base": "#D65F5F"}
CELLS = [
    ("flanT5Small", "bm25"), ("flanT5Small", "contriever"),
    ("flanT5Base", "bm25"), ("flanT5Base", "contriever"),
]


def cell_tag(generator, ranker):
    return f"{GEN_LABEL[generator]} / {RANKER_LABEL[ranker]}"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {name}")


def ci95(model, feature):
    dof = model["n"] - (len(model["coef"]) - 1) - 1
    tcrit = scipy_stats.t.ppf(0.975, dof)
    return tcrit * model["std_err"][feature]


print("Loading run data...")
run_dirs = list_run_dirs()
macro_df = select_best_precision(maybe_to_dataframe(build_macro_comparison_rows(run_dirs)))
query_df = select_best_precision(pd.DataFrame(build_query_metric_rows(run_dirs)))
query_norm_df = normalize_query_rows(query_df)
print(f"{len(run_dirs)} runs -> {len(macro_df)} macro rows, {len(query_norm_df)} query rows")


# ---------------------------------------------------------------- figure 1
def fig1_headroom():
    det = macro_df[macro_df["rerank_method"] == "deterministic"]
    tasks = sorted(det["lamp_num"].unique())
    x = np.arange(len(tasks))
    width = 0.2
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for i, (gen, ranker) in enumerate(CELLS):
        vals = [
            det[(det["generator_name"] == gen) & (det["ranker"] == ranker)
                & (det["lamp_num"] == t)]["expected_utility"].mean()
            for t in tasks
        ]
        ax.bar(x + (i - 1.5) * width, vals, width,
               color=GEN_COLOR[gen], alpha=1.0 if ranker == "bm25" else 0.55,
               label=cell_tag(gen, ranker))
    ax.set_xticks(x, [f"LaMP-{int(t)}" for t in tasks])
    ax.set_ylabel("Deterministic raw EU")
    ax.set_title("Generator headroom: deterministic-ranking utility per task", pad=44)
    ax.legend(ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig1_headroom")


# ---------------------------------------------------------------- figure 2
def fig2_fairness_utility_map():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6), sharex=True, sharey=True)
    for ax, (gen, ranker) in zip(axes.flat, CELLS):
        cell = query_norm_df[
            (query_norm_df["generator_name"] == gen) & (query_norm_df["ranker"] == ranker)
        ].dropna(subset=["ee_disparity_norm", "expected_utility_norm"])
        grouped = cell.groupby(["rerank_method", "mmr_lambda", "pl_alpha"], dropna=False)[
            ["ee_disparity_norm", "expected_utility_norm"]
        ].mean().reset_index()

        det = grouped[grouped["rerank_method"] == "deterministic"]
        mmr = grouped[grouped["rerank_method"] == "mmr"].sort_values("mmr_lambda")
        pl = grouped[grouped["rerank_method"] == "pl"].sort_values("pl_alpha")

        # The tradeoff path: deterministic -> PL alpha=8 -> ... -> alpha=1 (fairest).
        path = pd.concat([det, pl.sort_values("pl_alpha", ascending=False)])
        ax.plot(path["ee_disparity_norm"], path["expected_utility_norm"],
                color="gray", linewidth=1.1, linestyle="-", zorder=1)

        sc1 = ax.scatter(mmr["ee_disparity_norm"], mmr["expected_utility_norm"],
                         c=mmr["mmr_lambda"].astype(float), cmap="Blues", vmin=-0.2, vmax=1.2,
                         s=34, marker="o", edgecolor="k", linewidth=0.4, label="MMR ($\\lambda$)")
        sc2 = ax.scatter(pl["ee_disparity_norm"], pl["expected_utility_norm"],
                         c=np.log2(pl["pl_alpha"].astype(float)), cmap="Oranges", vmin=-1.5, vmax=4,
                         s=40, marker="^", edgecolor="k", linewidth=0.4, label="PL ($\\alpha$)")
        ax.scatter(det["ee_disparity_norm"], det["expected_utility_norm"],
                   color="black", marker="*", s=120, zorder=5, label="deterministic")
        ax.set_title(cell_tag(gen, ranker))
    for ax in axes[-1]:
        ax.set_xlabel("EE-D (normalized, higher = less fair)")
    for ax in axes[:, 0]:
        ax.set_ylabel("EU (normalized)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Fairness vs. utility across all reranking settings (per-setting means)", y=0.995)
    fig.tight_layout()
    save(fig, "fig2_fairness_utility_map")


# ---------------------------------------------------------------- figure 2b
def fig2b_fairness_diversity_coupling():
    """The proposal's premise, as a two-panel 'dials' figure: each panel is one
    reranker's knob, x = intervention strength (left anchor = deterministic).
    PL's knob moves fairness AND diversity together (the confound); MMR's knob
    moves diversity while EE-D stays pinned at 1 (the decoupling instrument).
    Bold line = mean across the four (generator, retriever) cells; thin = cells."""
    EED_COLOR, ILD_COLOR = "#6A3D9A", "#33A02C"

    def setting_means(method, param_col, param_values):
        per_cell, mean_rows = {}, []
        for gen, ranker in CELLS:
            cell = query_norm_df[
                (query_norm_df["generator_name"] == gen) & (query_norm_df["ranker"] == ranker)
            ].dropna(subset=["ee_disparity_norm", "avg_ild_jaccard_norm"])
            det = cell[cell["rerank_method"] == "deterministic"][
                ["ee_disparity_norm", "avg_ild_jaccard_norm"]].mean()
            rows = [det]
            for v in param_values:
                sub = cell[(cell["rerank_method"] == method)
                           & (np.isclose(cell[param_col].astype(float), v))]
                rows.append(sub[["ee_disparity_norm", "avg_ild_jaccard_norm"]].mean())
            per_cell[(gen, ranker)] = pd.DataFrame(rows).reset_index(drop=True)
        mean_df = sum(per_cell.values()) / len(per_cell)
        return per_cell, mean_df

    pl_alphas = [8, 4, 2, 1]
    mmr_lambdas = [0.85, 0.7, 0.55, 0.45, 0.3, 0.15]
    pl_cells, pl_mean = setting_means("pl", "pl_alpha", pl_alphas)
    mmr_cells, mmr_mean = setting_means("mmr", "mmr_lambda", mmr_lambdas)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    panels = [
        (axes[0], pl_cells, pl_mean, ["det."] + [f"$\\alpha$={a}" for a in pl_alphas],
         "PL dial: sampling temperature $\\alpha$\n(right = stronger fairness)"),
        (axes[1], mmr_cells, mmr_mean, ["det."] + [f"{l}" for l in mmr_lambdas],
         "MMR dial: relevance weight $\\lambda$\n(right = stronger diversity)"),
    ]
    ymin = min(pl_mean["avg_ild_jaccard_norm"].min(), mmr_mean["avg_ild_jaccard_norm"].min())
    for ax, cells, mean_df, ticks, title in panels:
        x = np.arange(len(ticks))
        ax_r = ax.twinx()
        for cdf in cells.values():
            ax.plot(x, cdf["ee_disparity_norm"], color=EED_COLOR, alpha=0.25, linewidth=0.8)
            ax_r.plot(x, cdf["avg_ild_jaccard_norm"], color=ILD_COLOR, alpha=0.25, linewidth=0.8)
        ax.plot(x, mean_df["ee_disparity_norm"], color=EED_COLOR, marker="o", ms=4,
                linewidth=2, label="EE-D (unfairness)")
        ax_r.plot(x, mean_df["avg_ild_jaccard_norm"], color=ILD_COLOR, marker="s", ms=4,
                  linewidth=2, label="ILD (diversity)")
        ax.set_ylim(0.0, 1.08)
        ax_r.set_ylim(ymin - 0.01, 1.0)
        ax.set_xticks(x, ticks)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("EE-D (normalized)", color=EED_COLOR)
        ax.tick_params(axis="y", labelcolor=EED_COLOR)
        ax_r.set_ylabel("ILD (normalized)", color=ILD_COLOR)
        ax_r.tick_params(axis="y", labelcolor=ILD_COLOR)
        ax_r.spines["top"].set_visible(False)
    handles = [
        plt.Line2D([], [], color=EED_COLOR, marker="o", ms=4, linewidth=2),
        plt.Line2D([], [], color=ILD_COLOR, marker="s", ms=4, linewidth=2),
    ]
    fig.legend(handles, ["EE-D (unfairness)", "ILD (diversity)"], frameon=False,
               fontsize=8, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Fairness interventions inherently raise diversity: PL moves both dials at once, MMR moves only diversity",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig2b_fairness_diversity_coupling")


# ---------------------------------------------------------------- figure 3
def fig3_lambda_manipulation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9))

    rep_gen, rep_ranker = "flanT5Base", "bm25"
    mmr = macro_df[(macro_df["rerank_method"] == "mmr")
                   & (macro_df["generator_name"] == rep_gen) & (macro_df["ranker"] == rep_ranker)]
    cmap = plt.get_cmap("viridis")
    tasks = sorted(mmr["lamp_num"].unique())
    for i, t in enumerate(tasks):
        sub = mmr[mmr["lamp_num"] == t].sort_values("mmr_lambda")
        ax1.plot(sub["mmr_lambda"].astype(float), sub["avg_ild_jaccard"],
                 marker="o", ms=3, color=cmap(i / max(len(tasks) - 1, 1)), label=f"LaMP-{int(t)}")
    ax1.invert_xaxis()
    ax1.set_xlabel("MMR $\\lambda$ (right to left = more diversity weight)")
    ax1.set_ylabel("Mean ILD (Jaccard)")
    ax1.set_title(f"$\\lambda \\rightarrow$ ILD ({cell_tag(rep_gen, rep_ranker)})")
    ax1.legend(ncols=2, frameon=False, fontsize=7)

    width = 0.2
    x = np.arange(len(tasks))
    for i, (gen, ranker) in enumerate(CELLS):
        cell = macro_df[(macro_df["rerank_method"] == "mmr")
                        & (macro_df["generator_name"] == gen) & (macro_df["ranker"] == ranker)]
        pv = cell.pivot_table(index="mmr_lambda", columns="lamp_num", values="avg_ild_jaccard")
        rng = (pv.max() - pv.min()).reindex(tasks)
        ax2.bar(x + (i - 1.5) * width, rng.values, width,
                color=GEN_COLOR[gen], alpha=1.0 if ranker == "bm25" else 0.55)
    ax2.set_xticks(x, [f"{int(t)}" for t in tasks])
    ax2.set_xlabel("LaMP task")
    ax2.set_ylabel("ILD range across $\\lambda$")
    ax2.set_title("Achievable ILD range (all four cells)")
    fig.tight_layout()
    save(fig, "fig3_lambda_manipulation")


# ---------------------------------------------------------------- figure 4
def fig4_ild_signflip():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs, coefs, cis, colors, labels, ps = [], [], [], [], [], []
    for i, (gen, ranker) in enumerate(CELLS):
        mmr_norm = query_norm_df[
            (query_norm_df["rerank_method"] == "mmr")
            & (query_norm_df["generator_name"] == gen) & (query_norm_df["ranker"] == ranker)
        ]
        m = fit_ols(mmr_norm, ["avg_ild_jaccard_norm"], "expected_utility_norm")
        xs.append(i)
        coefs.append(m["coef"]["avg_ild_jaccard_norm"])
        cis.append(ci95(m, "avg_ild_jaccard_norm"))
        colors.append(GEN_COLOR[gen])
        labels.append(f"{GEN_LABEL[gen].split(' ')[0]}\n{RANKER_LABEL[ranker]}")
        ps.append(m["p_value"]["avg_ild_jaccard_norm"])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.errorbar(xs, coefs, yerr=cis, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
    ax.scatter(xs, coefs, c=colors, s=70, zorder=5)
    for x, c, ci, p in zip(xs, coefs, cis, ps):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.annotate(f"{c:+.3f} {stars}", (x, c), textcoords="offset points", xytext=(12, 0),
                    ha="left", va="center", fontsize=8)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Within-MMR coef(ILD) on EU$_{norm}$")
    ax.set_title("Diversity's harm to utility fades with generator size\n(within-MMR regression, fairness held fixed at EE-D = 1)")
    ax.margins(x=0.18, y=0.25)
    save(fig, "fig4_ild_signflip")


# ---------------------------------------------------------------- figure 5
def fig5_matched_diversity():
    # Bin within task (group_by lamp_num) exactly like analyze_mmr_sweep.py section 4:
    # pooling bins across tasks would let task composition confound the delta.
    all_tasks = sorted(query_norm_df["lamp_num"].dropna().unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    width = 0.2
    x = np.arange(len(all_tasks))
    for i, (gen, ranker) in enumerate(CELLS):
        pl_mmr = query_norm_df[
            (query_norm_df["rerank_method"].isin(["pl", "mmr"]))
            & (query_norm_df["generator_name"] == gen) & (query_norm_df["ranker"] == ranker)
        ].dropna(subset=["avg_ild_jaccard_norm", "expected_utility_norm"])
        edges = pl_mmr["avg_ild_jaccard_norm"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
        edges[-1] += 1e-9
        labels = [f"Q{j+1}" for j in range(5)]
        matched_by_task = pool_delta_by_bin(
            pl_mmr, metric_col="avg_ild_jaccard_norm", bin_edges=edges, bin_labels=labels,
            baseline_method="mmr", comparison_methods=["pl"], group_by=["lamp_num"],
        )
        weighted = (
            matched_by_task.dropna(subset=["delta"])
            .assign(w=lambda d: d["n_comparison"])
            .groupby("lamp_num")
            .apply(lambda g: np.average(g["delta"], weights=g["w"]), include_groups=False)
            .reindex(all_tasks)
        )
        ax.bar(x + (i - 1.5) * width, weighted.values, width,
               color=GEN_COLOR[gen], alpha=1.0 if ranker == "bm25" else 0.55,
               label=cell_tag(gen, ranker))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-0.05, 0.05, color="gray", alpha=0.12, zorder=0)
    ax.set_xticks(x, [f"LaMP-{int(t)}" for t in all_tasks])
    ax.set_ylabel("$\\Delta$EU$_{norm}$ (PL $-$ MMR)\nat matched ILD, per task")
    ax.set_title("At matched diversity, stochastic-fair PL adds $\\approx$ nothing over deterministic MMR", pad=44)
    ax.legend(ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    save(fig, "fig5_matched_diversity")


# ---------------------------------------------------------------- figure 7
def fig7_eu_by_diversity_level():
    """Who does better at each diversity level? Mean EU_norm per ILD quintile for
    MMR vs PL, with the deterministic run as the 'original list' reference line.
    To keep the two methods comparable, each (bin, method) mean uses only
    (task, bin) combos where BOTH methods have >= MIN_ROWS rows, averaging tasks
    with equal weight; CIs are a task-stratified bootstrap."""
    MIN_ROWS, ROUNDS = 5, 400
    rng = np.random.default_rng(42)
    METHOD_STYLE = {"mmr": ("#2A9D8F", "o", "MMR (diversity only)"),
                    "pl": ("#E76F51", "^", "PL (stochastic fair)")}

    def strat_boot(task_arrays):
        means = []
        for _ in range(ROUNDS):
            task_means = [arr[rng.integers(0, len(arr), len(arr))].mean() for arr in task_arrays]
            means.append(np.mean(task_means))
        point = np.mean([arr.mean() for arr in task_arrays])
        return np.quantile(means, 0.025), point, np.quantile(means, 0.975)

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4), sharey=True)
    for ax, (gen, ranker) in zip(axes.flat, CELLS):
        cell = query_norm_df[
            (query_norm_df["generator_name"] == gen) & (query_norm_df["ranker"] == ranker)
        ].dropna(subset=["avg_ild_jaccard_norm", "expected_utility_norm"])
        pl_mmr = cell[cell["rerank_method"].isin(["pl", "mmr"])].copy()
        edges = pl_mmr["avg_ild_jaccard_norm"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
        edges[-1] += 1e-9
        pl_mmr["bin"] = pd.cut(pl_mmr["avg_ild_jaccard_norm"], bins=edges, labels=False,
                               include_lowest=True)

        for method, (color, marker, label) in METHOD_STYLE.items():
            xs, ys, lo, hi = [], [], [], []
            for b in range(5):
                sub = pl_mmr[pl_mmr["bin"] == b]
                counts = sub.groupby(["lamp_num", "rerank_method"]).size().unstack(fill_value=0)
                if not {"mmr", "pl"}.issubset(counts.columns):
                    continue
                ok_tasks = counts[(counts["mmr"] >= MIN_ROWS) & (counts["pl"] >= MIN_ROWS)].index
                m_sub = sub[(sub["rerank_method"] == method) & (sub["lamp_num"].isin(ok_tasks))]
                if m_sub.empty:
                    continue
                arrays = [g["expected_utility_norm"].to_numpy()
                          for _, g in m_sub.groupby("lamp_num")]
                ci_lo, point, ci_hi = strat_boot(arrays)
                xs.append(b); ys.append(point); lo.append(point - ci_lo); hi.append(ci_hi - point)
            ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ms=5,
                        linewidth=1.6, capsize=3, label=label)

        det = cell[cell["rerank_method"] == "deterministic"]
        det_eu = det["expected_utility_norm"].mean()
        ax.axhline(det_eu, color="black", linewidth=1.0, linestyle="--",
                   label="deterministic (original list)")
        det_ild = det["avg_ild_jaccard_norm"].mean()
        det_x = float(np.interp(det_ild, edges[:-1], np.arange(5)))
        ax.scatter([det_x], [det_eu], color="black", marker="*", s=110, zorder=5)
        ax.set_xticks(range(5), [f"Q{b+1}" for b in range(5)])
        ax.set_title(cell_tag(gen, ranker))
    for ax in axes[-1]:
        ax.set_xlabel("Diversity level (ILD quintile, Q5 = most diverse)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean EU (normalized)")
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8, loc="lower center",
               ncols=3, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Utility by diversity level: MMR vs PL vs the original deterministic list\n"
                 "(task-matched bins: only (task, bin) cells where both methods have data)",
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "fig7_eu_by_diversity_level")


# ---------------------------------------------------------------- figure 6
def fig6_generator_axis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), sharex=True)
    gens = ["flanT5Small", "flanT5Base"]
    for ax, feature, title in [
        (ax1, "ee_disparity_norm", "M2: fairness cost of exposure equality\ncoef(EE-D) on EU$_{norm}$, controlling ILD"),
        (ax2, "ee_d_x_ild", "M3: EE-D $\\times$ ILD interaction\n(the Small-only second-order structure)"),
    ]:
        for i, gen in enumerate(gens):
            g = query_norm_df[query_norm_df["generator_name"] == gen].dropna(
                subset=["ee_disparity_norm", "avg_ild_jaccard_norm", "expected_utility_norm"]
            ).copy()
            g["ee_d_x_ild"] = g["ee_disparity_norm"] * g["avg_ild_jaccard_norm"]
            feats = (["ee_disparity_norm", "avg_ild_jaccard_norm"] if feature == "ee_disparity_norm"
                     else ["ee_disparity_norm", "avg_ild_jaccard_norm", "ee_d_x_ild"])
            m = fit_ols(g, feats, "expected_utility_norm")
            c, ci, p = m["coef"][feature], ci95(m, feature), m["p_value"][feature]
            ax.errorbar([i], [c], yerr=[ci], fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
            ax.scatter([i], [c], color=GEN_COLOR[gen], s=70, zorder=5)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.annotate(f"{c:+.3f} {stars}", (i, c), textcoords="offset points", xytext=(10, 0),
                        ha="left", va="center", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks([0, 1], [GEN_LABEL[g].split(" ")[0] + "\n" + GEN_LABEL[g].split(" ")[1] for g in gens])
        ax.set_title(title, fontsize=9)
        ax.margins(x=0.35, y=0.3)
    ax1.set_ylabel("OLS coefficient (95% CI)")
    fig.tight_layout()
    save(fig, "fig6_generator_axis")


fig1_headroom()
fig2_fairness_utility_map()
fig2b_fairness_diversity_coupling()
fig3_lambda_manipulation()
fig4_ild_signflip()
fig5_matched_diversity()
fig6_generator_axis()
fig7_eu_by_diversity_level()
print(f"\nAll figures written to {OUT_DIR}")
