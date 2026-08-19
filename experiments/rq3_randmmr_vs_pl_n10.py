"""
N=10-consistent rebuild of the RandMMR-vs-PL comparison (rq3_randmmr_vs_pl.py).

That script compared RandMMR (uniformly run at N=20 samples/query) against each
cell's plain-PL alpha selected via select_best_precision, which independently picks
each (task, generator, ranker, alpha) condition's own highest available N - mostly
N=10, but N=20 for LaMP-3's alpha=2 cells and N=30 for four LaMP-1/2/3/4 Small x BM25
alpha=2 cells. Checked directly: 20 of the 26 chosen PL comparison partners were at
N=10 against RandMMR's N=20. Welch's t-test is valid under unequal N, but the
precision asymmetry itself was undisclosed and inconsistent with how carefully
Table 2 (select_consistent_precision) already treats the same issue for the base
sweep. Per instruction, this rebuilds every cell - both RandMMR and every PL alpha -
at a uniform N=10, so alpha selection (which depends on EE-D means) and the EU
comparison are both apples-to-apples.

EE-D/EE-R cannot be fixed by simply averaging an already-aggregated row: they are
computed once per query, jointly over the full set of that condition's sampled
rankings (framework.metrics.compute_ee), not per-sample-then-averaged like EU/ILD.
So this recomputes EE-D/EE-R from scratch, feeding compute_ee only the first 10
samples (by sample_idx, the order they were generated in) per query - the same
raw retrieval_lists.jsonl / per_list_metrics.jsonl inputs the original run produced,
just subset before the EE computation and the EU/ILD averaging.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import build_query_metric_rows, list_run_dirs
from framework.retrieval import load_retrieval_results
from framework.metrics import compute_ee
from analysis import select_best_precision
from analysis.loading import select_full_coverage_runs
from analysis.with_gold_normalization import normalize_query_rows_with_gold

TABLE_DIR = os.path.join(ROOT, "report", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

N_TARGET = 10
TOP_K = 5
GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]
PL_ALPHAS = [1.0, 2.0, 4.0, 8.0]
PRIMARY_ALPHA, PRIMARY_LO, PRIMARY_HI = 4.0, 0.8, 1.0
CLOSE_ENOUGH = 0.02


def relevance_mapping_path(lamp_num, generator_name):
    return os.path.join(ROOT, "data", f"lamp_utility_labels_{generator_name}", f"{lamp_num}_relevance_mapping.tsv")


def load_jsonl(fp):
    rows = []
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_retrieval_cache = {}
def get_retrieval_results(generator_name, ranker, lamp_num):
    key = (generator_name, ranker, lamp_num)
    if key not in _retrieval_cache:
        _retrieval_cache[key] = load_retrieval_results(
            generator_name=generator_name, ranker=ranker, lamp_num=lamp_num
        )
    return _retrieval_cache[key]


def recompute_run_at_n10(run_dir):
    """One row per qid: EE-D/EE-R recomputed fresh from the first N_TARGET sampled
    rankings (by sample_idx); EU/ILD averaged over that same subset."""
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    cfg = manifest.get("config", {})
    generator_name = cfg.get("generation", {}).get("generator_name")
    ranker = cfg.get("retrieval", {}).get("ranker")
    lamp_num = cfg.get("dataset", {}).get("lamp_num")
    method = cfg.get("rerank", {}).get("method")
    pl_alpha = cfg.get("rerank", {}).get("pl_alpha")
    lam_lo = cfg.get("rerank", {}).get("pl_randmmr_lambda_low")
    lam_hi = cfg.get("rerank", {}).get("pl_randmmr_lambda_high")

    retrieval_results = get_retrieval_results(generator_name, ranker, lamp_num)
    rel_fp = relevance_mapping_path(lamp_num, generator_name)

    lists_by_qid = {}
    for row in load_jsonl(os.path.join(run_dir, "retrieval_lists.jsonl")):
        lists_by_qid.setdefault(row["qid"], []).append(row)

    metrics_by_list_id = {
        row["list_id"]: row for row in load_jsonl(os.path.join(run_dir, "per_list_metrics.jsonl"))
    }

    # A handful of old runs contain queries that have since dropped out of the
    # canonical query set (retrieval_results.json is regenerated as utility-label
    # filtering evolves) - verified this is pure addition/removal, not reordering:
    # for queries present in both, det_indices still point at the same doc_ids under
    # the current file. Skip queries no longer present rather than recomputing EE-D
    # against index positions that no longer mean what they meant at generation time.
    stale_qids = [q for q in lists_by_qid if q not in retrieval_results]
    if stale_qids:
        print(f"    (skipping {len(stale_qids)} qid(s) no longer in current retrieval_results: "
              f"{stale_qids[:5]}{'...' if len(stale_qids) > 5 else ''})")
        for q in stale_qids:
            del lists_by_qid[q]

    out_rows = []
    for qid, rows in lists_by_qid.items():
        subset = sorted(rows, key=lambda r: r.get("sample_idx", 0))[:N_TARGET]
        if len(subset) < N_TARGET:
            continue
        det_indices_per_list = [r["det_indices"] for r in subset]
        ee_res = compute_ee(
            qid=qid,
            det_indices_per_list=det_indices_per_list,
            retrieval_results_for_qid=retrieval_results.get(qid, []),
            rel_mapping_fp=rel_fp,
            top_k=TOP_K,
        )
        eu_scores, ild_scores = [], []
        for r in subset:
            m = metrics_by_list_id.get(r["list_id"])
            if m is None:
                continue
            eu_scores.append(m["eu_score"])
            ild_scores.append(m.get("ild_jaccard"))
        if not eu_scores:
            continue
        out_rows.append({
            "qid": qid, "lamp_num": int(lamp_num), "generator_name": generator_name, "ranker": ranker,
            "rerank_method": method, "pl_alpha": pl_alpha,
            "pl_randmmr_lambda_low": lam_lo, "pl_randmmr_lambda_high": lam_hi,
            "expected_utility": float(np.mean(eu_scores)),
            "max_utility": float(np.max(eu_scores)), "min_utility": float(np.min(eu_scores)),
            "avg_ild_jaccard": float(np.mean([v for v in ild_scores if v is not None])) if any(v is not None for v in ild_scores) else None,
            "ee_disparity": ee_res["ee_disparity"], "ee_relevance": ee_res["ee_relevance"],
            "ee_difference": ee_res["ee_difference"], "n_lists": len(subset),
        })
    return out_rows


print("Identifying source run directories (post dedup/best-precision selection)...")
run_dirs = list_run_dirs()
all_rows_df = pd.DataFrame(build_query_metric_rows(run_dirs))
all_rows_df = select_full_coverage_runs(all_rows_df)
# A handful of LaMP-1/LaMP-4 plain-PL settings have a stray seed=43 duplicate
# run alongside the canonical seed=42 one; select_best_precision doesn't
# dedupe across seeds, so leaving both in silently pools 2x the intended N
# for exactly those (lamp,gen,ranker,alpha) cells - reintroducing a precision
# asymmetry against RandMMR's clean N=10. seed=42 is the project standard
# everywhere else, so pin to it explicitly.
all_rows_df = all_rows_df[all_rows_df["seed"] == 42]

# Full-precision frame (incl. the gold ranker) used only to build each cell's
# with-gold EU ceiling - independent of the N=10 recomputation below, matching
# how experiments/table4_with_gold.py sources ceilings.
ceiling_source_df = select_best_precision(all_rows_df.copy())

raw_df = all_rows_df[all_rows_df["generator_name"].isin(GENERATORS) & all_rows_df["ranker"].isin(RANKERS)]
raw_df = select_best_precision(raw_df)

pl_source_dirs = raw_df[
    (raw_df["rerank_method"] == "pl") & (raw_df["pl_alpha"].isin(PL_ALPHAS))
]["run_dir"].unique().tolist()
randmmr_source_dirs = raw_df[
    (raw_df["rerank_method"] == "pl_randmmr")
    & (raw_df["pl_alpha"] == PRIMARY_ALPHA)
    & (raw_df["pl_randmmr_lambda_low"] == PRIMARY_LO) & (raw_df["pl_randmmr_lambda_high"] == PRIMARY_HI)
]["run_dir"].unique().tolist()
det_rows = raw_df[raw_df["rerank_method"] == "deterministic"].copy()

print(f"{len(pl_source_dirs)} plain-PL run dirs, {len(randmmr_source_dirs)} RandMMR run dirs to recompute at N={N_TARGET}")

n10_rows = []
for i, run_dir in enumerate(pl_source_dirs + randmmr_source_dirs):
    print(f"  [{i+1}/{len(pl_source_dirs) + len(randmmr_source_dirs)}] {os.path.basename(run_dir)}", flush=True)
    n10_rows.extend(recompute_run_at_n10(run_dir))

n10_df = pd.DataFrame(n10_rows)
n10_df.to_csv(os.path.join(TABLE_DIR, "rq3_n10_recomputed_query_rows.csv"), index=False)
print(f"\nsaved {len(n10_df)} recomputed (qid, condition) rows -> rq3_n10_recomputed_query_rows.csv")

pooled = pd.concat([det_rows, n10_df], ignore_index=True)
norm_cells = pooled[["lamp_num", "generator_name", "ranker"]].drop_duplicates().itertuples(index=False, name=None)
qn = normalize_query_rows_with_gold(pooled, ceiling_source_df, cells=list(norm_cells))
qn["expected_utility_norm"] = pd.to_numeric(qn["expected_utility_norm"], errors="coerce")


def choose_alpha(cell_pl_means, rand_ee_d):
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
hybrid = qn[qn["rerank_method"] == "pl_randmmr"]
cells = hybrid[["lamp_num", "generator_name", "ranker"]].drop_duplicates()
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
        "pl_n": int(pl_chosen["n_lists"].mean()) if "n_lists" in pl_chosen.columns else None,
        "randmmr_ee_d": rand_ee_d, "randmmr_eu_norm": rh["expected_utility_norm"].mean(),
        "randmmr_n": int(rh["n_lists"].mean()) if "n_lists" in rh.columns else None,
        "delta_eu_norm": rh["expected_utility_norm"].mean() - pl_chosen["expected_utility_norm"].mean(),
        "p_value": p, "sig": "*" if (pd.notna(p) and p < 0.05) else "",
        "randmmr_better": rh["expected_utility_norm"].mean() > pl_chosen["expected_utility_norm"].mean(),
    })

out = pd.DataFrame(rows).sort_values(["lamp_num", "generator", "ranker"])
out.to_csv(os.path.join(TABLE_DIR, "rq3_randmmr_vs_pl_n10_all_cells.csv"), index=False)
print(f"\nsaved table: rq3_randmmr_vs_pl_n10_all_cells.csv ({len(out)} rows)")
pd.set_option("display.width", 160)
print(out.round(4).to_string(index=False))

n_better = int(out["randmmr_better"].sum())
n_total = len(out)
n_sig_better = int(((out["randmmr_better"]) & (out["sig"] == "*")).sum())
n_sig_worse = int(((~out["randmmr_better"]) & (out["sig"] == "*")).sum())
print(f"\nN=10-vs-N=10: RandMMR EU_norm > matched-PL EU_norm in {n_better}/{n_total} cells "
      f"({n_sig_better} significantly better, {n_sig_worse} significantly worse)")

main = out[out["lamp_num"].isin([1, 4])]
main.to_csv(os.path.join(TABLE_DIR, "rq3_randmmr_vs_pl_n10_lamp1_lamp4.csv"), index=False)
other = out[~out["lamp_num"].isin([1, 4])]
print(f"\nLaMP-1/LaMP-4 rows: {len(main)}; other-task rows: {len(other)}, "
      f"RandMMR better in {int(other['randmmr_better'].sum())}/{len(other)} of those")

print("\n% ---- Table 3 (LaMP-1/LaMP-4) LaTeX body ----")
GEN_SHORT = {"flanT5Small": "T5-Small", "flanT5Base": "T5-Base"}
RANK_SHORT = {"bm25": "BM25", "contriever": "Contriever"}
for _, r in main.iterrows():
    sig = r["sig"] if r["sig"] else ""
    print(f"LaMP-{int(r['lamp_num'])} & {GEN_SHORT[r['generator']]} & {RANK_SHORT[r['ranker']]} & "
          f"{int(r['chosen_pl_alpha'])} & {r['pl_ee_d']:.2f}/{r['pl_eu_norm']:.2f} & "
          f"{r['randmmr_ee_d']:.2f}/{r['randmmr_eu_norm']:.2f} & ${r['delta_eu_norm']:+.3f}{sig}$ & \\\\")

print("\n% ---- Full appendix table LaTeX body ----")
for task, g in out.groupby("lamp_num"):
    print(f"%--- LaMP-{int(task)} ({len(g)} rows) ---")
    for i, (_, r) in enumerate(g.iterrows()):
        taskcell = f"\\multirow{{{len(g)}}}{{*}}{{LaMP-{int(task)}}}" if i == 0 else ""
        sig = r["sig"] if r["sig"] else ""
        print(f"{taskcell} & {GEN_SHORT[r['generator']]} & {RANK_SHORT[r['ranker']]} & {int(r['chosen_pl_alpha'])} & "
              f"{r['pl_ee_d']:.2f}/{r['pl_eu_norm']:.2f} & {r['randmmr_ee_d']:.2f}/{r['randmmr_eu_norm']:.2f} & "
              f"${r['delta_eu_norm']:+.3f}{sig}$ \\\\")
