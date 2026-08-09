# Project Status & Handoff

This document reconstructs what was tried, in what order, what's
still open, and how to continue — based on the project proposal, full git history
(including two abandoned branches recovered from local git objects), and a read-through
of all three notebooks and the current codebase.

**Update (2026-07-23):** decided to continue from Effort B only. Repo cleaned up
accordingly: `main` is now the single branch (`refactored-experimentation` and the
recovered `experiment-recreation` branch were both retired — nothing was lost, see
below); `experiment.py`/`normalize_eu.py` moved to `legacy/`; the two salvageable
artifacts from the abandoned `experiment-recreation` branch (MLX generator code, the
TREC-RAG-2024 notebook) were extracted into `legacy/` as plain files before that branch
was deleted; `eval/` was renamed to `utility_metrics/` to stop it being confused with
`utility_labels/lamp_eval.py`; the large data zips are no longer tracked in git (still
present locally, regeneration path documented in `README.md`). The narrative below
still describes the full history accurately — only the current file layout has moved.

**Update (2026-07-23, later):** this document itself moved to `docs/PROJECT_STATUS.md`,
alongside the project proposal and paper PDFs (`docs/project_proposal.pdf`,
`docs/fair_rag_paper.pdf`) — see `docs/README.md` for an index. `README.md` and
`CLAUDE.md` stay at the repo root (GitHub/Claude Code both expect them there).

For the architecture of the code as it exists today, see `CLAUDE.md` — this document
focuses on the *story* (what was tried, why, and what's left), not the code layout.

## 1. Where this started: the proposal

From `docs/project_proposal.pdf` (the original project proposal):

- **Starting point**: the paper *"Towards Fair RAG"* (Eun Kim & Diaz, 2025) — this repo
  is a fork of its official code (`kimdanny/Fair-RAG`) — extends fair-ranking ideas
  (Castillo, 2018) into the RAG reranking step, and claims fair rankings can maintain or
  even *improve* generation quality relative to standard RAG.
- **Your shared skepticism**: that conclusion seems to contradict the usual
  fairness/performance tradeoff, so the proposal set out to (a) stress-test it with
  additional metrics, and (b) test your own hypothesis:
- **The actual research question** (verbatim from the proposal):
  > *"Does Fairness directly improve RAG performance, or does the performance gain
  > actually come from the increased diversity introduced through Fairness
  > interventions?"*
- **Planned method**: reproduce the paper's LaMP-based experiments, add diversity
  metrics on top of the paper's fairness/utility metrics, and see whether diversity —
  not fairness per se — is the real driver of any performance gain.

**This is important**: the "diversity as the mechanism" hypothesis you'll find baked
into `fair_rag_diversity_story.ipynb` (see §4) isn't something Avi invented later — it's
the core deliverable you two committed to in the proposal. It is still unresolved (see
§5) and is the natural centerpiece of whatever you do next.

## 2. Timeline

| Date | Commit | What happened |
|---|---|---|
| 2024-09-16 → 09-18 | `fe758df` → `fef9544` | Original paper authors' code (`kimdanny/Fair-RAG`). This is the shared fork point everything below branches from. |
| 2026-03-03 | `5078103` (`main`) | Avi adds a reproducibility notebook (`fair_rag_reproduction.ipynb`) + CPU/GPU compatibility fixes to `generator/lm.py` — a first pass at just getting the original paper's code running. |
| 2026-03-14 | *(same commit as `5078103`)* | Avi tags this exact state as `backup/pre-fork-reset-20260314` — a safety snapshot, not separate work — right before starting a new effort from scratch. (Recovered and confirmed identical to `main`; the extra branch ref has since been dropped as redundant.) |
| **2026-03-17 → 04-14** | `ed66372` → `b443d81` | **Effort A: `experiment-recreation`** branch — an ad-hoc, notebook-driven recreation of the paper's pipeline. See §3. |
| **2026-04-15 → present** | `ebfaa38` → `bacf6db` | **Effort B: `refactored-experimentation`** branch — a clean restart with a config-driven framework, started the day after Effort A's last commit, from the *same* fork point (not built on Effort A). This is what you have checked out now. See §4. |
| 2026-07-23 | — | You detach from Avi's fork into your own repo (`edanperetz2/Fair-RAG`), recover Effort A's orphaned branch, and write this document. |

Both efforts start from the same point and were never merged — Avi tried one approach,
set it aside, and restarted with a different architecture. Nothing was lost: Effort A's
full history is recovered locally on branch **`experiment-recreation-recovered`**.

## 3. Effort A — `experiment-recreation` (abandoned, now recovered as a branch)

8 commits, `ed66372` → `b443d81`, all authored by Avi. Ad-hoc and script/notebook-heavy
rather than framework-driven:

- `analyze_results.py`, `notebook_experiment_utils.py`, `run_recreation.ps1`,
  `smoke_test_ee.py` — standalone helper scripts, no shared abstraction layer.
- One ever-growing notebook, `modular_t5small_bm25_experiment.ipynb`, edited in place
  across almost every commit (920 → 2100+ lines) rather than parameterized/looped.
- `multimodel_t5base_splade_experiment.ipynb` — a second model (Flan-T5-Base) + SPLADE
  retriever combination.
- `multimodel_lfm25_mlx_experiment.ipynb` + **`generator/lm_mlx.py`** — a real, working
  integration for running quantized on-device models via Apple's MLX framework
  (LiquidAI's LFM 2.5 model). This explains a loose end you'll hit on your current
  branch: `fair_rag_experiment.ipynb`'s config table lists generator names
  `bonsai8BMLX1bit` / `lfm25MLX12B4bit` / `qwen35MLX4B4bit` that **do not work** on
  `refactored-experimentation` — `utils.py`'s `models_info` has no entries for them and
  `generator/lm.py::PromptLM` hard-crashes (`NotImplementedError`) on any non-"T5" model
  name. The real implementation exists only here, on the abandoned branch, and was never
  ported over.
- `trec_rag_2024_dataset_overview.ipynb` (1539 lines, added in the final commit) — an
  exploration of the newer **TREC RAG 2024** benchmark, suggesting Avi was considering
  extending beyond LaMP. Never acted on further.
- Several commits include actual run artifacts committed directly to git
  (`progress.csv`, `params.json`, and in the final commit `llm_outputs.jsonl` — real LLM
  outputs from completed runs). These are inspectable right now, without rerunning
  anything, on the `experiment-recreation-recovered` branch.

**Why it looks abandoned**: commits grew increasingly large and unwieldy (one commit
alone touched 208 files / +285k lines, mostly notebook output + data bloat), the
notebook was never refactored into something reusable, and there's no config/resume
system — every setting change meant hand-editing the monolithic notebook. Effort B
appears to be a direct reaction to this: a clean rewrite with a proper config layer and
crash-safe resumability.

**Worth mining before discarding**: the MLX generator code, the TREC-RAG-2024 notebook
(as a scoping idea), and the committed run artifacts. Browse it with:
```
git log experiment-recreation-recovered --oneline
git diff refactored-experimentation experiment-recreation-recovered --stat
```

## 4. Effort B — `refactored-experimentation` (current, active)

This is what's checked out today. Started `ebfaa38` (resumable framework) → `93fafd6`
(pipeline refactor) → `437d3c6` (PL-MMR rerank method + diversity-analysis notebooks) →
`bacf6db` (your latest commit: notebook update + `CLAUDE.md`).

Architecture is fully documented in `CLAUDE.md` — the short version: a `framework/`
package (`RunConfig`, `ExperimentRunner`, `BatchExperimentRunner`) that runs deterministic
/ MMR / Plackett-Luce / PL-MMR rerankings over LaMP, computes Expected Exposure +
Expected Utility + diversity (ILD-Jaccard) per query, and persists everything as
crash-safe JSONL so runs can resume. Three root notebooks drive it:

- **`fair_rag_experiment.ipynb`** — the run-driver. Not one clean experiment but ~15
  sequential "campaigns" appended over time (per-LaMP-task BM25 sweeps, cross-ranker
  normalized comparisons, missing-settings sweeps, a PL-MMR hybrid sweep). No
  interpretive markdown — pure plumbing + plots. Known rough edges:
  - Cell 1 is empty/dead.
  - One cell (full Contriever sweep) has a **past kernel crash** logged, but it happened
    *after* all 7 LaMP tasks had already finished and their summaries were written — so
    it looks like a Jupyter cleanup crash, not lost computation. Worth confirming before
    trusting that cell blindly.
  - Two cells hardcode a specific `run_id` string to resume one particular interrupted
    run — brittle, won't apply to your new runs.
- **`fair_rag_diversity_story.ipynb`** — the analysis notebook that most directly
  answers the proposal's research question (§1, §5).
- **`fair_rag_stats_exploration.ipynb`** — the most methodologically careful notebook:
  bootstrap CIs, permutation tests, a small decision-tree probe, explicit smoke tests.
  Portable (derives its repo root correctly); the other two notebooks are more
  exploratory in character.

**Known blockers on this machine right now:**

- `experiment_runs/` (where the framework writes all run output) **does not exist here**
  — it's gitignored and lived only on Avi's Mac (hardcoded paths like
  `/Users/asimk/Code/Fair-RAG/...` show up in `fair_rag_diversity_story.ipynb`). Every
  analysis cell in the two analysis notebooks will show empty results until experiments
  are re-run.
- `fair_rag_diversity_story.ipynb`'s first code cell hardcodes
  `REPO_ROOT = Path("/Users/asimk/Code/Fair-RAG")` — fix this to use `os.getcwd()`
  (like `fair_rag_stats_exploration.ipynb` already does) before that notebook will work
  here at all.
- Only **one** retrieval cache is precomputed:
  `retrieval/retrieval_results/flanT5Small/bm25/4.json` (LaMP-4, BM25, Flan-T5-Small).
  Every other generator/ranker/LaMP-task combination needs `retrieval/rank_profiles.py`
  (BM25/SPLADE/Contriever) or `retrieval/gold_retriever.py` (oracle) run first, per the
  data pipeline documented in `CLAUDE.md`.
- No experiment has actually been executed via the new framework on this machine —
  `experiment_runs/` is simply absent, not partially populated.

## 5. The open research question — where the evidence currently stands

The proposal's question — *is fairness's benefit really just increased diversity?* — is
tackled directly in `fair_rag_diversity_story.ipynb`, using data Avi generated on his
own machine (314 runs, 145,544 query rows, not present in this checkout). The evidence
collected so far is **genuinely mixed, and the notebook stops short of a real answer**:

- Global correlation between diversity (ILD) and fairness (EE-D): **r = −0.041**
  (statistically "significant" only because of the huge sample size — the effect size
  is essentially zero). This cuts *against* a simple "fairness → diversity → better
  performance" story.
- But looking at a more targeted comparison — Plackett-Luce (fairness via randomization)
  vs. MMR (fairness via *explicit* diversity) — **MMR beats PL on utility in 4 of 5
  ranker settings** (Contriever, Gold, SPLADE k=3, SPLADE k=5), losing only on BM25. If
  explicit diversity-seeking methods match or beat randomization-based fairness at
  similar fairness levels, that's actual (if partial) support for the diversity-mechanism
  hypothesis.
- The notebook's own final markdown cell is explicitly labeled **"Conclusion draft"**
  and hedges rather than concluding — it never resolves the tension between these two
  findings.
- `fair_rag_stats_exploration.ipynb` adds relevant context that hasn't been folded into
  the diversity story: EE-R↔utility correlation is unstable across LaMP tasks (ρ from
  0.035 to 0.466), base retrievers (BM25/SPLADE/Contriever) are only marginally better
  than chance at predicting relevance (ROC AUC 0.55–0.58), and even gold/oracle retrieval
  produces low utility on some tasks — hinting the generation model (`flanT5Small`), not
  retrieval or reranking, may be the real bottleneck on at least some LaMP tasks. This
  notebook's own last cell explicitly calls for **"a tighter confirmatory pass"** that
  was never done.

**This is the natural centerpiece of picking the project back up**: synthesize what's
already there (don't restart from zero — the PL-vs-MMR utility comparison and the
retriever-AUC finding are real, usable results) and run the confirmatory analysis both
notebooks call for but never finished.

## 6. Recommended next steps, in order

1. Fix the hardcoded Mac path in `fair_rag_diversity_story.ipynb` (cell 1).
2. Before spending compute re-running anything, check whether
   `experiment-recreation-recovered`'s committed run artifacts (`llm_outputs.jsonl`,
   `progress.csv`, `params.json`) contain anything reusable.
3. Precompute whatever retrieval caches you need beyond the one that already exists
   (`retrieval/rank_profiles.py` for BM25/SPLADE/Contriever, `retrieval/gold_retriever.py`
   for the oracle) — see `CLAUDE.md`'s data pipeline section.
4. Re-run `fair_rag_experiment.ipynb` for the settings you need — its batch mode already
   has safe skip/resume logic (`BATCH_REUSE_POLICY = "smart"`), so it's safe to rerun
   incrementally rather than needing one giant run.
5. Finish the diversity-mechanism analysis: reconcile the near-zero ILD↔EE-D correlation
   with the PL-vs-MMR utility comparison, and run the "tighter confirmatory pass" both
   analysis notebooks call for but never completed. This directly answers the proposal's
   research question.
6. Separately and deliberately (don't silently revive) decide whether the MLX local-model
   support or the TREC-RAG-2024 extension from Effort A are worth pursuing further, or
   were dead ends worth explicitly closing off.

## 7. Update (2026-07-24): a real scoped experiment was run, executed, and compared to the paper

Steps 3-5 above were carried out for real, on this machine, GPU-accelerated (CUDA
verified working via `torch.cuda.is_available()` + an on-device matmul before any
experiment code ran). Scope was deliberately narrowed to keep the run affordable:
BM25 only (SPLADE/Contriever retrieval was never precomputed — no `retrieval/rank_profiles.py`
run for them), `flanT5Small` only, `nq=100` (or fewer, where a LaMP task has under 100
queries after filtering — LaMP-1 has 51), 4 rerank settings per task (`deterministic`,
`mmr λ=0.55`, `pl α=2 s=10`, `pl α=8 s=10`), all 7 LaMP tasks, seed 42.

**Timing** (from `experiment_runs/batches/scoped_rq_experiment/batch_summary.json`
manifests, wall-clock, this machine): 28 settings, 2,604 (qid, list_id) generation
units, **5,856s (97.6 min) total**. Per-unit cost is dominated by LLM generation calls
(not reranking), roughly 0.1-0.8s/unit depending on task (LaMP-4/6/7's longer-output
tasks cost more per generation than LaMP-1/2's short classification answers). This is
~4.5% of the ~14,600-query full-scale scope mentioned when this run was planned — a
naive linear extrapolation to a full 3-retriever x 4-generator x 7-task x 4-rerank
sweep (matching the paper's 12 RAG models) would be on the order of **days of
continuous compute**, which is why this run stayed deliberately scoped rather than
attempting the paper's full grid on a single laptop.

**Finding 1 (structural, methodological — found by actually running the framework,
not derivable from reading the code alone): EE-D cannot register below 1.0 for any
single-list method.** `deterministic` and `mmr` each produce exactly one ranked list
per query; the vendored `expected_exposure` disparity metric is only ever non-trivial
across *multiple* differently-ordered lists for the same query. Confirmed empirically:
every deterministic and MMR run in this scoped experiment has `ee_disparity_norm`
pinned at 1.0 for 100% of queries. Only `pl` (which samples `pl_samples=10`
independent rankings per query) produces a spread of EE-D values. Practically, this
means **MMR is not a fairness intervention in the EE-D sense used by this paper** —
it's an unrelated (diversity-based) reranking method that happens to also get compared
on the same utility axis. Any RQ2-style "utility at a given disparity level" analysis
is only meaningful for the PL family of settings.

**Finding 2 (paper comparison): reproduced Table 2's methodology exactly, including a
bug it exposed.** The paper's Table 2 compares each disparity-bin's average
fairer-model utility against a *single, unbinned* deterministic-baseline utility
score (necessarily unbinned, per Finding 1 — the deterministic run only ever has one
EE-D value). `analysis/binning.py::pool_delta_by_bin` originally binned the baseline
too, which silently produced `NaN` in every bin except the single one the baseline's
EE-D value fell into — this was caught only once we tried to actually reproduce
Table 2's numbers and got mostly-NaN output. Fixed by adding a `bin_baseline=False`
mode (see `analysis/binning.py`, committed). With that fix, the paper-comparable
table for this scoped run (PL vs. deterministic, pooled Δ normalized EU by EE-D bin,
per LaMP task) shows the same *qualitative* shape the paper reports: near-baseline or
better utility once EE-D climbs into `[0.6, 1.0)`, and the worst deltas concentrated
in the most-disparate `[0.0, 0.2)`-`[0.2, 0.4)` range — e.g. LaMP-2 shows +0.11 to
+0.21 in `[0.4, 1.0)` vs. baseline, LaMP-4 shows +0.12 to +0.28 in the same range.
**Caveat:** several bins have very small n (as low as 2-7 comparison queries, since
this run only pooled `α=2` and `α=8`, not the paper's full `α∈{1,2,4,8}` sweep, and
`nq=100` vs. the paper's 51-833 queries per task) — treat this as a directional
replication, not a statistically powered one.

**Finding 3 (mediation test, extends beyond what the paper reports): diversity (ILD),
not disparity (EE-D), appears to be the actual driver of utility differences.**
`EU ~ EE-D` alone: coef=+0.016, p=0.53 (not significant). Adding ILD:
`EU ~ EE-D + ILD`: coef(EE-D) shrinks to +0.006 (p=0.81, still not significant),
coef(ILD)=-0.27 (p=0.00014, significant) — a **62% shrinkage in the EE-D coefficient**
once ILD is controlled for, classic evidence that ILD is doing the explanatory work
here, not EE-D directly. Note the sign: higher intra-list diversity is associated with
*lower* utility in this data, opposite to a naive "more diverse retrieval helps
generation" story — plausible mechanism: for these short LaMP profiles, less-similar
retrieved items are less likely to *all* be relevant to the same target, so ILD may be
partly proxying for retrieval-quality dilution rather than beneficial variety. This
should be treated as suggestive, not conclusive (R²=0.007 even with both terms — EE-D
and ILD together explain very little of the variance in per-query utility; most of
what determines utility here is presumably query/task-specific, not rerank-method-driven).

### Concrete next-step experiments (in rough priority order)

1. **Widen the α sweep to match the paper (α ∈ {1, 2, 4, 8}), same scope otherwise.**
   Directly fills the gaps in Finding 2's small-n bins without adding a new retriever
   or generator — cheapest way to get a real statistically-powered Table-2 replication.
2. **Add a second generator (`flanT5Base`, already have label data unzipped) at the
   same BM25/nq=100 scope.** Tests whether Finding 3's negative ILD coefficient is a
   `flanT5Small`-specific artifact (a weak, easily-diluted generator) or holds up with
   a stronger model — the paper's own Table 1 shows the fairness-quality tradeoff slope
   itself varies by retriever, so a size-varying generator comparison is a natural
   companion axis they didn't isolate this way.
3. **Precompute SPLADE and/or Contriever retrieval** (`retrieval/rank_profiles.py
   --ranker splade`) **for the same scoped tasks/generator**, to test whether Finding 1
   and Finding 3 are BM25-specific. The paper's Table 1 already shows retriever choice
   changes the fairness-quality tradeoff slope substantially (BM25 slope 0.14 vs.
   SPLADE 0.20 on LaMP-1) — worth checking whether it also changes the sign of the ILD
   mediation effect.
4. **Run an MMR-λ sweep** (e.g. λ ∈ {0.25, 0.5, 0.65, 0.75, 0.9}) instead of the
   single λ=0.55 used here. Since Finding 1 shows MMR can't move EE-D, the interesting
   question for MMR specifically is whether *utility* varies smoothly with λ on its
   own diversity axis — a different, ILD-focused research question than the EE-D-based
   RQ2, but directly relevant to Finding 3's diversity-mediation result.
5. **Increase `pl_samples` beyond 10** (e.g. 20-30) for a subset of tasks, to check
   whether the EE-D spread and utility estimates are sample-size-stable — 10 samples
   is on the low end for a Monte Carlo disparity estimate and could be adding noise to
   Finding 2's bin assignments.
6. **A dedicated, adequately-powered mediation analysis**: pool a wider α sweep (item 1)
   with a larger `nq` for at least the smaller LaMP tasks (LaMP-1/2/3 are cheap per-unit;
   LaMP-1 already uses its full 51-query corpus, but LaMP-2/3 could go well past 100)
   specifically to shrink Finding 3's confidence intervals — right now R²=0.007 makes
   it impossible to say how much of the true relationship this mediation model captures.

## 8. Update (2026-07-25): the widened α sweep (item 1 above), executed

Ran item 1 from §7's next-steps list: widen the PL α sweep from {2,8} to the full
paper-matching {1,2,4,8}. **Actual execution deviated from plan** (full detail in
the plan file used for this work, `happy-bouncing-walrus.md`, but summarized here
since it materially affects how to read the results below):

- The active Windows power scheme was "Legion Balance Mode" (a Lenovo OEM profile),
  which capped GPU clocks to ~13% of max and made the run ~2.5-4x slower than
  estimated — not discovered until partway through, since the original plan only
  checked GPU idle/memory, not the active power profile. Switching to "Legion
  Performance Mode" partly recovered speed but not fully (likely some genuine
  thermal buildup after many continuous hours of compute on top of the profile
  issue). Given the revised ETA was still ~15-22 more hours, the decision was made
  to stop the N=30 sweep partway and finish the remainder at the cheaper N=10.
- **Two real robustness bugs surfaced and were fixed** (both committed to
  `framework/`): `flush_manifest()` crashed the entire batch on a Windows
  `PermissionError` from `os.replace` (this repo lives under OneDrive, whose sync
  client can transiently lock a just-rewritten file — happens on every single
  completed unit since `flush_every=1`, so one unlucky collision killed a 9-hour
  job at unit 952/79422); fixed with a retry-with-backoff. `BatchExperimentRunner`
  also now isolates a per-setting failure (logs it, continues to the next config)
  instead of aborting the whole batch — one bad setting no longer costs the rest.
- **Final data composition** (49 total run directories, all individually verified
  `completed` with matching query counts and non-empty artifact files):
  LaMP-1/2/3 have a full, internally-consistent α∈{1,2,4,8} sweep at **N=30**.
  LaMP-4 has α∈{1,2,4} at **N=30** and α=8 at **N=10** (backfilled for free from the
  original §7 run, which already had it — this task's 4-point curve has mixed
  precision, flagged wherever it's shown). LaMP-5/6/7 have all 4 α values but only
  at **N=10** (α=2/8 reused from the original §7 run; α=1/4 freshly computed).
  `experiments/analyze_scoped_results.py` now has a `select_best_precision()` step
  that automatically keeps only the highest-`pl_samples` row per (LaMP task, α) and
  drops the lower-precision duplicate where both exist (e.g. LaMP-4's α=1/2/4 at
  N=10 from the original run are correctly superseded by their N=30 versions).

**Results, on the deduplicated 42-setting / 3,906-query-row dataset** (up from 28
settings / 2,604 rows in §7):

- **Paper-style Table 2 replication is now meaningfully better powered.** Most bins
  across all 7 tasks now have 20-200+ comparison queries (vs. as few as 2-7 before).
  A few thin bins remain (e.g. LaMP-1's `[0.2,0.4)` bin has n=1, LaMP-7's `[0.0,0.2)`
  has n=4) — LaMP-1 and LaMP-7's α-to-EE-D mapping is evidently less smooth than the
  other tasks', concentrating queries at the extremes. The qualitative pattern from
  §7 holds: deltas trend toward flat-or-positive as EE-D rises through `[0.4,1.0)`,
  worst in `[0.0,0.2)`.
- **The ILD↔EE-D correlation is now statistically significant with the larger
  sample**: r=-0.045, p=0.005, n=3,906 (was r=-0.037, p=0.06, n=2,604 in §7). Still a
  practically negligible effect size — more data made a real-but-tiny effect
  detectable, it didn't make the effect bigger.
- **Mediation test, rerun on n=3,552 (up from 2,184): EE-D's coefficient is not just
  small, it's unstable in sign.** `EU ~ EE-D` alone: coef=+0.0034, p=0.84. Adding
  ILD: coef(EE-D) flips to -0.0077, p=0.65 — still nowhere near significant, and the
  simple shrinkage-percentage framing from §7 breaks down here (a coefficient that's
  indistinguishable from zero in both models and flips sign isn't meaningfully
  "shrinking," so that metric is dropped for this dataset rather than reported as a
  misleading number). `coef(ILD)=-0.308, p=6.9e-08` — the ILD effect direction and
  significance from §7 holds up and gets *more* significant with more data, while
  EE-D's apparent (already-weak) effect in §7 does not replicate as anything more
  than noise. R²=0.008, still tiny — EE-D/ILD together explain very little per-query
  utility variance.
- **New: a borderline-significant EE-D×ILD interaction**, not visible in §7's
  smaller sample. `EU ~ EE-D + ILD + EE-D:ILD`: coef(EE-D)=+0.298 (p=0.058),
  coef(ILD)=-0.081 (p=0.53), coef(EE-D×ILD)=-0.316 (p=0.050). Read cautiously (both
  right at the conventional 0.05 threshold), but the pattern is coherent: at low
  ILD, higher EE-D trends toward *higher* utility; that relationship weakens or
  reverses as ILD increases. This is a genuinely new hypothesis this dataset
  surfaces — EE-D's effect on utility may be conditional on diversity level, not a
  fixed effect the earlier simple/additive models could detect. Needs a larger,
  purpose-built sample before treating as more than a lead worth following up.

**Process lesson for future long runs**: check the *active Windows power scheme*
(`powercfg /getactivescheme`), not just GPU idle/memory, before estimating
multi-hour GPU workload timing on a laptop — OEM power profiles (Lenovo Legion's
Balance/Performance/Quiet modes here) can silently cap clocks well below what
`nvidia-smi`'s idle reading suggests is available.

## 9. Update (2026-07-25): the MMR-λ diversity sweep — the direct mechanism test

**What ran**: the experiment planned at the end of the previous session (plan file
`~\.claude\plans\based-on-previous-experiments-misty-hamming.md`): MMR at
λ ∈ {0.15, 0.3, 0.45, 0.7, 0.85, 1.0} × all 7 LaMP tasks, BM25/flanT5Small/nq=100/
seed=42 — 42 new runs on top of the existing 49 (total now 91). Rationale: MMR is
deterministic (EE-D pinned at 1.0 at every λ), so λ manipulates diversity with
fairness held perfectly fixed — the cleanest test this framework can produce of
whether diversity, not fairness, drives utility. Execution was clean: pre-flight
caught the power scheme back on Balance Mode (switched before launch, per the §8
lesson); one relaunch was needed because flaky HuggingFace network checks were
stalling model loads (fixed with HF_HUB_OFFLINE=1 — model fully cached); 28.9 min
total on fresh-GPU rates. All 91 manifests completed with full query counts.
Analysis: `experiments/analyze_mmr_sweep.py`.

**Sanity check passed exactly**: λ=1.0 (pure-relevance MMR) reproduces the
deterministic run's EU and ILD to 6 decimals on all 7 tasks — the greedy MMR loop
with the diversity term zeroed out is the identity reranking, as designed.

**Finding 1 — the manipulation is real but weaker than hoped on most tasks.**
λ→ILD ranges per task: 0.01–0.09 on six tasks, 0.25 on LaMP-6. BM25's top-pool
candidates are already lexically similar, so even λ=0.15 can only diversify so
much. LaMP-6 (email subjects, apparently more heterogeneous pools) is the
exception. Within-task estimates below are correspondingly underpowered.

**Finding 2 — the core within-MMR regression (EE-D fixed): the pooled negative ILD
effect replicates, but per-task it's a composition story.** Pooled EU_norm~ILD_norm
on MMR rows only: coef −0.247 (p=1.7e-05, n=4,144) — so §8's negative ILD
coefficient is not a PL-randomization confound; it survives with fairness held
fixed. But per task: 5 of 7 non-significant, and the two (near-)significant ones
go in opposite directions — LaMP-7 −0.68 (p=0.043), LaMP-2 +2.53 (p=0.069). The
raw λ→EU table says the same thing more plainly: more diversity helps the
classification tasks (LaMP-2: 0.10→0.17 EU from λ=1.0 to λ=0.15; LaMP-3:
0.46→0.53-0.56) and mildly hurts generation tasks (LaMP-5, LaMP-7). The pooled
negative coefficient is substantially cross-task composition, not a uniform
within-task law. "Does diversity help?" has a task-dependent answer.

**Finding 3 — the matched-diversity fairness test: randomization adds nothing once
diversity is matched.** Binning all PL+MMR query rows into ILD quantiles and
comparing PL vs MMR utility within bins: deltas per bin between −0.044 and +0.020
(mostly negative); per-task n-weighted means between −0.035 and +0.016 with no
consistent sign. At matched diversity, the stochastic/fairness component of PL
contributes ~nothing to utility (if anything, slightly negative). This is the most
direct answer to the proposal's question the project has produced.

**Finding 4 — the §8 interaction lead, re-fit on n=7,104 (doubled)**: the EE-D×ILD
interaction is now marginally significant (coef −0.246, p=0.049) with a
marginally positive EE-D main effect (+0.243, p=0.041) — same coherent pattern as
§8 (at low ILD, less-fair rankings trend toward higher utility; the effect
cancels around ILD_norm≈0.99, which is where most of the data lives — hence the
~zero average EE-D effect). R² remains ~0.006 throughout: whatever structure is
here, it explains almost nothing of utility variance. Still lead-grade, not
conclusion-grade.

**Where this leaves the research question** (proposal: "does fairness directly
improve RAG performance, or does the gain come from the diversity it introduces?"):
on this replication's evidence — (a) fairness (EE-D) has no direct utility effect;
(b) at matched diversity, randomization-based fair reranking (PL) adds nothing
over explicit-diversity reranking (MMR); (c) diversity itself has small,
task-dependent effects (helps classification, mildly hurts generation), not a
uniform benefit; (d) so the paper's "fair rankings can maintain or even improve
quality" reads, in this setup, as "fairness interventions are approximately
utility-neutral, and what little movement exists is attributable to their
diversity side-effect, whose sign depends on the task." The remaining open lead
is the low-ILD conditional EE-D effect (Finding 4).

## 10. Update (2026-07-25): the Contriever generalization sweep

**What ran**: the full BM25 design replicated on Contriever — deterministic + MMR
λ∈{0.15,0.3,0.45,0.55,0.7,0.85,1.0} + PL α∈{1,2,4,8} at N=10, all 7 LaMP tasks,
nq=100, seed=42. 84 new runs (175 total), 31,248 generation units, 224.5 min.
Retrieval precompute (~50 min) + the run were done deliberately on **Balance
Mode** after the chassis thermally saturated on Performance (87°C, sw-throttle
active, ~117W — see the updated [[gpu-power-profile-gotcha]] policy: Balance's
~80W is roughly this chassis's sustainable envelope anyway, so the sustained-run
cost is modest). λ=1.0 reproduced deterministic **exactly** on all 7 tasks on
this ranker too. Analysis: `experiments/analyze_mmr_sweep.py`, now ranker-aware
(loops per-ranker sections over BM25 and Contriever, pooled interaction re-fit).
Note: per-query normalization now pools both rankers' rows per (task, qid) group,
so §9's BM25 numbers shift by ±0.01-0.02 when recomputed — qualitatively
unchanged, but §9 and §10 tables are not digit-for-digit comparable.

**Hypothesis that failed**: the stated motivation for choosing Contriever — dense
retrieval pools would be lexically more diverse, widening the λ→ILD manipulation
range — was wrong. Ranges are nearly identical to BM25's (0.02-0.08 on six tasks,
0.23 on LaMP-6). In hindsight the reason is structural: LaMP reranks documents
from a single user's own profile, so pool homogeneity is intrinsic to the data,
not to the retriever. Widening the manipulation would require a different
corpus/benchmark (e.g. TREC-RAG), not a different ranker.

**Generalization results — the three headline findings on a second ranker**:
1. **Fairness has no direct utility effect — replicates.** Pooled two-ranker
   Model 2 (n=13,228): coef(EE-D)=+0.023, p=0.051 — still marginal/near-zero.
2. **At matched diversity, PL adds nothing over MMR — replicates.** Contriever
   per-task n-weighted deltas: +0.016/−0.013/+0.013/+0.006 (tasks 4-7; tasks 1-3
   had no valid matched bins after normalization pruning) — small, sign-mixed,
   no consistent advantage, same as BM25.
3. **Diversity's effect is task-dependent — replicates in pattern, and the pooled
   negative does NOT replicate.** Contriever within-MMR pooled: coef(ILD)=−0.072,
   p=0.25 (vs BM25's −0.247, p=1.7e-05). Per task: LaMP-2's positive diversity
   effect is now *significant* (+2.66, p=0.026; was +2.53, p=0.069 on BM25) and
   the raw table agrees (EU 0.17→0.22 from λ=1.0 to λ=0.15); generation tasks
   flat-to-negative (LaMP-5: 0.202→0.181; LaMP-6: 0.161→0.133). This strengthens
   the §9 interpretation: BM25's pooled negative was substantially cross-task
   composition. There is no uniform "diversity is good/bad" law — classification
   benefits, generation mildly suffers.

**The interaction lead graduated.** EE-D×ILD on the pooled two-ranker dataset
(n=13,228): coef(EE-D)=+0.326 (p=0.00015), coef(EE-D×ILD)=−0.327 (p=0.00038) —
from p=0.050 (§8) to p=0.049 (§9) to p<0.001 with each doubling of data, same
coherent pattern throughout (at low diversity, less-fair rankings trend toward
higher utility; the effect cancels at ILD≈1, where most data lives). R² remains
~0.003, so this is a statistically robust but practically tiny structure —
honest framing: a real, replicable second-order effect, not a driver of RAG
quality.

**Bottom line for the research question**: all three §9 conclusions survive
their first generalization test. Fairness is utility-neutral on both a sparse
and a dense retriever; its small side-effects run through diversity; diversity's
sign depends on task type. The one thing two rankers could not fix — the narrow
ILD manipulation range — is a property of the LaMP benchmark itself, and is the
honest boundary of what this project's data can say.

## 11. Update (2026-07-26): the flanT5Base 2x2 grid — the generator axis changes the story

**What ran**: the complete Small design replicated on flanT5Base (250M params,
~3.2x Small) — deterministic + MMR λ∈{0.15..1.0} + PL α∈{1,2,4,8} at N=10, all 7
LaMP tasks, **both** BM25 and Contriever — completing a full 2×2 grid (generator
size × retrieval paradigm). 168 new runs (343 total), 67,200 generation units,
571 min overnight, zero failures. λ=1.0 reproduced deterministic exactly on all
14 (ranker, task) pairs. Note: Base's utility-label dataset is its own filtered
query set (per the paper's design), so Small-vs-Base comparisons are
generator-level contrasts, not per-query pairs. Two operational notes for the
record: the first launch attempt (a scheduled task) was blocked and then killed
by an unnoticed charger disconnection (schtasks' default AC-power conditions +
critical-battery sleep — full timeline in the power-event log); and an HF cache
layout mismatch (`HF_HOME`-style `hub/` vs `PromptLM`'s `cache_dir` layout) was
found and fixed. Analysis: `experiments/analyze_mmr_sweep.py`, now looping
(generator, ranker) cells with a per-generator interaction re-fit.

**Finding 0 — the floor effect was real.** Deterministic-run EU on Base vs Small:
LaMP-1 0.72 vs 0.12 (6x), LaMP-3 0.67 vs 0.46, LaMP-6/7 ~+35%. The weak
generator genuinely had little headroom to express reranking effects.

**Generator-ROBUST findings (survived all four cells):**
- **PL adds ~nothing over MMR at matched diversity.** Per-task n-weighted deltas
  stay small and sign-mixed on Base (BM25: +0.010/−0.011/−0.031/−0.024;
  Contriever: +0.026/−0.002/−0.003/+0.022). The proposal's core claim — whatever
  fair reranking does for utility, explicit diversity does about as well —
  holds at both generator sizes and both retrieval paradigms. This is the
  project's most robust conclusion.

**Generator-DEPENDENT findings (the story changed at 250M):**
1. **Diversity's effect flipped sign.** Within-MMR EU~ILD (fairness held fixed):
   Small was negative/ns (BM25 −0.237 p=3e-05; Contriever −0.072 ns) → Base is
   **positive and significant on both rankers** (BM25 +0.126 p=0.012; Contriever
   +0.214 p=7e-05). The stronger generator *benefits* from diverse retrieved
   contexts where the weaker one was confused by them. §9-10's "diversity mildly
   hurts generation tasks" was a weak-generator artifact.
2. **A real fairness–utility tradeoff emerged.** EE-D's main effect on Base:
   +0.143 (p=6e-21) — less-fair (higher-disparity) rankings associate with
   higher utility; on Small this was ~0 (p=0.05). In our replication, the
   fairness-utility tradeoff the original paper argued against is
   generator-dependent: invisible at 77M, measurable at 250M. (Magnitude
   caution: the matched-diversity comparison shows the *net* PL-vs-MMR cost
   stays small; the regression-level effect partly reflects cross-task
   composition. Both readings are reported.)
3. **The EE-D×ILD interaction did not replicate on Base** (p=0.65 vs Small's
   p=0.0004 at matched n). The §10 "graduated lead" is hereby demoted: it is a
   Small-generator-specific structure, not a general law. The lead-grade framing
   §10 insisted on proved warranted.

**Where this leaves the research question** (final empirical position for the
report): (a) the one fully generator- and ranker-robust result is that
randomization-based fair reranking provides no utility benefit beyond its
diversity side-effect — diversity is the only active ingredient; (b) whether
that ingredient helps or hurts depends on the *generator's capacity*: harmful/
neutral at 77M, helpful at 250M; (c) fairness itself costs utility at 250M in
regression terms, though the matched-diversity net cost is small; (d) the
honest scaling question — does the tradeoff keep growing with generator size? —
is exactly what an XXL-class run would answer and is out of this project's
compute reach (labels exist for flanT5XXL; the model does not fit an 8GB GPU).
That is the report's future-work section.

## 12. CORRECTION (2026-07-27): `select_best_precision` dedup bug and revised findings

While preparing the report figures, a question about which tasks feed each
figure exposed a data-loading bug that had silently skewed every pooled
analysis in sections 9-11 that included PL rows. `analysis/loading.py::
select_best_precision` deduplicated PL runs by `(lamp_num, pl_alpha)` only -
a key from the era when the analysis had a single (generator, ranker) cell.
Because the `pl_alpha_sweep_n30` experiment re-ran Small/BM25's LaMP-1..3
(and part of 4) at `pl_samples=30`, the "keep highest-N" rule deleted the
N=10 PL runs of the OTHER THREE cells for those tasks. Effect: Base and
Small/Contriever pooled analyses had no PL rows for LaMP-1..3 while their
deterministic/MMR rows did include those tasks (a task-composition asymmetry),
and - because `normalize_query_rows` normalizes per query across all loaded
lists - even MMR rows' normalized values (and NaN patterns) were distorted.
Fixed by keying the dedup on `(generator_name, ranker, lamp_num, pl_alpha)`.
Recovered rows: 28,120 -> 32,424 query rows; 291 -> 336 macro rows.

**Findings that survive unchanged:**
- Fairness never improves utility: 0/24 paired det-vs-fair comparisons positive;
  none significant after Holm (Wilcoxon, `experiments/significance_tests.py` A).
- lambda=1.0 sanity, headroom/floor-effect numbers, lambda->ILD manipulation
  (macro-level, unaffected by the dedup).
- The narrow-ILD-range benchmark property.

**Findings REVISED by the fix:**
- "Diversity's effect flips sign with generator size" is WRONG. Corrected:
  diversity's harm ATTENUATES with generator size. Within-MMR pooled coef(ILD):
  Small/BM25 -0.258 (p=6e-06), Small/Contriever -0.096 (ns), Base/BM25 -0.077
  (ns), Base/Contriever -0.055 (ns). The generator difference is still formally
  significant (ILD x is_base +0.185, p=1.3e-04, n=17,864) but the Base-positive
  coefficients (+0.126/+0.214) in section 11 were artifacts of the missing-task
  composition.
- The Base fairness-utility tradeoff shrinks from +0.143 (p=6e-21) to +0.038
  (p=2.8e-04) in the per-generator M2; the pooled EE-D coefficient is +0.027
  (p=2.3e-04). A small tradeoff is real and significant, but "grows with
  generator size" is NOT significant (EE-D x is_base p=0.20).
- NEW finding the old data masked: at matched diversity, PL costs a small but
  significant amount vs MMR on Base (is_pl -0.021, p=0.006 BM25; -0.021,
  p=0.010 Contriever; ns on Small) - the randomization component itself has a
  measurable cost on the stronger generator.

**Revised final empirical position for the report:** (a) fair reranking never
improves RAG utility anywhere in the grid - the original paper's "in many cases
even outperform" does not replicate; (b) diversity, not fairness, is the active
ingredient, and its own effect is harmful at 77M fading to neutral at 250M;
(c) a small but statistically solid fairness-utility tradeoff exists (visible
in pooled regressions; individual settings' drops are uniformly negative but
not Holm-significant), plus a small significant PL-vs-MMR residual cost on
Base; (d) whether any of this grows at XXL scale remains the future-work
question. The report should cite this correction openly - it is a methods
lesson (composition bugs from partial-coverage dedup) as much as a result.

## 13. FULL-COVERAGE SCALE-UP AND STRUCTURAL CLEANUP (2026-08-03 to 2026-08-09)

The lecturer flagged that `nq=100` (used throughout sections 7-12 above) wasn't
enough statistical power. Both generators were extended to **every available
query** for all 7 LaMP tasks and both rankers - `query_offset=100` runs
covering everything past the original 100-query sample, generated via
`framework`'s existing resumable batch machinery, then merged with the
original samples into single directories per setting (`experiments/
merge_run_pairs.py`, new this round - see `CLAUDE.md`'s "Committed experiment
data" section for the mechanism and why a real merge, not just concatenating
pre-aggregated stats, was necessary for macro-level correctness).

Final per-task query counts actually used (confirmed via `LaMPDataset.
total_queries()` - **not symmetric between generators**, unlike the assumption
implicit in sections 7-12's flat `nq=100` design):

| LaMP task | flanT5Base | flanT5Small |
|---|---|---|
| 1 | 232 | 51 |
| 2 | 280 | 192 |
| 3 | 378 | 311 |
| 4 | 827 | 833 |
| 5 | 759 | 826 |
| 6 | 783 | 760 |
| 7 | 211 | 365 |

Along the way, `experiment_runs/` was also restructured from a flat listing of
500+ timestamped directories into `experiment_runs/{generator}/lamp{N}/
{ranker}/{run_id}/`, and a `select_best_precision`-adjacent legacy issue was
cleaned up: 15 flanT5Small+BM25+LaMP1-4 directories left over from the section
12 `pl_alpha_sweep_n30` investigation (which used `pl_samples=30` instead of
the project's standard 10) were removed, with 8 of them backfilled at
`pl_samples=10` first since no `pl_samples=10` baseline existed for those
cells at all.

**Final state as of 2026-08-09 (commit `43d12b3`):** both generators at full
query coverage, all 7 tasks, both rankers - 336 committed run directories (168
per generator), one merged directory per setting. Verified exhaustively before
declaring done: every directory's row count, unique qid count, and true
per-task total agree; the merged `macro_summary.json` in all 312
baseline+scaleup-merged directories was independently recomputed from raw
per-query rows and matched the stored value exactly (max deviation `0.0`
across every metric); zero `pl_samples` mismatches across all 112 PL cells.

**This section documents data completeness only - the findings in section 12
above are still the `nq=100` numbers.** `experiments/make_report_figures.py`
/ `significance_tests.py` / `task_sensitivity.py` have **not yet been re-run**
on the full-coverage data (no code changes needed to do so - the enlarged
pooled N flows through automatically via `list_run_dirs`'s recursive scan).
That re-run, and whatever revised findings/report follow from it, is the
project's next and (pending that re-run) final step.
