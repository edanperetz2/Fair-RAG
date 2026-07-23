# Project Status & Handoff

Written for Idan Peretz, picking this project back up after project partner Avi Simkin
became unavailable. This document reconstructs what was tried, in what order, what's
still open, and how to continue — based on the project proposal, full git history
(including two abandoned branches recovered from local git objects), and a read-through
of all three notebooks and the current codebase.

For the architecture of the code as it exists today, see `CLAUDE.md` — this document
focuses on the *story* (what was tried, why, and what's left), not the code layout.

## 1. Where this started: the proposal

From `IDAN PERETZ 308560150 and AVI SIMKIN 312485816.pdf` (project proposal):

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
