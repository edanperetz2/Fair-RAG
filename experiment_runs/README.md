# experiment_runs/ — what's committed here and why

This directory is **gitignored by default** (`experiment_runs/` in `.gitignore`) —
run output is normally treated as regenerable local scratch, not repo content. The
run directories that *do* appear here on GitHub were deliberately force-added
(`git add -f`) because they underpin the report's final numbers and are worth
keeping alongside the code that produced them. This is a targeted, per-batch
exception, not a change to the default policy — new runs will keep being ignored
unless explicitly committed the same way.

**Browsing tip:** with 500+ flat run directories, don't scroll GitHub's file list —
open **[`INDEX.md`](INDEX.md)** instead. It's a single sortable/searchable table
(one row per run directory, auto-generated from every `manifest.json` by
`experiments/generate_run_index.py`) with columns for LaMP task, generator, ranker,
setting, query count/offset, and status. Regenerate it after adding new committed
runs — it's derived data, not hand-maintained.

## Directory naming

Every run directory is named `{YYYYMMDD_HHMMSS}_{setting_id}`, where `setting_id`
(built by `framework.config.setting_id`) encodes every hyperparameter that defines
the experiment condition, e.g.:

```
20260802_164136_lamp4_user__flanT5Base__bm25__pl_a1_s10__k5__nqall__off100__seed42
└─ timestamp ─┘ └lamp──┘ └gen───────┘ └ranker┘ └rerank──┘ └k┘ └nq──────────┘└seed┘
```

- `lamp{N}_user` — LaMP task N, "user" split
- generator name, ranker (`bm25` / `contriever`)
- rerank setting: `det`, `mmr_l{λ×100}`, or `pl_a{α}_s{pl_samples}`
- `k{top_k}` — documents passed to the LLM
- `nq{N}` or `nqall` — query cap (`nqall` = every remaining query)
- `__off{N}` — **only present when nonzero**: skip the first N queries (in
  dataset file order) before applying the query cap
- `seed{N}` — PL/MMR reproducibility seed

## Two generators, two different coverage stories

Every run directory is one (LaMP task, generator, ranker, rerank-setting) cell —
24 settings per (task, generator): 1 deterministic + 7 MMR λ + 4 PL α (× 10
samples each). The two generators were **not** scaled up equally:

- **flanT5Base** — scaled from the original nq=100 sample to full task coverage
  (see "Why flanT5Base has two directories per setting" below). This is the
  generator the report's headline numbers use.
- **flanT5Small** — stayed at the original **nq=100 sample only, one directory
  per setting** (24 per task, 168 total across 7 tasks). It was part of the
  original 2×2×2 grid ({flanT5Small, flanT5Base} × {BM25, Contriever} × 7 tasks ×
  12 rerank settings, 343 runs) but was out of scope for the lecturer-feedback-
  driven scale-up, which targeted flanT5Base only. Treat flanT5Small as
  background/robustness data at n=100, not as having the same statistical power
  as flanT5Base's full-coverage numbers.

**One flanT5Small wrinkle, not a bug:** 7 of the flanT5Small+BM25 PL runs across
LaMP 1-4 have **both** a `pl_samples=10` and a `pl_samples=30` directory for the
same α (e.g. two `lamp1_user__flanT5Small__bm25__pl_a2_s{10,30}...` dirs). These
are leftovers from investigating the `select_best_precision` dedup bug (see
`docs/PROJECT_STATUS.md` section 12) — the `s=30` re-runs were the ones that
exposed the bug, and both versions were kept rather than deleting data.
`analysis.loading.select_best_precision` already dedups this correctly (keeps
`max(pl_samples)` per (generator, ranker, lamp_num, pl_alpha) cell post-fix), so
having both on disk/GitHub does not double-count anything downstream — it's just
visible in `INDEX.md`/the directory listing as two rows instead of one.

## Why flanT5Base has two directories per setting

Each (task, ranker, rerank-setting) combination for flanT5Base was run in two
separate passes, each its own directory with its own `setting_id`:

| Pass | num_queries | query_offset | Coverage |
|---|---|---|---|
| Original | 100 | 0 (absent from name) | queries 1-100 |
| Scale-up | all remaining | 100 | queries 101-N (task's full size) |

Query selection is a **deterministic file-order prefix** (no random sampling), so
the two passes are guaranteed non-overlapping and jointly exhaustive: querying
qids across both directories for the same setting reproduces every available
query for that task exactly once. This was verified directly (zero qid
intersection) before committing each task's data — see `docs/PROJECT_STATUS.md`
for the verification log.

So for N settings in a task (2 rankers × 12 rerank configs = 24 per task), expect
**2N directories** for flanT5Base on disk/GitHub (48/task), vs **N directories**
for flanT5Small (24/task, +7 extra `s30` dirs as noted above). Analysis code
(`framework.cross_run_analysis`, `analysis.loading`) pools directories per
setting together automatically — there is no need to merge them manually.

## Per-task total query counts

| LaMP task | total queries | flanT5Base: original (nq=100) | flanT5Base: scale-up (offset=100) | flanT5Small |
|---|---|---|---|---|
| 1 | 232 | 100 | 132 | 100 only |
| 2 | 280 | 100 | 180 | 100 only |
| 3 | 378 | 100 | 278 | 100 only |
| 4 | 827 | 100 | 727 | 100 only |
| 5 | 759 | 100 | 659 | 100 only |
| 6 | 783 | 100 | 683 | 100 only |
| 7 | 211 | 100 | 111 | 100 only |

All tasks are complete for both generators as of this commit — see
`docs/PROJECT_STATUS.md` for the fuller experiment history.

## Finding a specific run

Prefer **[`INDEX.md`](INDEX.md)** (a single table, sortable/searchable) over
browsing the flat, timestamp-sorted directory listing. If filtering by folder
name substring instead:
- `lamp{N}_user__` — one task
- `flanT5Small__` / `flanT5Base__` — one generator
- `__bm25__` / `__contriever__` — one ranker
- `__nq100__` vs `__nqall__off100__` — original vs scale-up pass (flanT5Base only)
- `__det__` / `__mmr_l{...}__` / `__pl_a{...}_s{...}__` — rerank setting

Each run directory's `manifest.json` records the exact config; `query_summary.jsonl`
has one row per completed query (qid + metrics); `summary.json`/`macro_summary.json`
are the run's aggregate stats.
