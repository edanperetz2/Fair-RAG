# experiment_runs/ — what's committed here and why

This directory is **gitignored by default** (`experiment_runs/` in `.gitignore`) —
run output is normally treated as regenerable local scratch, not repo content. The
run directories that *do* appear here on GitHub were deliberately force-added
(`git add -f`) because they underpin the report's final numbers and are worth
keeping alongside the code that produced them. This is a targeted, per-batch
exception, not a change to the default policy — new runs will keep being ignored
unless explicitly committed the same way.

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

## Why every setting has (at least) two directories, not one

Each (task, ranker, rerank-setting) combination was run in two separate passes,
each its own directory with its own `setting_id`:

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
**2N directories** on disk/GitHub. Analysis code (`framework.cross_run_analysis`,
`analysis.loading`) pools both directories' query rows together automatically —
there is no need to merge them manually.

## Per-task total query counts (flanT5Base, "user" split)

| LaMP task | total queries | original (nq=100) | scale-up (offset=100) |
|---|---|---|---|
| 1 | 232 | 100 | 132 |
| 2 | 280 | 100 | 180 |
| 3 | 378 | 100 | 278 |
| 4 | 827 | 100 | 727 |
| 5 | 759 | 100 | 659 |
| 6 | 783 | 100 | 683 |
| 7 | 211 | 100 | 111 |

Tasks committed so far: see `docs/PROJECT_STATUS.md` for the current status —
this README describes the convention, not a live progress tracker.

## Finding a specific run

Filter by substring rather than browsing the flat, timestamp-sorted listing:
- `lamp{N}_user__` — one task
- `__bm25__` / `__contriever__` — one ranker
- `__nq100__` vs `__nqall__off100__` — original vs scale-up pass
- `__det__` / `__mmr_l{...}__` / `__pl_a{...}_s{...}__` — rerank setting

Each run directory's `manifest.json` records the exact config; `query_summary.jsonl`
has one row per completed query (qid + metrics); `summary.json`/`macro_summary.json`
are the run's aggregate stats.
