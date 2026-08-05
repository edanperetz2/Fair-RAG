# experiment_runs/ — what's committed here and why

This directory is **gitignored by default** (`experiment_runs/` in `.gitignore`) —
run output is normally treated as regenerable local scratch, not repo content. The
run directories that *do* appear here on GitHub were deliberately force-added
(`git add -f`) because they underpin the report's final numbers and are worth
keeping alongside the code that produced them. This is a targeted, per-batch
exception, not a change to the default policy — new runs will keep being ignored
unless explicitly committed the same way.

**Browsing tip:** open **[`INDEX.md`](INDEX.md)** for a single sortable/searchable
table (one row per run, auto-generated from every `manifest.json` by
`experiments/generate_run_index.py`) instead of navigating folders by hand.
Regenerate it after adding new committed runs — it's derived data, not
hand-maintained.

## Structure: one directory per (generator, LaMP task, ranker, rerank setting)

```
experiment_runs/
├── README.md
├── INDEX.md
├── flanT5Small/
│   ├── lamp1/
│   │   ├── bm25/        (12 run dirs — one per rerank setting)
│   │   └── contriever/  (12 run dirs)
│   ├── lamp2/  ...
│   └── lamp7/
└── flanT5Base/
    ├── lamp1/
    │   ├── bm25/        (12 run dirs)
    │   └── contriever/  (12 run dirs)
    ├── lamp2/  ...
    └── lamp7/
```

Each leaf directory holds exactly **12 run directories** — 1 deterministic +
7 MMR λ + 4 PL α — each one covering that task's **full query range**, not a
partial sample. There is deliberately **no per-setting split** between an
original sample and a later scale-up: every setting's data was combined into
one directory with correctly recomputed aggregate stats (see "How full
coverage was reached" below) specifically so that no downstream code has to
know or care that some of this data was originally collected in two passes.

## Directory naming

Every run directory is named `{YYYYMMDD_HHMMSS}_{setting_id}`, where `setting_id`
(built by `framework.config.setting_id`) encodes every hyperparameter that defines
the experiment condition, e.g.:

```
20260805_125323_lamp7_user__flanT5Base__bm25__pl_a1_s10__k5__nqall__seed42
└─ timestamp ─┘ └lamp──┘ └gen───────┘ └ranker┘ └rerank──┘ └k┘ └nq───┘└seed┘
```

- `lamp{N}_user` — LaMP task N, "user" split
- generator name, ranker (`bm25` / `contriever`) — redundant with the folder
  path, kept in the name so a run directory is self-describing even out of
  context (e.g. linked from a notebook cell)
- rerank setting: `det`, `mmr_l{λ×100}`, or `pl_a{α}_s{pl_samples}`
- `k{top_k}` — documents passed to the LLM
- `nqall` — every completed run now covers the task's full query range, so
  this is constant across all committed runs (no `nq100`/`off100` split
  remains — see below)
- `seed{N}` — PL/MMR reproducibility seed

## How full coverage was reached (for provenance, not something you need to redo)

Query selection is a **deterministic file-order prefix** (no random sampling).
Both generators' data were originally collected as two passes per setting — a
100-query baseline, then a `query_offset=100` pass covering every remaining
query — verified non-overlapping and jointly exhaustive (zero qid intersection,
combined count == task's true total) before being **merged into one directory**
per setting via `experiments/merge_run_pairs.py`. That merge recomputes
`summary.json`/`macro_summary.json` from the combined raw data (via
`ArtifactStore.write_summary`/`write_macro_summary`) rather than concatenating
the two passes' pre-aggregated stats — necessary because a plain average of two
unequally-sized partial means is not the same as the true combined mean (this
was measured directly on real data: differed by ~0.006 EU on one spot-checked
setting). The original two-pass directories were deleted only after this
verification; recoverable from git history if ever needed.

flanT5Small additionally had 15 directories (LaMP 1-4, BM25 only) left over
from an earlier `select_best_precision` dedup-bug investigation that used
`pl_samples=30` instead of the project's standard `pl_samples=10` — these were
deleted and, where no `pl_samples=10` baseline existed at all (8 of the 15
cells), regenerated at `pl_samples=10` before the full-coverage scale-up ran.

## Per-task total query counts (same for both generators — shared dataset)

| LaMP task | total queries |
|---|---|
| 1 | 232 |
| 2 | 280 |
| 3 | 378 |
| 4 | 827 |
| 5 | 759 |
| 6 | 783 |
| 7 | 211 |

## Finding a specific run

Prefer **[`INDEX.md`](INDEX.md)** (sortable/searchable table) or just navigate
`{generator}/lamp{N}/{ranker}/` directly — at most 12 entries in any folder.

Each run directory's `manifest.json` records the exact config (a merged
directory's manifest also records `merged_from` with the two source run IDs
for provenance); `query_summary.jsonl` has one row per completed query (qid +
metrics); `summary.json`/`macro_summary.json` are the run's aggregate stats,
computed over the full query range.
