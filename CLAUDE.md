# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Official code for *"Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation"* (arXiv:2409.11598). It evaluates RAG systems built on the LaMP benchmark under different (stochastic and deterministic) rankers, measuring the tradeoff between item-side fairness (Expected Exposure) and generation quality (Expected Utility), plus intra-list diversity.

All active work goes through **`framework/`** — the config-driven, resumable experimentation framework (JSONL-based artifacts). Data/retrieval precomputation still goes through the original paper's top-level scripts (`retrieval/rank_profiles.py`, `retrieval/gold_retriever.py`, `utility_labels/*.py`), which `framework/` consumes as inputs.

The paper authors' original single-setting driver scripts (`experiment.py`, `normalize_eu.py`) live under **`legacy/`** — fully superseded by `framework.ExperimentRunner`/`BatchExperimentRunner`, kept only for reference (see `legacy/README.md`). Don't build new work on them. `legacy/` also holds two recovered-but-never-integrated artifacts from an earlier, abandoned approach (`mlx_generator_reference.py`, `trec_rag_2024_dataset_overview.ipynb`) — see `docs/PROJECT_STATUS.md` for that history.

Day-to-day work happens in three notebooks at the repo root, following a strict separation:

- **`fair_rag_experiment.ipynb`** — pure infrastructure. Only responsible for building `RunConfig`s and calling `ExperimentRunner`/`BatchExperimentRunner`. Should never contain post-hoc analysis of completed runs — if you find yourself adding a cell that reads `experiment_runs/` to compute/plot something rather than to run something, it belongs in one of the other two notebooks instead.
- **`fair_rag_diversity_story.ipynb`** / **`fair_rag_stats_exploration.ipynb`** — the "theses": read-only research narratives over already-completed runs. Must never call `ExperimentRunner`/`BatchExperimentRunner` — if a hypothesis needs a specific setting that isn't run yet, add it to `fair_rag_experiment.ipynb`, not here.
- **`analysis/`** — shared plotting/normalization/binning/labeling/run-discovery helpers used by the two thesis notebooks (and, sparingly, by `fair_rag_experiment.ipynb`'s own "quick look at what I just ran" plots). Builds on top of `framework/cross_run_analysis.py` rather than duplicating it. See its own section below. Before adding a new helper function to a notebook, check whether it's generic enough to belong here instead — this package exists specifically because near-identical label-formatting/normalization/binning functions were independently reimplemented 5-8 times across the three notebooks before being consolidated (see `docs/PROJECT_STATUS.md` for that history).

This repo has a single branch, `main`.

## Environment

- Python venv at `.venv/` (Python 3.10) — activate with `.venv\Scripts\activate` on Windows.
- Dependencies: `pip install -r requirements.txt` (torch, transformers, sentence-transformers, langchain, faiss-cpu, rouge, evaluate, sparsembed, python-dotenv, accelerate, plus matplotlib/scipy/scikit-learn/ipykernel for the analysis notebooks). Registered as a Jupyter kernel named `fair-rag` (`Fair-RAG (.venv)`) via `python -m ipykernel install --user --name fair-rag --display-name "Fair-RAG (.venv)"` — select that kernel when opening the notebooks. Note: installing the full `jupyter`/`jupyterlab` meta-package can fail here with a Windows long-path error (this project sits under a deep path) — `ipykernel` alone is sufficient to run notebooks from an existing Jupyter/VS Code frontend.
- No test suite, linter, or CI config exists in this repo — there is nothing to run for "tests"/"lint" beyond executing the scripts/notebooks themselves.
- **GPU/CUDA**: `pip install -r requirements.txt` installs plain `torch==2.1.2`, which resolves to a CPU-only wheel on Windows unless you point pip at PyTorch's CUDA index. To use an NVIDIA GPU, after the normal install run:
  `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121`
  (cu121 wheels work with any driver reporting CUDA 12.x, e.g. driver-reported "CUDA Version: 12.3" — cu121 is backward compatible with it; check your driver's CUDA version via `nvidia-smi`). Verify with `python -c "import torch; print(torch.cuda.is_available())"`. `torchvision` isn't actually imported anywhere in this codebase — it only comes along because it's commonly pinned next to `torch`; safe to skip if you want a minimal install.
- GPU is optional either way; `PromptLM` (`generator/lm.py`) auto-selects CUDA → MPS → CPU with no code changes needed. Multi-GPU inference goes through `accelerate` and `generator/lm_distributed_inference.py`.

## Data pipeline (one-time setup, in order)

1. Place the [LaMP dataset](https://github.com/LaMP-Benchmark/LaMP) under `data/lamp` (external download, CC-BY-NC-SA-4.0).
2. Generate utility labels per generator model (already-built label sets ship in `data/lamp_utility_labels_flanT5Small(.zip)` / `flanT5Base` / `flanT5XXL.zip` — unzip before use):
   - `python utility_labels/inference.py --model_name flanT5XXL --lamp_num 4` (redirect stdout to a log)
   - `python utility_labels/lamp_eval.py --model_name flanT5XXL --lamp_num 4`
   - `python utility_labels/analyze_delta.py --model_name flanT5XXL --lamp_num 4` (optional)
   - `python utility_labels/make_utility_dataset.py --model_name flanT5XXL --lamp_num 4`
3. Precompute deterministic retrieval (BM25 / Contriever / SPLADE), consumed later by `framework.retrieval.load_retrieval_results`:
   - `python retrieval/rank_profiles.py --ranker splade --generator_name flanT5XXL --lamp_num 4`
   - Oracle/gold retriever (needed for EU normalization baseline): `python retrieval/gold_retriever.py --generator_name flanT5XXL --lamp_num 4`
   - Results land in `retrieval/retrieval_results/{generator_name}/{ranker}/{lamp_num}.json`.

## Running experiments (current framework)

Everything is driven by `framework.config.RunConfig` (nested dataclasses: `DatasetConfig`, `RetrievalConfig`, `RerankConfig`, `GenerationConfig`, `MetricsConfig`, `CheckpointConfig`). Build a config, then either:

```python
from framework import RunConfig, ExperimentRunner, BatchExperimentRunner

runner = ExperimentRunner(cfg)      # single setting
store = runner.run()

batch_runner = BatchExperimentRunner(list_of_cfgs, batch_id="my_batch", reuse_policy="smart")
batch_output = batch_runner.run_all()   # sequential sweep across settings
```

`fair_rag_experiment.ipynb` is the canonical way this is invoked — it exposes toggles for dataset/ranker/rerank method and switches between `RUN_MODE = "single"` vs `"batch"`. Edit the toggle cell and re-run; in batch mode, `BATCH_REUSE_POLICY = "smart"` auto-skips already-completed settings and auto-resumes interrupted ones.

The legacy equivalent (single setting, no resume/reuse machinery, kept only for reference — see `legacy/README.md`) is:
```
python legacy/experiment.py --retriever_name splade --generator_name flanT5XXL --lamp_num 4 --alpha 2
python legacy/normalize_eu.py --retriever_name splade --generator_name flanT5XXL --lamp_num 4 --alpha 2
```

### `setting_id` — the canonical experiment key

`framework.config.setting_id(cfg)` encodes every hyperparameter that defines an experiment condition into one string (e.g. `lamp4_user__flanT5Small__bm25__pl_a1_s10__k5__nq50__seed42`). It is used as the run-reuse key and should stay stable — if you add a new config field that changes experiment semantics, it must be reflected in `setting_id`, or reuse/resume logic in `RunRegistry`/`BatchExperimentRunner` will silently treat different configs as identical.

### Rerank methods (`framework/reranking.py`)

- `"pl"` — Plackett-Luce stochastic sampling (Gumbel-max trick via `perturbation/plackettluce.py`); produces `pl_samples` independent `RetrievalList`s per query, temperature-controlled by `pl_alpha`.
- `"mmr"` — deterministic Maximal Marginal Relevance re-ranking (Jaccard-similarity diversity term), one list per query.
- `"pl_mmr"` — hybrid: PL-sampled rank 1, then sequential Gumbel-max sampling with a per-step diversity penalty for ranks 2..k.
- `"deterministic"` (anything else) — passthrough of the precomputed base ranking; used as the EE baseline.

### Metrics (`framework/metrics.py`)

- **EE-D / EE-R / EE-L** (Expected Exposure disparity/relevance/difference) — computed once per query across *all* ranked lists together via the vendored `expected_exposure/expeval.py` (modified from [diazf/expeval](https://github.com/diazf/expeval)). Requires materializing temporary TREC-format `trec_top_files/`/`trec_rel_files/` per query (see `utils.make_trec_top_file_for_single_qid` / `make_trec_rel_file_for_single_qid`) — these are cleaned up automatically unless `remove_temp=False`.
- **EU** (Expected Utility) — per-(qid, list_id) task metric from `utility_metrics/lamp_metrics.py`, selected by LaMP task number: accuracy (LaMP 1–2), MAE (LaMP 3, lower-is-better), ROUGE-L (LaMP 4–7).
- **Diversity** — ILD-Jaccard (1 − mean pairwise Jaccard similarity of profile text) and raw mean Jaccard, computed per list.

### Artifacts and resumability (`framework/artifacts.py`)

Each run writes to `experiment_runs/{run_id}/` as append-only JSONL (crash-safe: each write is flushed + fsynced):
- `manifest.json` — run metadata, atomically replaced on each update.
- `retrieval_lists.jsonl`, `ee_metrics.jsonl`, `llm_answers.jsonl`, `per_list_metrics.jsonl`, `query_summary.jsonl`, `progress_reports.jsonl`.
- `summary.json` / `macro_summary.json` — written once at the end of a run.

On resume (`cfg.resume=True`), `ExperimentRunner` reconstructs completed-work sets by scanning these JSONL files rather than trusting `manifest.json` counters, so partial/interrupted runs never redo completed units. `RunRegistry` (also in `artifacts.py`) matches configs against `experiment_runs/*/manifest.json` (via `comparable_config_dict`, which excludes `run_id`/checkpoint-only fields) to find completed/resumable runs for `BatchExperimentRunner`'s `reuse_policy` (`"smart"` skips completed + resumes interrupted, `"fresh"` never reuses).

### Cross-run analysis (`framework/cross_run_analysis.py`)

Loads and flattens `experiment_runs/*/manifest.json` + `macro_summary.json`/`query_summary.jsonl` into comparison rows (optionally as a pandas DataFrame): `list_run_dirs`, `build_macro_comparison_rows`, `build_query_metric_rows`, `maybe_to_dataframe`. This is the foundation the `analysis/` package and both thesis notebooks build on — don't duplicate it.

## Shared analysis toolkit (`analysis/`)

Generic, notebook-agnostic helpers for analyzing *already-completed* runs — not for running experiments (that's `framework/`'s job). Each module is single-purpose:

- **`analysis/loading.py`** — `find_completed_run_dirs(...)` (filters completed runs by dataset/generator/ranker/top_k/lamp_num/rerank_method against each run's manifest; note it does *not* special-case the `"gold"` ranker's deterministic-only convention — callers needing that filter it at the call site) and `existing_setting_ids(...)`; also `load_relevance_mapping(lamp_num, generator_name)` / `load_retrieval_scores(ranker, lamp_num, generator_name)` for measurement-input loading.
- **`analysis/normalization.py`** — `normalize_query_rows(df, group_cols=...)`: per-query max-normalization of EU/EE-R/ILD to comparable ranges (LaMP-3's MAE metric flipped to higher-is-better first); `safe_div` returns `None`/NaN on a zero/invalid denominator (a deliberate choice — never fabricates a value); `macro_from_normalized_query_rows(df)` aggregates normalized per-query rows to one row per run (to include a reference run like gold in the normalization *bounds* without it appearing in the output, pass it into `normalize_query_rows` and filter it out of the result afterward).
- **`analysis/binning.py`** — `pool_delta_by_bin(df, metric_col=..., bin_edges=..., bin_labels=..., baseline_method=... | baseline_filter=..., comparison_methods=..., group_by=...)`: the canonical "bin queries by a metric, report pooled utility delta vs. a baseline" function. Handles both "every method vs. one baseline" and "one method vs. a specifically-filtered baseline" (e.g. "MMR at exactly λ=0.65") by varying arguments — don't write a new one-off variant of this, extend the call site instead. Only handles *fixed* bin edges; adaptive/quantile-derived bin edges per group (see `fair_rag_diversity_story.ipynb`'s `build_ild_bin_analysis_v6`) are a genuinely different approach and intentionally not folded into this function.
- **`analysis/labels.py`** — `format_rerank_label(row, ranker_prefix=False)`: the one label formatter for deterministic/mmr/pl/pl_mmr settings.
- **`analysis/plotting.py`** — `plot_metric_scatter_panels(rows_df, panels=[(x,y,xlabel,ylabel), ...], label_fn=... | label_col=...)`: the canonical 3-panel (or N-panel) scatter plotter; `format_macro_table_for_display(df)` for display formatting (keeps columns raw/separate — no compact "recipe" string column).
- **`analysis/stats.py`** — `bootstrap_ci`, `assign_quantile_bins`, `summarize_binned_metric`, `middle_vs_tail_summary`, `add_quantile_trend`, `weighted_mean`. `bootstrap_ci`/`summarize_binned_metric` take `rounds`/`seed` as explicit parameters (no notebook-global config fallback) — always pass them explicitly if you need reproducibility across a session.

## Dataset abstraction (`framework/dataset.py`)

`DatasetHandler` is the abstract adapter interface (`iter_queries`, `relevance_mapping_path`, `get_aip_func`, `find_profiles_by_pids`, `get_metric_fn`, `total_queries`); `LaMPDataset` is the only implementation today, wrapping `data/lamp_handler.py::LaMPHandler` and pre-loading all profiles into memory for O(1) lookup during long runs. Add new datasets (e.g. TREC-RAG) by subclassing `DatasetHandler` and registering in `make_dataset()` — don't bypass the abstraction from `framework/runner.py`.

## Key conventions worth knowing before editing

- **`list_id` format** encodes method + sample index and is relied on elsewhere (resume matching, EE rebuilding): PL → `{qid}__pl_s{idx:03d}`, PL-MMR → `{qid}__pl_mmr_s{idx:03d}`, MMR → `{qid}__mmr`, deterministic → `{qid}__det`. Builder functions live in `framework/config.py`.
- Retrieval scores are normalized to `[1, 2]` before PL temperature exponentiation (`framework/retrieval.py::normalize_scores_for_pl`), except the `"gold"` oracle ranker, whose binary `{0,1}` scores are amplified to `{0,10}` so `pl_alpha` still has visible effect at low values.
- All module-level scripts under `framework/`, `generator/`, `retrieval/` prepend the repo root to `sys.path` via `ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))` — preserve this pattern if you add new modules that need root-relative imports (`data.*`, `utility_metrics.*`, `utils`, `expected_exposure.*`).
- `PromptLM` caches loaded HF models process-wide in `_MODEL_CACHE` keyed by `(model_name, sorted(model_kwargs))` — instantiating multiple `PromptLM`s with the same model reuses the same weights in memory.
