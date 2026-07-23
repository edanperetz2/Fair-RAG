# Legacy / reference material

Everything in this folder is **not part of the active pipeline**. It's kept purely as
reference. Nothing here is imported by `framework/` or the root notebooks. See
`docs/PROJECT_STATUS.md` for the full backstory.

- **`experiment.py`, `normalize_eu.py`** — the original paper authors' driver scripts
  for running one experiment setting and normalizing its Expected Utility. Fully
  superseded by `framework.ExperimentRunner` / `framework.BatchExperimentRunner`, which
  reimplement the same logic (EE, EU, diversity) with config management and crash-safe
  resumability. Still runnable from the repo root for reference
  (`python legacy/experiment.py --retriever_name bm25 --generator_name flanT5Small --lamp_num 4 --alpha 2`),
  but there's no reason to use these over the framework for new work.

- **`mlx_generator_reference.py`** — a working generator implementation for running
  quantized on-device models via Apple's MLX framework, recovered from Avi's abandoned
  `experiment-recreation` branch (never merged into the active `framework/` line). If
  MLX/local-model support is ever wanted, this is the starting point — it is **not**
  wired into `generator/lm.py` or `utils.models_info`, so it won't run as-is.

- **`trec_rag_2024_dataset_overview.ipynb`** — an exploration of the TREC RAG 2024
  benchmark, also recovered from the abandoned branch. Signals a possible future
  direction (extending evaluation beyond LaMP) that was never pursued further.
