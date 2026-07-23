# Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation

This repo builds on the official code for the paper
[Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation](https://arxiv.org/abs/2409.11598),
extending it to test whether fairness's benefit to RAG performance actually comes from
the diversity it introduces. See **`docs/PROJECT_STATUS.md`** for the full project
history and open research question, **`docs/project_proposal.pdf`** /
**`docs/fair_rag_paper.pdf`** for the source documents, and **`CLAUDE.md`** for code
architecture.

<details>
<summary><b>Original paper abstract</b></summary>

Despite retrieval being a core component of RAG, much of the research in this area overlooks the extensive body of work on fair ranking, neglecting the importance of considering all stakeholders involved. This paper presents the first systematic evaluation of RAG systems integrated with fair rankings. We focus specifically on measuring the fair exposure of each relevant item across the rankings utilized by RAG systems (i.e., item-side fairness), aiming to promote equitable growth for relevant item providers. To gain a deep understanding of the relationship between item-fairness, ranking quality, and generation quality in the context of RAG, we analyze nine different RAG systems that incorporate fair rankings across seven distinct datasets. Our findings indicate that RAG systems with fair rankings can maintain a high level of generation quality and, in many cases, even outperform traditional RAG systems, despite the general trend of a tradeoff between ensuring fairness and maintaining system-effectiveness. We believe our insights lay the groundwork for responsible and equitable RAG systems and open new avenues for future research.
</details>

## Quick Start

Steps to go from a clean checkout to a running experiment.

### 1. Set up the environment
```
python -m venv .venv
.venv\Scripts\activate                          # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m ipykernel install --user --name fair-rag --display-name "Fair-RAG (.venv)"
```

### 2. Get the data
Pre-built LaMP utility-label test collections ship as zip archives under `data/` —
unzip the one(s) you need:
```
unzip "data/lamp_utility_labels_flanT5Small.zip" -d data/lamp_utility_labels_flanT5Small
```
If the zips aren't on your disk (e.g. a fresh clone — they're intentionally not tracked
in git, see [Data](#data) below), fetch them from the original paper's repo first:
```
git fetch upstream
git show upstream/main:data/lamp_utility_labels_flanT5Small.zip > data/lamp_utility_labels_flanT5Small.zip
```

### 3. Precompute retrieval
Every `(generator, ranker, LaMP task)` combination you plan to evaluate needs its
retrieval results precomputed once, ahead of time:
```
python retrieval/rank_profiles.py --ranker bm25 --generator_name flanT5Small --lamp_num 4
python retrieval/gold_retriever.py --generator_name flanT5Small --lamp_num 4    # oracle baseline, needed for EU normalization
```
(`--ranker` also accepts `splade`/`contriever`; repeat per LaMP task 1–7 as needed.)

### 4. Run an experiment
Open `fair_rag_experiment.ipynb` in Jupyter/VS Code, select the **`Fair-RAG (.venv)`**
kernel, edit the configuration toggle cell near the top, then run all cells. This drives
the `framework` package (`RunConfig`/`ExperimentRunner`/`BatchExperimentRunner`) —
see `CLAUDE.md` for the full architecture, config options, and rerank methods
(deterministic / MMR / Plackett-Luce / PL-MMR).

Results land in `experiment_runs/{run_id}/` (crash-safe, resumable — safe to re-run the
same notebook cell if interrupted). From there, `fair_rag_diversity_story.ipynb` and
`fair_rag_stats_exploration.ipynb` analyze completed runs.

## Data

We provide a filtered version of the LaMP dataset for fairness evaluation, along with item-level utility labels as detailed in the paper. The dataset includes three distinct utility-based test collections, each constructed based on a different generator model: Flan-T5 Small, Flan-T5 Base, and Flan-T5 XXL.

The data has been filtered and annotated based on the [LaMP dataset](https://github.com/LaMP-Benchmark/LaMP/tree/main/LaMP), which is available under the `CC-BY-NC-SA-4.0` license. All provided data can be found in the `data/` directory of this repository, as `lamp_utility_labels_{generator}.zip` archives — unzip before use (see [Quick Start](#quick-start) above).

These zip archives are **not tracked in this repository's git history** (only the unzipped `.json`/`.tsv` files they contain are used at runtime, and those are also untracked/gitignored, since they're large derived data). If you're setting this repo up fresh and don't already have the zips on disk, pull them from the original paper's repository, which is configured as the read-only `upstream` remote (see Quick Start step 2).

### Data Generation Pipeline (optional — only for regenerating labels from scratch)
1. Place the [LaMP dataset](https://github.com/LaMP-Benchmark/LaMP/tree/main/LaMP) under `data/lamp`
2. Run vanilla LMs and augmented LMs and save the inference results
    - e.g., `python utility_labels/inference.py --model_name flanT5XXL --lamp_num 4`
    - redirect your system output to a `.log` file
3. Evaluate the inference results
    - e.g., `python utility_labels/lamp_eval.py --model_name flanT5XXL --lamp_num 4`
4. (Optional) Get statistics of delta per query
    - e.g., `python utility_labels/analyze_delta.py --model_name flanT5XXL --lamp_num 4`
5. Make utility labels per item and generators
    - e.g., `python utility_labels/make_utility_dataset.py --model_name flanT5XXL --lamp_num 4`

## Legacy scripts

The original paper authors' single-setting CLI scripts (`experiment.py`,
`normalize_eu.py`) live under `legacy/` for reference/reproducibility of the paper's
exact original method, but are superseded by the `framework` package described above —
see `legacy/README.md`.

## Reference
If you find our research valuable, please consider citing it as follows:
```
@misc{kim2024fairragimpactfair,
      title={Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation}, 
      author={To Eun Kim and Fernando Diaz},
      year={2024},
      eprint={2409.11598},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2409.11598}, 
}
```
