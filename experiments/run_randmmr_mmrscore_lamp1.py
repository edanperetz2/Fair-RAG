"""
Real generation runs (LLM calls, EU/ILD computed) for the pl_randmmr_mmrscore
variant, LaMP-1 only, all 4 (generator, ranker) cells. Table 2's EU column is
actually N=10 (rq3_randmmr_vs_pl_n10.py truncates the real pl_randmmr_a4_l08to10_s20
runs to the first 10 of 20 generated samples by sample_idx) -- generating
directly at pl_samples=10 here produces byte-identical first-10 samples (each
sample's RNG draws are self-contained per sample_idx, independent of the total
count), just without the wasted N=11..20 generation. Same alpha=4,
lambda~Uniform(0.8,1.0), top_k=5, seed=42 as the real runs.
"""
import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import RunConfig, ExperimentRunner
from framework.config import DatasetConfig, RetrievalConfig, RerankConfig, GenerationConfig

CELLS = [
    ("flanT5Base", "bm25"),
    ("flanT5Base", "contriever"),
    ("flanT5Small", "bm25"),
    ("flanT5Small", "contriever"),
]

for generator, ranker in CELLS:
    print(f"\n=== lamp1 {generator}/{ranker} pl_randmmr_mmrscore a=4 l=0.8to1.0 s=10 ===", flush=True)
    cfg = RunConfig(
        dataset=DatasetConfig(lamp_num=1, lamp_split_type="user", num_queries=None),
        retrieval=RetrievalConfig(ranker=ranker, top_k=5),
        rerank=RerankConfig(
            method="pl_randmmr_mmrscore",
            pl_alpha=4,
            pl_randmmr_lambda_low=0.8,
            pl_randmmr_lambda_high=1.0,
            pl_samples=10,
            seed=42,
        ),
        generation=GenerationConfig(generator_name=generator),
        resume=True,
    )
    runner = ExperimentRunner(cfg)
    store = runner.run()
    print(f"Done: {store.run_dir}")

print("\nAll 4 LaMP-1 cells complete.")
