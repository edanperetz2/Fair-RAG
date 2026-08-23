"""
Run the corrected RandMMR (pl_randmmr_fixed) for LaMP-1, all four (generator, ranker)
cells, with alpha=6 for the rank-1 PL draw, lambda~Uniform(0.8,1.0) (matching the
original a4 condition's lambda range), N=10 samples/query, seed=42.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import RunConfig, BatchExperimentRunner
from framework.config import DatasetConfig, RetrievalConfig, RerankConfig, GenerationConfig

GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]

configs = []
for generator in GENERATORS:
    for ranker in RANKERS:
        configs.append(RunConfig(
            dataset=DatasetConfig(lamp_num=1, lamp_split_type="user", num_queries=None),
            retrieval=RetrievalConfig(ranker=ranker, top_k=5),
            rerank=RerankConfig(
                method="pl_randmmr_fixed",
                pl_alpha=6,
                pl_samples=10,
                pl_randmmr_lambda_low=0.8,
                pl_randmmr_lambda_high=1.0,
                seed=42,
            ),
            generation=GenerationConfig(generator_name=generator),
        ))

print(f"Running {len(configs)} pl_randmmr_fixed (alpha=6, lambda=0.8-1.0) configs for LaMP-1...")
batch = BatchExperimentRunner(configs, batch_id="randmmr_fixed_lamp1_a6", reuse_policy="smart")
result = batch.run_all()
print("Done.")
print(result)
