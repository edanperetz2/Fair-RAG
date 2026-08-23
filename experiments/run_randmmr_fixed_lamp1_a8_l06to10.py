"""
Run the corrected RandMMR (pl_randmmr_fixed) for LaMP-1, all four (generator, ranker)
cells, with an expanded lambda range (Uniform(0.6, 1.0) instead of (0.8, 1.0)) and
PL(alpha=8) for the rank-1 draw instead of alpha=4. N=10 samples/query, seed=42.

Does not touch or overwrite any prior pl_randmmr_fixed runs (different alpha/lambda
range means a distinct setting_id, so this lands in its own experiment_runs/ dirs).
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
                pl_alpha=8,
                pl_samples=10,
                pl_randmmr_lambda_low=0.6,
                pl_randmmr_lambda_high=1.0,
                seed=42,
            ),
            generation=GenerationConfig(generator_name=generator),
        ))

print(f"Running {len(configs)} pl_randmmr_fixed (alpha=8, lambda=0.6-1.0) configs for LaMP-1...")
batch = BatchExperimentRunner(configs, batch_id="randmmr_fixed_lamp1_a8_l06to10", reuse_policy="smart")
result = batch.run_all()
print("Done.")
print(result)
