"""
Run the corrected RandMMR (pl_randmmr_fixed: rank 1 via pure PL, ranks 2..k via
deterministic MMR with a fresh lambda drawn per rank) for LaMP-1, all four
(generator, ranker) cells, matching the original pl_randmmr condition's
hyperparameters exactly (alpha=4, lambda~Uniform(0.8,1.0), seed=42) so it's a
direct like-for-like replacement -- but run natively at N=10 samples/query
(the precision the paper's comparison already uses) instead of N=20+truncate.

Does not touch or overwrite the original pl_randmmr runs; this is a new
method string, so it lands in its own experiment_runs/ directories.
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
                pl_alpha=4,
                pl_samples=10,
                pl_randmmr_lambda_low=0.8,
                pl_randmmr_lambda_high=1.0,
                seed=42,
            ),
            generation=GenerationConfig(generator_name=generator),
        ))

print(f"Running {len(configs)} pl_randmmr_fixed configs for LaMP-1...")
batch = BatchExperimentRunner(configs, batch_id="randmmr_fixed_lamp1", reuse_policy="smart")
result = batch.run_all()
print("Done.")
print(result)
