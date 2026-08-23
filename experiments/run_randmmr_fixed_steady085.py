"""
Run pl_randmmr_fixed with a STEADY lambda=0.85 (lambda_low == lambda_high, so
every rank uses the exact same lambda rather than a per-rank random draw), for
alpha in {4, 8}, LaMP tasks 1-3, all 4 (generator, ranker) cells, N=10, seed=42.
All configs run in one process/batch so the generator model cache is reused.
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
LAMP_NUMS = [1, 2, 3]
ALPHAS = [4, 8]

configs = []
for lamp_num in LAMP_NUMS:
    for alpha in ALPHAS:
        for generator in GENERATORS:
            for ranker in RANKERS:
                configs.append(RunConfig(
                    dataset=DatasetConfig(lamp_num=lamp_num, lamp_split_type="user", num_queries=None),
                    retrieval=RetrievalConfig(ranker=ranker, top_k=5),
                    rerank=RerankConfig(
                        method="pl_randmmr_fixed",
                        pl_alpha=alpha,
                        pl_samples=10,
                        pl_randmmr_lambda_low=0.85,
                        pl_randmmr_lambda_high=0.85,
                        seed=42,
                    ),
                    generation=GenerationConfig(generator_name=generator),
                ))

print(f"Running {len(configs)} pl_randmmr_fixed (steady lambda=0.85) configs across LaMP 1-3...")
batch = BatchExperimentRunner(configs, batch_id="randmmr_fixed_steady085_lamp1to3", reuse_policy="smart")
result = batch.run_all()
print("Done.")
print(result)
