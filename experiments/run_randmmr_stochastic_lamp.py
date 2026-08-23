"""
Run pl_randmmr_stochastic (alpha=4, lambda~Uniform(0.7,1.0), tunable tau, N=10,
seed=42) for a given LaMP task, all 4 (generator, ranker) cells.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import RunConfig, BatchExperimentRunner
from framework.config import DatasetConfig, RetrievalConfig, RerankConfig, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--lamp_num", type=int, required=True)
parser.add_argument("--tau", type=float, required=True)
parser.add_argument("--alpha", type=float, default=4.0)
parser.add_argument("--lambda_low", type=float, default=0.7)
parser.add_argument("--lambda_high", type=float, default=1.0)
args = parser.parse_args()

GENERATORS = ["flanT5Small", "flanT5Base"]
RANKERS = ["bm25", "contriever"]

configs = []
for generator in GENERATORS:
    for ranker in RANKERS:
        configs.append(RunConfig(
            dataset=DatasetConfig(lamp_num=args.lamp_num, lamp_split_type="user", num_queries=None),
            retrieval=RetrievalConfig(ranker=ranker, top_k=5),
            rerank=RerankConfig(
                method="pl_randmmr_stochastic",
                pl_alpha=args.alpha,
                pl_samples=10,
                pl_randmmr_lambda_low=args.lambda_low,
                pl_randmmr_lambda_high=args.lambda_high,
                pl_randmmr_tau=args.tau,
                seed=42,
            ),
            generation=GenerationConfig(generator_name=generator),
        ))

batch_id = f"randmmr_stoch_lamp{args.lamp_num}_a{int(args.alpha)}_tau{str(args.tau).replace('.', '')}"
print(f"Running {len(configs)} pl_randmmr_stochastic (alpha={args.alpha}, tau={args.tau}) configs for LaMP-{args.lamp_num}...")
batch = BatchExperimentRunner(configs, batch_id=batch_id, reuse_policy="smart")
result = batch.run_all()
print("Done.")
print(result)
