"""
Driver for the scoped research-question experiment (see docs/PROJECT_STATUS.md
section 7 for the original run this generalizes from). Builds a batch of
RunConfigs across LaMP tasks / rerank settings / generators / rankers and hands
them to BatchExperimentRunner, exactly like fair_rag_experiment.ipynb's toggle
cell does - this script just makes the config-building reusable and scriptable
for the wider follow-up sweeps (alpha sweep, generator comparison, MMR-lambda
sweep, etc.) without hand-editing the notebook each time.

reuse_policy="smart" means already-completed settings (matched by setting_id)
are skipped, not overwritten - safe to rerun with an overlapping config list.

Usage examples:
    python scripts/scoped_experiment.py --smoke-test
    python scripts/scoped_experiment.py --alphas 1 2 4 8 --batch-id alpha_sweep_full
    python scripts/scoped_experiment.py --generators flanT5Small flanT5Base --batch-id generator_compare
    python scripts/scoped_experiment.py --mmr-lambdas 0.25 0.5 0.65 0.75 0.9 --no-pl --no-deterministic --batch-id mmr_lambda_sweep
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from framework import (
    RunConfig, DatasetConfig, RetrievalConfig, RerankConfig, GenerationConfig,
    MetricsConfig, CheckpointConfig, BatchExperimentRunner, setting_id,
)

DEFAULT_LAMP_TASKS = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_PL_SAMPLES = 10
DEFAULT_NUM_QUERIES = 100
DEFAULT_SEED = 42


def build_cfgs(
    lamp_nums,
    num_queries,
    generators,
    rankers,
    top_k=5,
    seed=DEFAULT_SEED,
    pl_samples=DEFAULT_PL_SAMPLES,
    alphas=(2, 8),
    mmr_lambdas=(0.55,),
    include_deterministic=True,
    include_pl=True,
    include_mmr=True,
    query_offset=0,
):
    cfgs = []
    for lamp_num in lamp_nums:
        for generator_name in generators:
            for ranker in rankers:
                base = dict(
                    dataset=DatasetConfig(dataset_type="lamp", lamp_num=lamp_num, lamp_split_type="user", num_queries=num_queries, query_offset=query_offset),
                    retrieval=RetrievalConfig(ranker=ranker, top_k=top_k),
                    generation=GenerationConfig(generator_name=generator_name, multi_gpu=False),
                    metrics=MetricsConfig(compute_ee=True, compute_eu=True, compute_diversity=True),
                    checkpoint=CheckpointConfig(flush_every=1, report_every_queries=20),
                )
                settings = []
                if include_deterministic:
                    settings.append(RerankConfig(method="deterministic", seed=seed))
                if include_mmr:
                    for lam in mmr_lambdas:
                        settings.append(RerankConfig(method="mmr", mmr_lambda=lam, seed=seed))
                if include_pl:
                    for alpha in alphas:
                        settings.append(RerankConfig(method="pl", pl_alpha=alpha, pl_samples=pl_samples, seed=seed))
                for rerank in settings:
                    cfgs.append(RunConfig(rerank=rerank, **base))
    return cfgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--lamp-tasks", type=int, nargs="+", default=DEFAULT_LAMP_TASKS)
    parser.add_argument("--num-queries", type=str, default=str(DEFAULT_NUM_QUERIES),
                         help="int, or 'all' to process every available query (respects --query-offset)")
    parser.add_argument("--query-offset", type=int, default=0,
                         help="skip this many queries (in dataset file order) before starting")
    parser.add_argument("--generators", nargs="+", default=["flanT5Small"])
    parser.add_argument("--rankers", nargs="+", default=["bm25"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pl-samples", type=int, default=DEFAULT_PL_SAMPLES)
    parser.add_argument("--alphas", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--mmr-lambdas", type=float, nargs="+", default=[0.55])
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--no-pl", action="store_true")
    parser.add_argument("--no-mmr", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        cfgs = build_cfgs([1], num_queries=2, generators=["flanT5Small"], rankers=["bm25"])
        batch_id = args.batch_id or "smoke_test_scoped_run"
    else:
        num_queries = None if args.num_queries.lower() == "all" else int(args.num_queries)
        cfgs = build_cfgs(
            args.lamp_tasks,
            num_queries=num_queries,
            generators=args.generators,
            rankers=args.rankers,
            top_k=args.top_k,
            seed=args.seed,
            pl_samples=args.pl_samples,
            alphas=args.alphas,
            mmr_lambdas=args.mmr_lambdas,
            include_deterministic=not args.no_deterministic,
            include_pl=not args.no_pl,
            include_mmr=not args.no_mmr,
            query_offset=args.query_offset,
        )
        batch_id = args.batch_id or "scoped_experiment"

    print(f"Built {len(cfgs)} configs. Settings:")
    for cfg in cfgs:
        print("  -", setting_id(cfg))

    t0 = time.time()
    runner = BatchExperimentRunner(cfgs, batch_id=batch_id, reuse_policy="smart")
    summary = runner.run_all()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print("Run dirs:")
    for d in summary["run_dirs"]:
        print(" -", d)
