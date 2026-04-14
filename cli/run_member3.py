"""
Member 3 — Entity Matching Orchestrator
========================================
Value similarity (Jaccard + TF-IDF + SBERT) + Collective ER (neighbor similarity)
-> match_results_*.csv

Usage:
    # All real datasets
    python cli/run_member3.py

    # One dataset
    python cli/run_member3.py --dataset abt_buy
    python cli/run_member3.py --dataset amazon_google

    # Mock data (output/mock/<dataset>/)
    python cli/run_member3.py --mock

    # Limit number of pairs (for quick testing)
    python cli/run_member3.py --dataset abt_buy --limit 100

    # Skip datasets with missing candidate_pairs.csv
    python cli/run_member3.py --skip-missing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_ingestion import DATASET_REGISTRY
from pipeline.matching import run_matching

OUTPUT_DIR = "output"
MOCK_ROOT = os.path.join(OUTPUT_DIR, "mock")


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run_all(
    datasets: list[str],
    use_mock: bool,
    limit: int | None,
    *,
    candidate_strategy: str,
    progressive_stages: str | None,
    time_limit_minutes: float | None,
    resume: bool,
    chunk_size: int,
    online_clustering: bool,
    clustering_algorithm: str,
    online_cluster_every_n_chunks: int,
    skip_missing: bool,
) -> None:
    """Run Member 3 matching for each dataset."""
    results = {}

    for ds_key in datasets:
        banner(f"DATASET: {ds_key.upper()}")

        prefix = os.path.join(MOCK_ROOT, ds_key) if use_mock else os.path.join(OUTPUT_DIR, ds_key)
        pairs_path = os.path.join(prefix, "candidate_pairs.csv")

        if skip_missing and not os.path.isfile(pairs_path):
            print(f"  [SKIP] Missing: {pairs_path}")
            continue

        result = run_matching(
            ds_key,
            mock=use_mock,
            limit=limit,
            candidate_strategy=candidate_strategy,
            progressive_stages=progressive_stages,
            time_limit_seconds=(None if time_limit_minutes is None else time_limit_minutes * 60.0),
            resume=resume,
            chunk_size=chunk_size,
            online_clustering=online_clustering,
            clustering_algorithm=clustering_algorithm,
            online_cluster_every_n_chunks=online_cluster_every_n_chunks,
        )
        if result is not None:
            results[ds_key] = result

    banner("MEMBER 3 COMPLETE — SUMMARY")
    print(f"\n  {'Dataset':<20} {'Status'}")
    print(f"  {'-'*20} {'-'*20}")
    for ds_key in datasets:
        if ds_key not in results:
            status = "SKIPPED / FAILED"
        else:
            status = "OK" if results[ds_key].get("completed") else "PARTIAL"
        print(f"  {ds_key:<20} {status}")

    print("\n  Output files per dataset:")
    for ds_key in results:
        out = os.path.join(MOCK_ROOT if use_mock else OUTPUT_DIR, ds_key)
        processed = results[ds_key].get("processed_rows")
        total = results[ds_key].get("total_rows")
        print(f"    {ds_key:<20} -> {out}/match_results_*.csv  ({processed}/{total} rows)")


def main() -> None:
    valid = list(DATASET_REGISTRY.keys()) + ["all"]
    parser = argparse.ArgumentParser(
        description="Member 3 — Entity matching -> match_results_*.csv"
    )
    parser.add_argument(
        "--dataset",
        choices=valid,
        default="all",
        help="Dataset to process (default: all)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use output/mock/<dataset>/ instead of output/<dataset>/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of candidate pairs processed (for quick testing)",
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=["v0", "cw_semantic_predictive"],
        default="v0",
        help="Candidate ordering strategy consumed by matching (default: v0)",
    )
    parser.add_argument(
        "--progressive-stages",
        default=None,
        help="Comma-separated checkpoints for progressive evaluation, e.g. 1000,5000,10000",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets whose candidate_pairs.csv is missing instead of failing",
    )
    parser.add_argument(
        "--time-limit-minutes",
        type=float,
        default=None,
        help="Stop Member 3 after this many minutes and keep resumable partial outputs",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Number of candidate pairs processed per chunk before checkpointing (default: 250)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing Member 3 cache/checkpoints and restart matching from zero",
    )
    parser.add_argument(
        "--online-clustering",
        action="store_true",
        help="Update incremental clustering outputs during Member 3 checkpoints",
    )
    parser.add_argument(
        "--clustering-algorithm",
        choices=["connected_components", "center"],
        default="connected_components",
        help="Algorithm used for incremental clustering checkpoints (default: connected_components)",
    )
    parser.add_argument(
        "--online-cluster-every-n-chunks",
        type=int,
        default=1,
        help="Refresh online clustering every N matching chunks instead of every chunk (default: 1)",
    )
    args = parser.parse_args()

    datasets = (
        list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    )

    run_all(
        datasets=datasets,
        use_mock=args.mock,
        limit=args.limit,
        candidate_strategy=args.candidate_strategy,
        progressive_stages=args.progressive_stages,
        time_limit_minutes=args.time_limit_minutes,
        resume=not args.no_resume,
        chunk_size=args.chunk_size,
        online_clustering=args.online_clustering,
        clustering_algorithm=args.clustering_algorithm,
        online_cluster_every_n_chunks=args.online_cluster_every_n_chunks,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    main()
