"""
Member 3 — Entity Matching Orchestrator
========================================
Value similarity (Jaccard + TF-IDF + SBERT) + Collective ER (neighbor similarity)
-> match_results_*.csv

Usage:
    # All real datasets
    python run_member3.py

    # One dataset
    python run_member3.py --dataset abt_buy
    python run_member3.py --dataset amazon_google

    # Mock data (output/mock/<dataset>/)
    python run_member3.py --mock

    # Limit number of pairs (for quick testing)
    python run_member3.py --dataset abt_buy --limit 100

    # Skip datasets with missing candidate_pairs.csv
    python run_member3.py --skip-missing
"""

from __future__ import annotations

import argparse
import os

from data_ingestion import DATASET_REGISTRY
from matching import run_matching

OUTPUT_DIR = "output"
MOCK_ROOT = os.path.join(OUTPUT_DIR, "mock")


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run_all(
    datasets: list[str],
    use_mock: bool,
    limit: int | None,
    *,
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

        result = run_matching(ds_key, mock=use_mock, limit=limit)
        if result is not None:
            results[ds_key] = result

    banner("MEMBER 3 COMPLETE — SUMMARY")
    print(f"\n  {'Dataset':<20} {'Status'}")
    print(f"  {'-'*20} {'-'*20}")
    for ds_key in datasets:
        status = "OK" if ds_key in results else "SKIPPED / FAILED"
        print(f"  {ds_key:<20} {status}")

    print("\n  Output files per dataset:")
    for ds_key in results:
        out = os.path.join(MOCK_ROOT if use_mock else OUTPUT_DIR, ds_key)
        print(f"    {ds_key:<20} -> {out}/match_results_*.csv")


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
        "--skip-missing",
        action="store_true",
        help="Skip datasets whose candidate_pairs.csv is missing instead of failing",
    )
    args = parser.parse_args()

    datasets = (
        list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    )

    run_all(
        datasets=datasets,
        use_mock=args.mock,
        limit=args.limit,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    main()
