"""
Member 2 — Block Processing Orchestrator
========================================
Block purging + meta-blocking (Jaccard on shared block sets) -> candidate_pairs.csv

Usage:
    # All real datasets (reads output/<dataset>/blocks.csv)
    python cli/run_member2.py

    # Mock data (reads output/mock/<dataset>/blocks.csv)
    python cli/run_member2.py --mock

    # One dataset
    python cli/run_member2.py --dataset abt_buy
    python cli/run_member2.py --mock --dataset amazon_google

    # Tuning + metrics JSON for the report
    python cli/run_member2.py --max-block-size 500 --min-jaccard 0.15 --write-stats

    # Skip datasets with no blocks.csv (e.g. teammate missing local files)
    python cli/run_member2.py --skip-missing
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
from pipeline.block_processing import run_block_processing

OUTPUT_DIR = "output"
MOCK_ROOT = os.path.join(OUTPUT_DIR, "mock")


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run_all(
    datasets: list[str],
    use_mock: bool,
    max_block_size: int,
    min_jaccard: float,
    *,
    candidate_strategy: str,
    top_candidates_per_entity: int | None,
    skip_missing: bool,
    write_stats: bool,
) -> dict[str, dict]:
    """Run Member 2 for each dataset; returns dataset -> stats dict."""
    results: dict[str, dict] = {}

    for ds_key in datasets:
        cfg = DATASET_REGISTRY[ds_key]
        banner(f"DATASET: {ds_key.upper()}")

        if use_mock:
            blocks_path = os.path.join(MOCK_ROOT, ds_key, "blocks.csv")
            out_dir = os.path.join(MOCK_ROOT, ds_key)
        else:
            blocks_path = os.path.join(OUTPUT_DIR, ds_key, "blocks.csv")
            out_dir = os.path.join(OUTPUT_DIR, ds_key)
        cleaned_dir = out_dir if use_mock else os.path.join("data", "cleaned", ds_key)

        out_path = os.path.join(out_dir, "candidate_pairs.csv")
        stats_path = (
            os.path.join(out_dir, "member2_stats.json") if write_stats else None
        )

        if skip_missing and not os.path.isfile(blocks_path):
            print(f"  [SKIP] Missing blocks file: {blocks_path}")
            continue

        try:
            _df, stats = run_block_processing(
                blocks_path=blocks_path,
                output_path=out_path,
                source1=cfg["source1"],
                source2=cfg["source2"],
                max_block_size=max_block_size,
                min_jaccard=min_jaccard,
                candidate_strategy=candidate_strategy,
                source1_cleaned_path=os.path.join(cleaned_dir, "cleaned_source1.csv"),
                source2_cleaned_path=os.path.join(cleaned_dir, "cleaned_source2.csv"),
                top_candidates_per_entity=top_candidates_per_entity,
                write_stats_json=stats_path,
            )
        except FileNotFoundError as e:
            if skip_missing:
                print(f"  [SKIP] {e}")
                continue
            raise

        results[ds_key] = stats

    banner("MEMBER 2 SUMMARY")
    print(
        f"\n  {'Dataset':<18} {'Cartesian':>12} {'Candidates':>12} "
        f"{'Reduction':>10}"
    )
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10}")
    for ds_key, st in results.items():
        cart = st.get("cartesian_pairs", 0)
        cand = st.get("candidate_pairs", 0)
        red = st.get("reduction_vs_cartesian")
        red_s = f"{red:.4f}" if isinstance(red, (int, float)) else "n/a"
        print(f"  {ds_key:<18} {cart:>12,} {cand:>12,} {red_s:>10}")
    print("\n  -> Share candidate_pairs.csv with Member 3 (per dataset).")
    if write_stats and results:
        print("  -> Wrote member2_stats.json next to each candidate_pairs.csv.")
    if use_mock:
        print(f"  -> Ran on mocks under {MOCK_ROOT}/")

    return results


def main() -> None:
    valid = list(DATASET_REGISTRY.keys()) + ["all"]
    parser = argparse.ArgumentParser(
        description="Member 2 — Block purging + meta-blocking -> candidate_pairs.csv"
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
        help="Use output/mock/<dataset>/blocks.csv and write candidate_pairs there",
    )
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=1000,
        help="Drop blocks with more than this many rows (default: 1000)",
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=0.0,
        help="Meta-blocking: keep cross-source pairs with Jaccard(block sets) "
        ">= this (default: 0 = keep any pair sharing a post-purge block)",
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=["v0", "cw_semantic_predictive"],
        default="v0",
        help="Candidate generation strategy (default: v0 baseline)",
    )
    parser.add_argument(
        "--top-candidates-per-entity",
        type=int,
        default=None,
        help="Legacy option kept for CLI compatibility; ignored by the remaining strategies",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets whose blocks.csv is missing instead of failing",
    )
    parser.add_argument(
        "--write-stats",
        action="store_true",
        help="Write member2_stats.json (metrics) beside candidate_pairs.csv",
    )
    args = parser.parse_args()

    datasets = (
        list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    )

    run_all(
        datasets=datasets,
        use_mock=args.mock,
        max_block_size=args.max_block_size,
        min_jaccard=args.min_jaccard,
        candidate_strategy=args.candidate_strategy,
        top_candidates_per_entity=args.top_candidates_per_entity,
        skip_missing=args.skip_missing,
        write_stats=args.write_stats,
    )


if __name__ == "__main__":
    main()
