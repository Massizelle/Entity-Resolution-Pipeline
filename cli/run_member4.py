"""
Member 4 — Clustering + Entity Merging Orchestrator
===================================================
Reads match results from Member 3, groups matched entities into clusters
with connected components, then merges their attributes into canonical records.

Usage:
    # All real datasets
    python cli/run_member4.py

    # Single dataset
    python cli/run_member4.py --dataset abt_buy
    python cli/run_member4.py --dataset amazon_google

    # Use output/mock/<dataset>/ fixtures (no real data required)
    python cli/run_member4.py --mock
    python cli/run_member4.py --mock --dataset amazon_google

    # Choose which match file to cluster on (default: match_results_collective.csv)
    python cli/run_member4.py --match-file match_results_combined.csv

    # Skip datasets whose match file is missing instead of crashing
    python cli/run_member4.py --skip-missing
"""

from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.clustering import build_connected_components, merge_cluster_attributes
from pipeline.data_ingestion import DATASET_REGISTRY

OUTPUT_DIR = "output"
MOCK_ROOT  = os.path.join(OUTPUT_DIR, "mock")

# Default match file produced by Member 3 (value + collective ER)
DEFAULT_MATCH_FILE = "match_results_collective.csv"


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def _resolve_cleaned_source_paths(dataset: str, use_mock: bool, prefix: str) -> tuple[str, str]:
    if use_mock:
        return (
            os.path.join(prefix, "cleaned_source1.csv"),
            os.path.join(prefix, "cleaned_source2.csv"),
        )
    return (
        os.path.join("data", "cleaned", dataset, "cleaned_source1.csv"),
        os.path.join("data", "cleaned", dataset, "cleaned_source2.csv"),
    )


def _load_matches(match_path: str) -> pd.DataFrame:
    """
    Load a Member 3 match result CSV and return only confirmed matches
    (is_match == 1) with a normalised 'similarity_score' column.

    Handles both 'final_score' (collective) and 'similarity_score' (single
    method) column names.
    """
    df = pd.read_csv(match_path, dtype={"id_A": str, "id_B": str})

    if "final_score" in df.columns and "similarity_score" not in df.columns:
        df = df.rename(columns={"final_score": "similarity_score"})

    required = {"id_A", "id_B", "is_match", "similarity_score"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Match file {match_path!r} is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df["is_match"] = df["is_match"].astype(int)
    df_matches = df[df["is_match"] == 1].copy()
    return df_matches


def run_clustering(
    dataset: str,
    use_mock: bool,
    match_file: str,
    *,
    skip_missing: bool,
) -> dict | None:
    """
    Run the full Member 4 pipeline for one dataset.
    Returns a stats dict on success, None when skipped.
    """
    cfg = DATASET_REGISTRY[dataset]

    prefix = os.path.join(MOCK_ROOT, dataset) if use_mock else os.path.join(OUTPUT_DIR, dataset)
    match_path   = os.path.join(prefix, match_file)
    out_clusters = os.path.join(prefix, "clusters.csv")
    out_merged   = os.path.join(prefix, "merged_entities.csv")

    source1_path, source2_path = _resolve_cleaned_source_paths(dataset, use_mock, prefix)

    # ── Guard missing files ───────────────────────────────────────────────────
    if not os.path.isfile(match_path):
        msg = f"Missing match file: {match_path}"
        if skip_missing:
            print(f"  [SKIP] {msg}")
            return None
        raise FileNotFoundError(msg)

    for p in [source1_path, source2_path]:
        if not os.path.isfile(p):
            msg = f"Missing cleaned entity file: {p}"
            if skip_missing:
                print(f"  [SKIP] {msg}")
                return None
            raise FileNotFoundError(msg)

    # ── Load matches ──────────────────────────────────────────────────────────
    print(f"\n  [LOAD] {match_path}")
    df_all     = pd.read_csv(match_path, dtype={"id_A": str, "id_B": str})
    df_matches = _load_matches(match_path)

    n_pairs   = len(df_all)
    n_matches = len(df_matches)
    print(f"  [LOAD] {n_pairs:,} total pairs  →  {n_matches:,} confirmed matches (is_match=1)")

    if n_matches == 0:
        print("  [WARN] No confirmed matches — writing empty cluster files.")
        empty_c = pd.DataFrame(columns=["cluster_id", "entity_id"])
        os.makedirs(prefix, exist_ok=True)
        empty_c.to_csv(out_clusters, index=False)
        merge_cluster_attributes(empty_c, source1_path, source2_path).to_csv(out_merged, index=False)
        return {
            "n_input_pairs": n_pairs,
            "n_confirmed_matches": 0,
            "n_clusters": 0,
            "n_merged_entities": 0,
        }

    # ── Clustering ────────────────────────────────────────────────────────────
    print("  [CLUST] Running 'connected_components' ...")
    t0 = time.time()
    df_clusters = build_connected_components(df_matches, source1_name=cfg["source1"], source2_name=cfg["source2"])
    elapsed = time.time() - t0

    n_clusters = df_clusters["cluster_id"].nunique()
    print(f"  [CLUST] {n_clusters:,} clusters  ({elapsed:.2f}s)")

    # ── Attribute merging ─────────────────────────────────────────────────────
    print(f"  [MERGE] Merging attributes from cleaned sources ...")
    df_merged = merge_cluster_attributes(df_clusters, source1_path, source2_path)
    print(f"  [MERGE] {len(df_merged):,} canonical records produced")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(prefix, exist_ok=True)
    df_clusters.to_csv(out_clusters, index=False)
    df_merged.to_csv(out_merged,     index=False)
    print(f"  [SAVE] {out_clusters}")
    print(f"  [SAVE] {out_merged}")

    return {
        "n_input_pairs":      n_pairs,
        "n_confirmed_matches": n_matches,
        "n_clusters":         n_clusters,
        "n_merged_entities":  len(df_merged),
    }


def main() -> None:
    valid = list(DATASET_REGISTRY.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description="Member 4 — Clustering + entity merging → clusters.csv / merged_entities.csv"
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
        help="Read from and write to output/mock/<dataset>/ instead of output/<dataset>/",
    )
    parser.add_argument(
        "--match-file",
        default=DEFAULT_MATCH_FILE,
        metavar="FILENAME",
        help=f"Match result CSV to cluster on (default: {DEFAULT_MATCH_FILE})",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets whose match file or cleaned sources are missing",
    )
    args = parser.parse_args()

    datasets = (
        list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    )

    results: dict[str, dict] = {}

    for ds_key in datasets:
        banner(f"DATASET: {ds_key.upper()}  [algorithm=connected_components]")
        stats = run_clustering(
            dataset   = ds_key,
            use_mock  = args.mock,
            match_file= args.match_file,
            skip_missing=args.skip_missing,
        )
        if stats is not None:
            results[ds_key] = stats

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("MEMBER 4 COMPLETE — SUMMARY")
    print(
        f"\n  {'Dataset':<20} {'Matches':>8} {'Clusters':>10} {'Merged':>8}"
    )
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8}")
    for ds_key, st in results.items():
        print(
            f"  {ds_key:<20} "
            f"{st['n_confirmed_matches']:>8,} "
            f"{st['n_clusters']:>10,} "
            f"{st['n_merged_entities']:>8,}"
        )

    root = MOCK_ROOT if args.mock else OUTPUT_DIR
    print(f"\n  Output files per dataset (under {root}/<dataset>/):")
    print(f"    clusters.csv        — cluster_id, entity_id")
    print(f"    merged_entities.csv — one canonical record per cluster")


if __name__ == "__main__":
    main()
