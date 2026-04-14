"""
Full Entity Resolution Pipeline — End-to-End Orchestrator
==========================================================
Chains all four members in sequence for one or more datasets:

  Step 1  Member 1  pipeline/data_ingestion.py + pipeline/blocking.py
              → data/cleaned/<dataset>/cleaned_source{1,2}.csv
              → output/<dataset>/blocks.csv

  Step 2  Member 2  pipeline/block_processing.py
              → output/<dataset>/candidate_pairs.csv

  Step 3  Member 3  pipeline/matching.py
              → output/<dataset>/match_results_*.csv

  Step 4  Member 4  pipeline/clustering.py
              → output/<dataset>/clusters.csv
              → output/<dataset>/merged_entities.csv

Usage:
    # Full pipeline — all datasets
    python cli/run_pipeline.py

    # Single dataset
    python cli/run_pipeline.py --dataset abt_buy
    python cli/run_pipeline.py --dataset amazon_google

    # Start from a specific step (steps 1-4, useful when upstream outputs exist)
    python cli/run_pipeline.py --from-step 2
    python cli/run_pipeline.py --dataset abt_buy --from-step 3

    # Stop after a specific step
    python cli/run_pipeline.py --to-step 2

    # Range of steps
    python cli/run_pipeline.py --from-step 2 --to-step 3

    # Tuning parameters
    python cli/run_pipeline.py --max-block-size 500 --min-jaccard 0.1 --algorithm center

    # Skip datasets that are missing required input files (instead of crashing)
    python cli/run_pipeline.py --from-step 2 --skip-missing

    # Mock mode — use output/mock/<dataset>/ fixtures throughout
    python cli/run_pipeline.py --mock --from-step 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_ingestion import DATASET_REGISTRY, run_ingestion
from pipeline.blocking        import run_token_blocking, compute_blocking_stats
from pipeline.block_processing import run_block_processing
from pipeline.matching        import run_matching
from pipeline.clustering      import build_connected_components, build_center_clustering, merge_cluster_attributes

OUTPUT_DIR = "output"
MOCK_ROOT  = os.path.join(OUTPUT_DIR, "mock")

CLUSTERING_ALGORITHMS = {
    "connected_components": build_connected_components,
    "center":               build_center_clustering,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def step_banner(step: int, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  STEP {step} — {label}")
    print(f"{'─'*60}")


def _missing(path: str, skip: bool) -> bool:
    """Return True (and warn) if path does not exist and skip is set."""
    if not os.path.isfile(path):
        if skip:
            print(f"  [SKIP] Missing: {path}")
            return True
        raise FileNotFoundError(f"Required file not found: {path}")
    return False


# ── Per-step runners ───────────────────────────────────────────────────────────

def step1_ingest_and_block(
    dataset: str,
    max_block_size: int,
    *,
    skip_missing: bool = False,
) -> dict[str, Any] | None:
    """Member 1: ingestion + token blocking. Returns None when skipped."""
    cfg = DATASET_REGISTRY[dataset]

    t0 = time.time()
    try:
        df1, df2, df_truth = run_ingestion(dataset=dataset)
    except FileNotFoundError as exc:
        if skip_missing:
            print(f"  [SKIP] {exc}")
            return None
        raise
    ingest_time = time.time() - t0

    out_dir     = os.path.join(OUTPUT_DIR, dataset)
    blocks_path = os.path.join(out_dir, "blocks.csv")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    df_blocks = run_token_blocking(df1, df2, max_block_size=max_block_size)
    block_time = time.time() - t0

    df_blocks.to_csv(blocks_path, index=False)
    print(f"  [SAVE] {blocks_path}")

    stats = compute_blocking_stats(
        df_blocks, df_truth,
        source1_name=cfg["source1"],
        source2_name=cfg["source2"],
        truth_col_s1=cfg.get("truth_col_s1", ""),
        truth_col_s2=cfg.get("truth_col_s2", ""),
    )

    return {
        "blocks_path":   blocks_path,
        "n_blocks":      df_blocks["block_id"].nunique(),
        "n_entities":    df_blocks["entity_id"].nunique(),
        "ingest_time":   ingest_time,
        "block_time":    block_time,
        "rr":  stats["reduction_ratio"],
        "pc":  stats["pairs_completeness"],
    }


def step2_block_processing(
    dataset: str,
    use_mock: bool,
    max_block_size: int,
    min_jaccard: float,
    *,
    candidate_strategy: str,
    top_candidates_per_entity: int | None,
    skip_missing: bool,
    write_stats: bool = False,
) -> dict[str, Any] | None:
    """Member 2: block purging + meta-blocking → candidate_pairs.csv."""
    cfg = DATASET_REGISTRY[dataset]

    prefix      = os.path.join(MOCK_ROOT, dataset) if use_mock else os.path.join(OUTPUT_DIR, dataset)
    blocks_path = os.path.join(prefix, "blocks.csv")
    out_path    = os.path.join(prefix, "candidate_pairs.csv")
    stats_path  = os.path.join(prefix, "member2_stats.json") if write_stats else None
    cleaned_root = prefix if use_mock else os.path.join("data", "cleaned", dataset)

    if _missing(blocks_path, skip_missing):
        return None

    _df, stats = run_block_processing(
        blocks_path    = blocks_path,
        output_path    = out_path,
        source1        = cfg["source1"],
        source2        = cfg["source2"],
        max_block_size = max_block_size,
        min_jaccard    = min_jaccard,
        candidate_strategy = candidate_strategy,
        source1_cleaned_path = os.path.join(cleaned_root, "cleaned_source1.csv"),
        source2_cleaned_path = os.path.join(cleaned_root, "cleaned_source2.csv"),
        top_candidates_per_entity = top_candidates_per_entity,
        write_stats_json=stats_path,
    )
    return stats


def step3_matching(
    dataset: str,
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
) -> dict[str, Any] | None:
    """Member 3: value similarity + collective ER → match_results_*.csv."""
    prefix      = os.path.join(MOCK_ROOT, dataset) if use_mock else os.path.join(OUTPUT_DIR, dataset)
    pairs_path  = os.path.join(prefix, "candidate_pairs.csv")

    if _missing(pairs_path, skip_missing):
        return False

    result = run_matching(
        dataset,
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
    return result


def step4_clustering(
    dataset: str,
    use_mock: bool,
    algorithm: str,
    *,
    skip_missing: bool,
) -> dict[str, Any] | None:
    """Member 4: clustering + attribute merging → clusters.csv / merged_entities.csv."""
    prefix       = os.path.join(MOCK_ROOT, dataset) if use_mock else os.path.join(OUTPUT_DIR, dataset)
    match_path   = os.path.join(prefix, "match_results_collective.csv")
    out_clusters = os.path.join(prefix, "clusters.csv")
    out_merged   = os.path.join(prefix, "merged_entities.csv")
    # In mock mode, cleaned sources live alongside the other mock fixtures.
    if use_mock:
        source1_path = os.path.join(prefix, "cleaned_source1.csv")
        source2_path = os.path.join(prefix, "cleaned_source2.csv")
    else:
        source1_path = os.path.join("data", "cleaned", dataset, "cleaned_source1.csv")
        source2_path = os.path.join("data", "cleaned", dataset, "cleaned_source2.csv")

    for p in [match_path, source1_path, source2_path]:
        if _missing(p, skip_missing):
            return None

    df_all     = __import__("pandas").read_csv(match_path, dtype={"id_A": str, "id_B": str})
    df_matches = df_all[df_all["is_match"].astype(int) == 1].copy()

    if "final_score" in df_matches.columns and "similarity_score" not in df_matches.columns:
        df_matches = df_matches.rename(columns={"final_score": "similarity_score"})

    n_matches = len(df_matches)
    print(f"  [CLUST] {n_matches:,} confirmed matches → clustering with '{algorithm}'")

    if n_matches == 0:
        print("  [WARN] No confirmed matches — writing empty output files.")
        import pandas as pd
        empty_clusters = pd.DataFrame(columns=["cluster_id", "entity_id"])
        empty_clusters.to_csv(out_clusters, index=False)
        merge_cluster_attributes(empty_clusters, source1_path, source2_path).to_csv(out_merged, index=False)
        return {"n_clusters": 0, "n_merged_entities": 0}

    cfg         = DATASET_REGISTRY[dataset]
    cluster_fn  = CLUSTERING_ALGORITHMS[algorithm]
    df_clusters = cluster_fn(df_matches, source1_name=cfg["source1"], source2_name=cfg["source2"])
    n_clusters  = df_clusters["cluster_id"].nunique()
    print(f"  [CLUST] {n_clusters:,} clusters produced")

    df_merged = merge_cluster_attributes(df_clusters, source1_path, source2_path)
    print(f"  [MERGE] {len(df_merged):,} canonical records")

    os.makedirs(prefix, exist_ok=True)
    df_clusters.to_csv(out_clusters, index=False)
    df_merged.to_csv(out_merged,     index=False)
    print(f"  [SAVE] {out_clusters}")
    print(f"  [SAVE] {out_merged}")

    return {
        "n_confirmed_matches": n_matches,
        "n_clusters":          n_clusters,
        "n_merged_entities":   len(df_merged),
    }


# ── Main orchestration ─────────────────────────────────────────────────────────

def run_pipeline(
    datasets:       list[str],
    from_step:      int,
    to_step:        int,
    use_mock:       bool,
    max_block_size: int,
    min_jaccard:    float,
    limit:          int | None,
    algorithm:      str,
    *,
    candidate_strategy: str,
    top_candidates_per_entity: int | None,
    progressive_stages: str | None,
    time_limit_minutes: float | None,
    resume: bool,
    chunk_size: int,
    online_clustering: bool,
    online_cluster_every_n_chunks: int,
    skip_missing:   bool,
) -> None:
    t_pipeline_start = time.time()
    all_results: dict[str, dict] = {}

    for ds_key in datasets:
        banner(f"DATASET: {ds_key.upper()}")
        ds_stats: dict[str, Any] = {}

        # ── Step 1 ────────────────────────────────────────────────────────────
        if from_step <= 1 <= to_step:
            step_banner(1, "Ingestion + Token Blocking  [Member 1]")
            s = step1_ingest_and_block(ds_key, max_block_size, skip_missing=skip_missing)
            if s is None:
                print(f"  [SKIP] Dataset {ds_key!r} skipped at step 1.")
                continue
            ds_stats["step1"] = s
            print(
                f"\n  Step 1 done — blocks={s['n_blocks']:,}  "
                f"RR={s['rr']}  PC={s['pc']}"
            )

        # ── Step 2 ────────────────────────────────────────────────────────────
        if from_step <= 2 <= to_step:
            step_banner(2, "Block Purging + Meta-Blocking  [Member 2]")
            s = step2_block_processing(
                ds_key, use_mock, max_block_size, min_jaccard,
                candidate_strategy=candidate_strategy,
                top_candidates_per_entity=top_candidates_per_entity,
                skip_missing=skip_missing,
            )
            if s is None:
                print(f"  [SKIP] Dataset {ds_key!r} skipped at step 2.")
                continue
            ds_stats["step2"] = s
            print(
                f"\n  Step 2 done — candidate_pairs={s.get('candidate_pairs', '?'):,}  "
                f"reduction={s.get('reduction_vs_cartesian', 'n/a')}"
            )

        # ── Step 3 ────────────────────────────────────────────────────────────
        if from_step <= 3 <= to_step:
            step_banner(3, "Entity Matching  [Member 3]")
            s = step3_matching(
                ds_key,
                use_mock,
                limit,
                candidate_strategy=candidate_strategy,
                progressive_stages=progressive_stages,
                time_limit_minutes=time_limit_minutes,
                resume=resume,
                chunk_size=chunk_size,
                online_clustering=online_clustering,
                clustering_algorithm=algorithm,
                online_cluster_every_n_chunks=online_cluster_every_n_chunks,
                skip_missing=skip_missing,
            )
            if s is None:
                print(f"  [SKIP] Dataset {ds_key!r} skipped at step 3.")
                continue
            ds_stats["step3"] = s
            if s.get("completed"):
                print(f"\n  Step 3 done — match_results_*.csv written.")
            else:
                print(
                    f"\n  Step 3 partial — processed={s.get('processed_rows', 0):,}/"
                    f"{s.get('total_rows', 0):,} candidate rows. Resume later to continue."
                )

        # ── Step 4 ────────────────────────────────────────────────────────────
        if from_step <= 4 <= to_step:
            if "step3" in ds_stats and not ds_stats["step3"].get("completed", True):
                print("\n  [SKIP] Step 4 skipped because Step 3 is partial/incomplete.")
                continue
            step_banner(4, "Clustering + Entity Merging  [Member 4]")
            s = step4_clustering(
                ds_key, use_mock, algorithm,
                skip_missing=skip_missing,
            )
            if s is None:
                print(f"  [SKIP] Dataset {ds_key!r} skipped at step 4.")
                continue
            ds_stats["step4"] = s
            print(
                f"\n  Step 4 done — clusters={s['n_clusters']:,}  "
                f"merged_entities={s['n_merged_entities']:,}"
            )

        all_results[ds_key] = ds_stats

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_pipeline_start
    banner(f"PIPELINE COMPLETE  (steps {from_step}→{to_step})  [{elapsed:.1f}s total]")

    root = MOCK_ROOT if use_mock else OUTPUT_DIR

    for ds_key, ds_stats in all_results.items():
        print(f"\n  [{ds_key}]")
        if "step1" in ds_stats:
            s = ds_stats["step1"]
            print(f"    Step 1  blocks={s['n_blocks']:,}  RR={s['rr']}  PC={s['pc']}")
        if "step2" in ds_stats:
            s = ds_stats["step2"]
            print(
                f"    Step 2  candidate_pairs={s.get('candidate_pairs', '?'):,}  "
                f"reduction={s.get('reduction_vs_cartesian', 'n/a')}"
            )
        if "step3" in ds_stats:
            s = ds_stats["step3"]
            if s.get("completed"):
                print(f"    Step 3  match_results_*.csv  ✓")
            else:
                print(
                    f"    Step 3  partial {s.get('processed_rows', 0):,}/"
                    f"{s.get('total_rows', 0):,} rows"
                )
        if "step4" in ds_stats:
            s = ds_stats["step4"]
            print(
                f"    Step 4  clusters={s['n_clusters']:,}  "
                f"merged_entities={s['n_merged_entities']:,}"
            )

    print(f"\n  Outputs under: {root}/<dataset>/")
    print(f"    blocks.csv, candidate_pairs.csv")
    print(f"    match_results_{{jaccard,tfidf,sbert,combined,collective}}.csv")
    print(f"    clusters.csv, merged_entities.csv")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    valid_datasets = list(DATASET_REGISTRY.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description="End-to-end ER pipeline: ingestion → blocking → matching → clustering"
    )
    parser.add_argument(
        "--dataset",
        choices=valid_datasets,
        default="all",
        help="Dataset to process (default: all)",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        metavar="N",
        help="First step to run (1=ingestion, 2=block-processing, 3=matching, 4=clustering)",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        metavar="N",
        help="Last step to run inclusive (default: 4)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use output/mock/<dataset>/ fixtures for steps 2-4 (skips step 1)",
    )
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=1000,
        help="Block purging threshold used in steps 1 and 2 (default: 1000)",
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=0.0,
        help="Meta-blocking Jaccard threshold for step 2 (default: 0 = keep all co-occurring pairs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of candidate pairs sent to step 3 (for quick testing)",
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=["v0", "cw_semantic_predictive"],
        default="v0",
        help="Candidate strategy for Member 2: v0 baseline or cw_semantic_predictive",
    )
    parser.add_argument(
        "--top-candidates-per-entity",
        type=int,
        default=None,
        help="Legacy option kept for CLI compatibility; ignored by the remaining strategies",
    )
    parser.add_argument(
        "--progressive-stages",
        default=None,
        help="Comma-separated progressive evaluation checkpoints, e.g. 1000,5000,10000",
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
        help="Number of candidate pairs processed per checkpointed matching chunk (default: 250)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing Member 3 cache/checkpoints and restart matching from zero",
    )
    parser.add_argument(
        "--online-clustering",
        action="store_true",
        help="Update checkpointed clustering outputs during Step 3",
    )
    parser.add_argument(
        "--online-cluster-every-n-chunks",
        type=int,
        default=1,
        help="Refresh online clustering every N matching chunks instead of every chunk (default: 1)",
    )
    parser.add_argument(
        "--algorithm",
        choices=list(CLUSTERING_ALGORITHMS.keys()),
        default="connected_components",
        help="Clustering algorithm for step 4 (default: connected_components)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets with missing input files instead of raising an error",
    )
    args = parser.parse_args()

    if args.from_step > args.to_step:
        parser.error(f"--from-step ({args.from_step}) must be <= --to-step ({args.to_step})")

    if args.mock and args.from_step == 1:
        print("[INFO] --mock implies step 1 output already exists. "
              "Forcing --from-step 2.")
        args.from_step = 2

    datasets = (
        list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    )

    run_pipeline(
        datasets       = datasets,
        from_step      = args.from_step,
        to_step        = args.to_step,
        use_mock       = args.mock,
        max_block_size = args.max_block_size,
        min_jaccard    = args.min_jaccard,
        limit          = args.limit,
        algorithm      = args.algorithm,
        candidate_strategy = args.candidate_strategy,
        top_candidates_per_entity = args.top_candidates_per_entity,
        progressive_stages = args.progressive_stages,
        time_limit_minutes = args.time_limit_minutes,
        resume         = not args.no_resume,
        chunk_size     = args.chunk_size,
        online_clustering = args.online_clustering,
        online_cluster_every_n_chunks = args.online_cluster_every_n_chunks,
        skip_missing   = args.skip_missing,
    )


if __name__ == "__main__":
    main()
