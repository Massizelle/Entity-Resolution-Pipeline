"""
Member 1 - Main Pipeline Orchestrator
=======================================
Runs the full Member 1 pipeline (ingestion + blocking) for:
  - abt_buy
  - amazon_google
  - spimbench

Usage:
    # Day 1 — generate mocks only (fast, no data read needed)
    python run_member1.py --mocks-only

    # Full pipeline — all three datasets
    python run_member1.py

    # One dataset only
    python run_member1.py --dataset abt_buy
    python run_member1.py --dataset amazon_google
    python run_member1.py --dataset spimbench

    # If raw data directories differ from defaults, set env variables:
    #   ABT_BUY_DIR, AMAZON_GOOGLE_DIR, SPIMBENCH_DIR
"""

from __future__ import annotations

import os
import time
import argparse

# All imports at top level — errors surface at startup, not mid-run
from create_mocks   import generate_all_mocks
from data_ingestion import run_ingestion, DATASET_REGISTRY
from blocking       import run_token_blocking, compute_blocking_stats

OUTPUT_DIR = "output"
MOCK_DIR   = os.path.join(OUTPUT_DIR, "mock")


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run_pipeline(
    datasets_to_run: list[str],
    download: bool = False,
    mocks_only: bool = False,
    max_block_size: int = 1000,
) -> None:

    # ── Step 0: Mocks ─────────────────────────────────────────────────────
    banner("STEP 0 — Generating mock files for the team")
    generate_all_mocks(root_dir=MOCK_DIR)

    if mocks_only:
        print("\n[INFO] --mocks-only flag set. Stopping here.")
        print(f"[INFO] Push {MOCK_DIR}/ to Git so your teammates can start!")
        return

    all_stats: dict = {}

    for ds_key in datasets_to_run:
        cfg = DATASET_REGISTRY[ds_key]
        banner(f"DATASET: {ds_key.upper()}  [{cfg['data_type']}]")

        # ── Step 1: Ingestion ──────────────────────────────────────────────
        print(f"\n--- Step 1: Ingestion ---")
        t0 = time.time()
        df1, df2, df_truth = run_ingestion(dataset=ds_key, download=download)
        print(f"  [TIME] Ingestion: {time.time()-t0:.1f}s")

        # ── Step 2: Token Blocking ─────────────────────────────────────────
        print(f"\n--- Step 2: Token Blocking ---")
        ds_out = os.path.join(OUTPUT_DIR, ds_key)
        os.makedirs(ds_out, exist_ok=True)
        blocks_path = os.path.join(ds_out, "blocks.csv")

        # Time algorithm separately from I/O
        t0 = time.time()
        df_blocks = run_token_blocking(
            df1, df2,
            max_block_size=max_block_size,
            output_path=None,   # no I/O inside timed section
        )
        blocking_time = time.time() - t0
        print(f"  [TIME] Blocking (algo only): {blocking_time:.1f}s")

        df_blocks.to_csv(blocks_path, index=False)
        print(f"  [SAVE] {blocks_path}")

        # ── Step 3: Stats ──────────────────────────────────────────────────
        print(f"\n--- Step 3: Blocking Quality Metrics ---")
        tc1 = cfg.get("truth_col_s1", "")
        tc2 = cfg.get("truth_col_s2", "")
        stats = compute_blocking_stats(
            df_blocks, df_truth,
            source1_name=cfg["source1"],
            source2_name=cfg["source2"],
            truth_col_s1=tc1,
            truth_col_s2=tc2,
        )

        all_stats[ds_key] = {
            "blocks_path":   blocks_path,
            "blocking_time": blocking_time,
            "n_blocks":      df_blocks["block_id"].nunique(),
            "n_entities":    df_blocks["entity_id"].nunique(),
            "stats":         stats,
        }

    # ── Final summary ──────────────────────────────────────────────────────
    banner("MEMBER 1 COMPLETE — SUMMARY")
    print(f"\n  {'Dataset':<20} {'Blocks':>8} {'Entities':>10} "
          f"{'Time(s)':>8} {'RR':>6} {'PC':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")

    for ds_key, info in all_stats.items():
        s = info["stats"]
        rr = s["reduction_ratio"]
        pc = s["pairs_completeness"]
        print(f"  {ds_key:<20} {info['n_blocks']:>8,} "
              f"{info['n_entities']:>10,} "
              f"{info['blocking_time']:>8.1f} "
              f"{str(rr):>6} {str(pc):>6}")

    print("\n  Output files per dataset:")
    for ds_key, info in all_stats.items():
        print(f"    {ds_key:<20} → {info['blocks_path']}")

    print(f"\n  ➜ Share output/<dataset>/blocks.csv with Member 2 on April 9.")
    print(f"  ➜ {MOCK_DIR}/ is ready — push to Git now!")


def main():
    valid_choices = list(DATASET_REGISTRY.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description="Member 1 — Full ER Pipeline (Ingestion + Blocking)"
    )
    parser.add_argument(
        "--dataset",
        choices=valid_choices,
        default="all",
        help="Dataset to process (default: all)"
    )
    parser.add_argument(
        "--mocks-only", action="store_true",
        help="Only generate mock files (fast, no data read)"
    )
    parser.add_argument(
        "--max-block-size", type=int, default=1000,
        help="Block purging threshold (default: 1000)"
    )
    args = parser.parse_args()

    datasets = (list(DATASET_REGISTRY.keys())
                if args.dataset == "all" else [args.dataset])

    run_pipeline(
        datasets_to_run=datasets,
        mocks_only=args.mocks_only,
        max_block_size=args.max_block_size,
    )


if __name__ == "__main__":
    main()