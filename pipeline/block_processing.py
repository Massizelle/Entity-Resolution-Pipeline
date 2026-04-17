"""
Member 2 — Witness-first candidate generation.

Reads blocks.csv from Member 1, purges oversized blocks, then runs the
`cw_semantic_predictive` candidate strategy to produce candidate_pairs.csv
for Member 3.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pipeline.constraint_witness import run_constraint_witness_resolution
from pipeline.progress import ProgressBar


@dataclass
class _GraphSummary:
    """Minimal graph-like summary for Member 2 stats."""

    node_count: int = 0
    edge_count: int = 0

    def number_of_nodes(self) -> int:
        return self.node_count

    def number_of_edges(self) -> int:
        return self.edge_count


def load_blocks_csv(path: str) -> pd.DataFrame:
    """Load and validate Member 1 blocks format."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"blocks.csv not found: {path}")
    df = pd.read_csv(path, dtype=str)
    required = {"block_id", "entity_id", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"blocks.csv missing columns {missing}: {path}")
    return df


def purge_oversized_blocks(
    df_blocks: pd.DataFrame,
    max_block_size: int,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Remove every row belonging to a block with strictly more than
    max_block_size entity rows (Member 2 block purging).

    Returns (filtered DataFrame, purge stats).
    """
    blocks_before = int(df_blocks["block_id"].nunique())
    rows_before = len(df_blocks)

    if max_block_size <= 0:
        if verbose:
            print(f"  [PURGE] max_block_size<=0: skipping purge ({blocks_before} blocks)")
        return df_blocks.copy(), {
            "blocks_before": blocks_before,
            "blocks_after": blocks_before,
            "rows_before": rows_before,
            "rows_after": rows_before,
            "oversized_blocks_removed": 0,
        }

    counts = df_blocks.groupby("block_id", observed=True).size()
    keep = counts[counts <= max_block_size].index
    removed_blocks = int((counts > max_block_size).sum())
    out = df_blocks[df_blocks["block_id"].isin(keep)].copy()
    blocks_after = int(out["block_id"].nunique())
    rows_after = len(out)

    if verbose:
        if removed_blocks:
            print(
                f"  [PURGE] Dropped {removed_blocks} oversized blocks "
                f"({blocks_before} -> {blocks_after} blocks kept, "
                f"threshold={max_block_size})"
            )
        else:
            print(f"  [PURGE] No blocks exceeded threshold={max_block_size}")

    stats = {
        "blocks_before": blocks_before,
        "blocks_after": blocks_after,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "oversized_blocks_removed": removed_blocks,
    }
    return out, stats


def _cartesian_bounds(
    df_blocks: pd.DataFrame, source1: str, source2: str
) -> tuple[int, int, int]:
    """Unique entity counts per source and n1*n2 (upper bound on distinct pairs)."""
    n1 = df_blocks.loc[df_blocks["source"] == source1, "entity_id"].nunique()
    n2 = df_blocks.loc[df_blocks["source"] == source2, "entity_id"].nunique()
    return int(n1), int(n2), int(n1 * n2)


def meta_blocking_candidate_pairs(
    df_blocks: pd.DataFrame,
    source1: str,
    source2: str,
    *,
    strategy: str = "cw_semantic_predictive",
    source1_cleaned_path: str | None = None,
    source2_cleaned_path: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Any, dict[str, int]]:
    """Run the single supported witness-driven candidate strategy."""
    strategy_stats: dict[str, int] = {}
    if strategy != "cw_semantic_predictive":
        raise ValueError(f"Unknown candidate strategy: {strategy}")
    if not (source1_cleaned_path and source2_cleaned_path):
        raise ValueError(
            f"strategy='{strategy}' requires source1_cleaned_path and source2_cleaned_path"
        )
    df_source1 = pd.read_csv(source1_cleaned_path, dtype=str).fillna("")
    df_source2 = pd.read_csv(source2_cleaned_path, dtype=str).fillna("")
    _witnesses_df, out, cw_stats = run_constraint_witness_resolution(
        df_source1,
        df_source2,
        source1=source1,
        source2=source2,
        blocks_df=df_blocks,
        max_cartesian_size=4,
        semantic_rescue=True,
        predictive_rescue=True,
        progress_label="  [STEP2]",
    )
    G = _GraphSummary(
        node_count=int(out["id_A"].astype(str).nunique() + out["id_B"].astype(str).nunique()) if not out.empty else 0,
        edge_count=len(out),
    )
    strategy_stats = {f"cw_{k}": v for k, v in cw_stats.items()}
    pair_count_before = int(cw_stats.get("candidate_pairs", len(out)))

    meta_stats = {
        "pairs_after_blocking_before_jaccard": pair_count_before,
        "pairs_after_jaccard": len(out),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "candidate_strategy": strategy,
        **strategy_stats,
    }

    if verbose:
        print(
            f"  [META] Cross-source pairs before Jaccard prune: {pair_count_before:,}  "
            f"after witness collapse: {len(out):,}"
        )
        print(
            f"  [META] Graph: |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}"
        )
        print(
            f"  [META] Witness-first collapse generated {len(out):,} candidate rows "
            f"from {strategy_stats.get('cw_seed_regions', 0):,} seed regions"
        )
        print(
            f"  [META] Semantic rescue added "
            f"{strategy_stats.get('cw_semantic_rescue_pairs', 0):,} rows"
        )
        print(
            f"  [META] Strong-witness rescue added "
            f"{strategy_stats.get('cw_strong_rescue_pairs', 0):,} rows"
        )
        print(
            f"  [META] Asymmetric-text rescue added "
            f"{strategy_stats.get('cw_asymmetric_rescue_pairs', 0):,} rows"
        )
        print(
            f"  [META] Facet rescue added "
            f"{strategy_stats.get('cw_facet_rescue_pairs', 0):,} rows"
        )
        print(
            f"  [META] Predictive rescue added "
            f"{strategy_stats.get('cw_predictive_rescue_pairs', 0):,} rows"
        )

    return out, G, meta_stats


def run_block_processing(
    blocks_path: str,
    output_path: str,
    source1: str,
    source2: str,
    max_block_size: int = 1000,
    *,
    candidate_strategy: str = "cw_semantic_predictive",
    source1_cleaned_path: str | None = None,
    source2_cleaned_path: str | None = None,
    verbose: bool = True,
    write_stats_json: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full Member 2 pipeline for one dataset.
    Writes candidate_pairs.csv and returns (DataFrame, combined stats dict).

    write_stats_json: if set, path to write JSON metrics (e.g. .../member2_stats.json).
    """
    if verbose:
        print(f"\n[MEMBER2] blocks:  {blocks_path}")
        print(f"[MEMBER2] sources: {source1!r} x {source2!r}")

    df = load_blocks_csv(blocks_path)
    if verbose:
        print(f"  [LOAD] {len(df):,} rows, {df['block_id'].nunique():,} blocks")

    df, purge_stats = purge_oversized_blocks(
        df, max_block_size=max_block_size, verbose=verbose
    )

    n1, n2, cartesian = _cartesian_bounds(df, source1, source2) if len(df) else (0, 0, 0)

    if df.empty:
        if verbose:
            print("  [WARN] No blocks left after purging — empty candidate set.")
        empty = pd.DataFrame(columns=["id_A", "id_B"])
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        empty.to_csv(output_path, index=False)
        combined: dict[str, Any] = {
            **{f"purge_{k}": v for k, v in purge_stats.items()},
            "entities_source1": n1,
            "entities_source2": n2,
            "cartesian_pairs": cartesian,
            "candidate_pairs": 0,
            "reduction_vs_cartesian": None if cartesian == 0 else 1.0,
        }
        if write_stats_json:
            os.makedirs(os.path.dirname(write_stats_json) or ".", exist_ok=True)
            with open(write_stats_json, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2)
        if verbose:
            print(f"  [SAVE] {output_path} (0 rows)")
        return empty, combined

    if candidate_strategy == "cw_semantic_predictive":
        for required_path in [source1_cleaned_path, source2_cleaned_path]:
            if not required_path or not os.path.isfile(required_path):
                raise FileNotFoundError(
                    f"strategy='{candidate_strategy}' requires cleaned source files. "
                    f"Missing: {required_path}"
                )

    pairs, _G, meta_stats = meta_blocking_candidate_pairs(
        df,
        source1,
        source2,
        strategy=candidate_strategy,
        source1_cleaned_path=source1_cleaned_path,
        source2_cleaned_path=source2_cleaned_path,
        verbose=verbose,
    )

    n_candidates = len(pairs)
    reduction = (
        1.0 - (n_candidates / cartesian) if cartesian > 0 else 0.0
    )

    combined = {
        **{f"purge_{k}": v for k, v in purge_stats.items()},
        "entities_source1": n1,
        "entities_source2": n2,
        "cartesian_pairs": cartesian,
        "candidate_pairs": n_candidates,
        "reduction_vs_cartesian": round(reduction, 6),
        **{f"meta_{k}": v for k, v in meta_stats.items()},
        "max_block_size": max_block_size,
        "candidate_strategy": candidate_strategy,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pairs.to_csv(output_path, index=False)
    if verbose:
        print(f"  [SAVE] {output_path}  ({len(pairs):,} pairs)")
        if cartesian > 0:
            print(
                f"  [STATS] vs Cartesian (n1*n2={cartesian:,}): "
                f"reduction ratio = {reduction:.4f}"
            )

    if write_stats_json:
        os.makedirs(os.path.dirname(write_stats_json) or ".", exist_ok=True)
        with open(write_stats_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        if verbose:
            print(f"  [SAVE] {write_stats_json}")

    return pairs, combined
