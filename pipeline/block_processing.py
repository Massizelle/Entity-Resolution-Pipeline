"""
Member 2 — Block Processing (Block Purging + Meta-Blocking)
============================================================
Reads blocks.csv from Member 1, reduces comparisons, outputs candidate_pairs.csv
for Member 3.

1. Block purging: drop any block_id whose row count exceeds max_block_size.
2. Meta-blocking: entities are linked if they co-occur in at least one block
   after purging. Edge weight = Jaccard similarity of their block-id sets.
   Pairs below min_jaccard are pruned.
3. Candidate pairs: remaining cross-source edges (source1 x source2).

Input:  block_id, entity_id, source
Output: id_A, id_B  (id_A always from source1, id_B from source2)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import pandas as pd

from pipeline.constraint_witness import run_constraint_witness_resolution

try:
    import networkx as nx
except ImportError:  # pragma: no cover - exercised only in lean environments
    class _FallbackGraph:
        def __init__(self) -> None:
            self._nodes: dict[str, dict[str, Any]] = {}
            self._edges: dict[tuple[str, str], dict[str, Any]] = {}

        def add_node(self, node: str, **attrs: Any) -> None:
            self._nodes.setdefault(node, {}).update(attrs)

        def add_edge(self, left: str, right: str, **attrs: Any) -> None:
            self.add_node(left)
            self.add_node(right)
            key = tuple(sorted((left, right)))
            self._edges[key] = attrs

        def number_of_nodes(self) -> int:
            return len(self._nodes)

        def number_of_edges(self) -> int:
            return len(self._edges)

    class _NXFallback:
        Graph = _FallbackGraph

    nx = _NXFallback()


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


def _entity_block_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    """entity_id -> set of block_ids."""
    m: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        m[str(row["entity_id"])].add(str(row["block_id"]))
    return dict(m)


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
    min_jaccard: float = 0.0,
    *,
    strategy: str = "v0",
    source1_cleaned_path: str | None = None,
    source2_cleaned_path: str | None = None,
    top_candidates_per_entity: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, nx.Graph, dict[str, int]]:
    """
    Build weighted graph: nodes = entities, edges = cross-source pairs that
    share at least one block, weight = Jaccard(block_sets(u), block_sets(v)).
    Keep edges with weight >= min_jaccard.

    Returns (candidate_pairs DataFrame, graph, meta stats).
    """
    eb = _entity_block_sets(df_blocks)

    pair_set: set[tuple[str, str]] = set()
    for _, grp in df_blocks.groupby("block_id", observed=True):
        s1 = grp.loc[grp["source"] == source1, "entity_id"].astype(str).tolist()
        s2 = grp.loc[grp["source"] == source2, "entity_id"].astype(str).tolist()
        for a in s1:
            for b in s2:
                pair_set.add((a, b))

    G = nx.Graph()
    kept_pair_set: set[tuple[str, str]] = set()

    for a, b in pair_set:
        Ba, Bb = eb.get(a, set()), eb.get(b, set())
        inter = len(Ba & Bb)
        union = len(Ba | Bb)
        if union == 0:
            continue
        w = inter / union
        if w < min_jaccard:
            continue
        G.add_node(a, source=source1)
        G.add_node(b, source=source2)
        G.add_edge(a, b, weight=w, jaccard=w)
        kept_pair_set.add((a, b))

    filtered_pair_set = kept_pair_set
    block_sizes = (
        df_blocks.groupby("block_id", observed=True).size().astype(int).to_dict()
    )

    strategy_stats: dict[str, int] = {}
    if strategy == "cw_semantic_predictive":
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
        )
        strategy_stats = {f"cw_{k}": v for k, v in cw_stats.items()}
    elif strategy == "v0":
        out = (
            pd.DataFrame(sorted(filtered_pair_set), columns=["id_A", "id_B"])
            .drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        raise ValueError(f"Unknown candidate strategy: {strategy}")

    meta_stats = {
        "pairs_after_blocking_before_jaccard": len(pair_set),
        "pairs_after_jaccard": len(out),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "candidate_strategy": strategy,
        **strategy_stats,
    }

    if verbose:
        print(
            f"  [META] Cross-source pairs before Jaccard prune: {len(pair_set):,}  "
            f"after (>={min_jaccard:g}): {len(out):,}"
        )
        print(
            f"  [META] Graph: |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}"
        )
        if strategy == "cw_semantic_predictive":
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
    min_jaccard: float = 0.0,
    *,
    candidate_strategy: str = "v0",
    source1_cleaned_path: str | None = None,
    source2_cleaned_path: str | None = None,
    top_candidates_per_entity: int | None = None,
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
        min_jaccard=min_jaccard,
        strategy=candidate_strategy,
        source1_cleaned_path=source1_cleaned_path,
        source2_cleaned_path=source2_cleaned_path,
        top_candidates_per_entity=top_candidates_per_entity,
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
        "min_jaccard": min_jaccard,
        "max_block_size": max_block_size,
        "candidate_strategy": candidate_strategy,
        "top_candidates_per_entity": top_candidates_per_entity,
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
