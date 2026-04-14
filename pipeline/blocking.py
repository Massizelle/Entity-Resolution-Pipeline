"""
Member 1 - Token Blocking
==========================
Schema-agnostic Token Blocking for three datasets:
  - abt_buy          (CSV, semi-structured)
  - amazon_google    (CSV, semi-structured)
  - spimbench        (RDF/TTL, structured)

Output per dataset:
  output/<dataset>/blocks.csv   →  block_id, entity_id, source

Ground truth columns per dataset:
  abt_buy       : idAbt      / idBuy
  amazon_google : idAmazon   / idGoogleBase
  spimbench     : N/A (no CSV ground truth)
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict

import pandas as pd


OUTPUT_DIR = "output"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "it",
    "on", "with", "as", "at", "by", "from", "that", "this", "was", "are",
    "be", "been", "have", "has", "had", "not", "but", "they", "we", "you",
    "he", "she", "do", "did", "will", "would", "can", "could", "may",
    "might", "should", "shall", "its", "their", "our", "your", "his", "her",
    "new", "one", "all", "more", "also", "so", "if", "then", "than",
}

# Ground truth column names per dataset
TRUTH_COLS: dict[str, tuple[str, str]] = {
    "abt_buy":       ("idAbt",    "idBuy"),
    "amazon_google": ("idAmazon", "idGoogleBase"),
    "spimbench":     ("",         ""),   # no CSV ground truth
}


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize(text: str, min_len: int = 2) -> set[str]:
    """Normalize then tokenize: lowercase, strip specials, remove stopwords."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return {
        tok for tok in text.split()
        if len(tok) >= min_len and tok not in STOPWORDS
    }


def get_entity_tokens(row: pd.Series, exclude_cols: set[str]) -> set[str]:
    """Collect tokens from all non-metadata attribute values of a row."""
    tokens: set[str] = set()
    for col, val in row.items():
        if col in exclude_cols:
            continue
        if isinstance(val, str) and val:
            tokens |= tokenize(val)
    return tokens


# ── Inverted index ────────────────────────────────────────────────────────────

def build_inverted_index(
    df: pd.DataFrame,
    entity_id_col: str = "id",
    source_col: str = "source",
) -> dict[str, list[tuple[str, str]]]:
    """token → [(entity_id, source), ...]"""
    exclude: set[str] = {entity_id_col, source_col}
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for _, row in df.iterrows():
        eid    = str(row[entity_id_col])
        source = str(row[source_col])
        for tok in get_entity_tokens(row, exclude):
            index[tok].append((eid, source))

    return dict(index)


def apply_block_size_limit(
    index: dict[str, list[tuple[str, str]]],
    max_block_size: int = 1000,
) -> dict[str, list[tuple[str, str]]]:
    """Remove oversized blocks (Block Purging)."""
    before   = len(index)
    filtered = {t: e for t, e in index.items() if len(e) <= max_block_size}
    removed  = before - len(filtered)
    if removed:
        print(f"  [PURGE] Removed {removed} oversized blocks "
              f"(threshold={max_block_size}). Remaining: {len(filtered)}")
    return filtered


def index_to_blocks_df(
    index: dict[str, list[tuple[str, str]]]
) -> pd.DataFrame:
    """Convert inverted index to canonical blocks.csv format."""
    rows = [
        {"block_id": token, "entity_id": eid, "source": src}
        for token, entries in index.items()
        for (eid, src) in entries
    ]
    return pd.DataFrame(rows, columns=["block_id", "entity_id", "source"])


# ── Main blocking pipeline ────────────────────────────────────────────────────

def run_token_blocking(
    df_source1: pd.DataFrame,
    df_source2: pd.DataFrame,
    entity_id_col: str = "id",
    source_col: str = "source",
    max_block_size: int = 1000,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Full Token Blocking on two entity DataFrames.
    I/O (CSV write) is intentionally excluded — call output_path=None
    for clean timing, then save manually.
    """
    df_all = pd.concat([df_source1, df_source2], ignore_index=True)
    n1, n2 = len(df_source1), len(df_source2)
    print(f"  [BLOCK] Entities: {len(df_all)} ({n1} + {n2})")

    print("  [BLOCK] Building inverted token index ...")
    index = build_inverted_index(df_all, entity_id_col, source_col)
    print(f"  [BLOCK] Distinct tokens: {len(index)}")

    index = apply_block_size_limit(index, max_block_size)

    df_blocks = index_to_blocks_df(index)

    n_blocks  = df_blocks["block_id"].nunique()
    avg_size  = df_blocks.groupby("block_id").size().mean()
    # Count unique entities per source and sum — avoids undercounting when
    # source1 and source2 share the same integer IDs (e.g. dblp_acm).
    n_in_blk  = df_blocks.groupby("source")["entity_id"].nunique().sum()
    print(f"  [BLOCK] Final blocks        : {n_blocks}")
    print(f"  [BLOCK] Entities in blocks  : {n_in_blk}")
    print(f"  [BLOCK] Avg block size      : {avg_size:.2f}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df_blocks.to_csv(output_path, index=False)
        print(f"  [BLOCK] Saved: {output_path}")

    return df_blocks


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_blocking_stats(
    df_blocks: pd.DataFrame,
    df_truth: pd.DataFrame,
    source1_name: str,
    source2_name: str,
    truth_col_s1: str = "",
    truth_col_s2: str = "",
) -> dict:
    """
    Compute Pairs Completeness, Pairs Quality, and Reduction Ratio.

    truth_col_s1 / truth_col_s2: column names in df_truth that hold entity IDs
    for source1 and source2 respectively.

    Candidate pairs are COUNTED per block (not materialised in memory)
    to avoid MemoryError on large datasets.
    """
    n1 = df_blocks[df_blocks["source"] == source1_name]["entity_id"].nunique()
    n2 = df_blocks[df_blocks["source"] == source2_name]["entity_id"].nunique()
    total_pairs = n1 * n2

    # ── Load true pairs ──────────────────────────────────────────────────
    true_pairs: set[tuple[str, str]] = set()
    if not df_truth.empty and truth_col_s1 and truth_col_s2:
        # Resolve actual column names (handle minor name variations)
        cols = df_truth.columns.tolist()
        col_a = next(
            (c for c in cols if truth_col_s1.lower() in c.lower()),
            cols[0]
        )
        col_b = next(
            (c for c in cols if truth_col_s2.lower() in c.lower()),
            cols[1]
        )
        true_pairs = set(
            zip(df_truth[col_a].astype(str), df_truth[col_b].astype(str))
        )
        true_s1_ids = {a for a, _ in true_pairs}
        true_s2_ids = {b for _, b in true_pairs}
    else:
        true_s1_ids: set[str] = set()
        true_s2_ids: set[str] = set()

    # ── Count candidate pairs + collect only PC-relevant pairs ───────────
    # Track UNIQUE cross-source pairs explicitly.
    # A pair may co-occur in many blocks, but it must only contribute once
    # to the candidate count and blocking quality metrics.
    seen_pairs: set[tuple[str, str]] = set()
    n_candidates = 0
    candidate_pairs_for_pc: set[tuple[str, str]] = set()

    for _, grp in df_blocks.groupby("block_id"):
        s1_ids = list(grp[grp["source"] == source1_name]["entity_id"])
        s2_ids = list(grp[grp["source"] == source2_name]["entity_id"])

        for a in s1_ids:
            for b in s2_ids:
                pair = (str(a), str(b))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    n_candidates += 1

        if true_pairs:
            s1_set = set(s1_ids)
            s2_set = set(s2_ids)
            for a in s1_set & true_s1_ids:
                for b in s2_set & true_s2_ids:
                    candidate_pairs_for_pc.add((a, b))

    rr = 1.0 - (n_candidates / total_pairs) if total_pairs else 0.0
    pc, pq = None, None

    if true_pairs:
        found = true_pairs & candidate_pairs_for_pc
        pc    = len(found) / len(true_pairs) if true_pairs else 0.0
        pq    = len(found) / n_candidates    if n_candidates else 0.0

    stats = {
        "total_entities_s1":  n1,
        "total_entities_s2":  n2,
        "brute_force_pairs":  total_pairs,
        "candidate_pairs":    n_candidates,
        "reduction_ratio":    round(rr, 4),
        "pairs_completeness": round(pc, 4) if pc is not None else "N/A",
        "pairs_quality":      round(pq, 4) if pq is not None else "N/A",
        "true_pairs":         len(true_pairs) if true_pairs else "N/A",
    }

    print("\n  [STATS]")
    for k, v in stats.items():
        print(f"    {k:30s}: {v}")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pipeline.data_ingestion import run_ingestion, DATASET_REGISTRY

    valid = list(DATASET_REGISTRY.keys()) + ["all"]
    parser = argparse.ArgumentParser(description="Member 1 — Token Blocking")
    parser.add_argument("--dataset", choices=valid, default="all")
    parser.add_argument("--max-block-size", type=int, default=1000)
    args = parser.parse_args()

    datasets = (list(DATASET_REGISTRY.keys())
                if args.dataset == "all" else [args.dataset])

    for ds_key in datasets:
        cfg = DATASET_REGISTRY[ds_key]
        print(f"\n{'='*60}\n  {ds_key.upper()}\n{'='*60}")

        df1, df2, df_truth = run_ingestion(dataset=ds_key)

        t0 = time.time()
        df_blocks = run_token_blocking(
            df1, df2,
            max_block_size=args.max_block_size,
            output_path=None,
        )
        print(f"  [TIME] Blocking (algo only): {time.time()-t0:.1f}s")

        out = os.path.join(OUTPUT_DIR, ds_key, "blocks.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df_blocks.to_csv(out, index=False)
        print(f"  [SAVE] {out}")

        tc1, tc2 = cfg.get("truth_col_s1", ""), cfg.get("truth_col_s2", "")
        compute_blocking_stats(
            df_blocks, df_truth,
            source1_name=cfg["source1"],
            source2_name=cfg["source2"],
            truth_col_s1=tc1,
            truth_col_s2=tc2,
        )
