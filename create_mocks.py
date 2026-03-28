"""
Member 1 - Mock File Generator
================================
Creates small, realistic mock CSV files for ALL pipeline stages,
adapted for the three datasets:
  - abt_buy
  - amazon_google
  - spimbench

DATA CONTRACT:
  is_match is always int: 1 (match) or 0 (non-match).
  Filter with df[df["is_match"] == 1], never == True.

Run on Day 1:
    python create_mocks.py
"""

from __future__ import annotations

import os
import pandas as pd

MOCK_DIR = os.path.join("output", "mock")


# ══════════════════════════════════════════════════════════════════════════════
# Generic helpers — used by all three datasets
# ══════════════════════════════════════════════════════════════════════════════

def _match_results(pairs: list[tuple], scores: list[float],
                   threshold: float = 0.50) -> pd.DataFrame:
    return pd.DataFrame(
        [{"id_A": a, "id_B": b,
          "similarity_score": s,
          "is_match": 1 if s >= threshold else 0}
         for (a, b), s in zip(pairs, scores)],
        columns=["id_A", "id_B", "similarity_score", "is_match"]
    )


SCORE_OFFSETS = {
    "jaccard":  0.00,
    "tfidf":   -0.03,
    "sbert":   +0.05,
    "combined": +0.01,
}


# ══════════════════════════════════════════════════════════════════════════════
# ABT-BUY mocks
# ══════════════════════════════════════════════════════════════════════════════

def _abt_buy_blocks() -> pd.DataFrame:
    rows = [
        {"block_id": "laptop",    "entity_id": "abt_001", "source": "abt"},
        {"block_id": "laptop",    "entity_id": "abt_002", "source": "abt"},
        {"block_id": "laptop",    "entity_id": "buy_001", "source": "buy"},
        {"block_id": "laptop",    "entity_id": "buy_002", "source": "buy"},
        {"block_id": "dell",      "entity_id": "abt_001", "source": "abt"},
        {"block_id": "dell",      "entity_id": "buy_001", "source": "buy"},
        {"block_id": "inspiron",  "entity_id": "abt_001", "source": "abt"},
        {"block_id": "inspiron",  "entity_id": "buy_001", "source": "buy"},
        {"block_id": "samsung",   "entity_id": "abt_003", "source": "abt"},
        {"block_id": "samsung",   "entity_id": "buy_003", "source": "buy"},
    ]
    return pd.DataFrame(rows, columns=["block_id", "entity_id", "source"])


def _abt_buy_candidate_pairs() -> pd.DataFrame:
    rows = [
        {"id_A": "abt_001", "id_B": "buy_001"},
        {"id_A": "abt_001", "id_B": "buy_002"},
        {"id_A": "abt_002", "id_B": "buy_001"},
        {"id_A": "abt_002", "id_B": "buy_002"},
        {"id_A": "abt_003", "id_B": "buy_003"},
        {"id_A": "abt_003", "id_B": "buy_004"},
    ]
    return pd.DataFrame(rows, columns=["id_A", "id_B"])


_ABT_BUY_BASE_SCORES = [0.85, 0.20, 0.15, 0.72, 0.90, 0.12]
_ABT_BUY_PAIRS = [
    ("abt_001", "buy_001"), ("abt_001", "buy_002"),
    ("abt_002", "buy_001"), ("abt_002", "buy_002"),
    ("abt_003", "buy_003"), ("abt_003", "buy_004"),
]


def _abt_buy_match_results(variant: str) -> pd.DataFrame:
    offset = SCORE_OFFSETS.get(variant, 0.0)
    scores = [min(1.0, max(0.0, s + offset)) for s in _ABT_BUY_BASE_SCORES]
    return _match_results(_ABT_BUY_PAIRS, scores)


def _abt_buy_clusters() -> pd.DataFrame:
    rows = [
        {"cluster_id": 0, "entity_id": "abt_001", "source": "abt"},
        {"cluster_id": 0, "entity_id": "buy_001", "source": "buy"},
        {"cluster_id": 1, "entity_id": "abt_002", "source": "abt"},
        {"cluster_id": 1, "entity_id": "buy_002", "source": "buy"},
        {"cluster_id": 2, "entity_id": "abt_003", "source": "abt"},
        {"cluster_id": 2, "entity_id": "buy_003", "source": "buy"},
        {"cluster_id": 3, "entity_id": "buy_004",  "source": "buy"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "entity_id", "source"])


def _abt_buy_merged_entities() -> pd.DataFrame:
    rows = [
        {"cluster_id": 0, "attribute_name": "name",  "merged_value": "dell inspiron 15 laptop"},
        {"cluster_id": 0, "attribute_name": "price", "merged_value": "499.99"},
        {"cluster_id": 1, "attribute_name": "name",  "merged_value": "hp pavilion 15"},
        {"cluster_id": 1, "attribute_name": "price", "merged_value": "549.00"},
        {"cluster_id": 2, "attribute_name": "name",  "merged_value": "samsung galaxy tab s7"},
        {"cluster_id": 2, "attribute_name": "price", "merged_value": "649.99"},
        {"cluster_id": 3, "attribute_name": "name",  "merged_value": "lenovo ideapad 3"},
        {"cluster_id": 3, "attribute_name": "price", "merged_value": "379.00"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "attribute_name", "merged_value"])


# ══════════════════════════════════════════════════════════════════════════════
# AMAZON-GOOGLE mocks
# ══════════════════════════════════════════════════════════════════════════════

def _amazon_google_blocks() -> pd.DataFrame:
    rows = [
        {"block_id": "laptop",   "entity_id": "amazon_001", "source": "amazon"},
        {"block_id": "laptop",   "entity_id": "amazon_002", "source": "amazon"},
        {"block_id": "laptop",   "entity_id": "google_001", "source": "google"},
        {"block_id": "dell",     "entity_id": "amazon_001", "source": "amazon"},
        {"block_id": "dell",     "entity_id": "google_001", "source": "google"},
        {"block_id": "inspiron", "entity_id": "amazon_001", "source": "amazon"},
        {"block_id": "inspiron", "entity_id": "google_001", "source": "google"},
        {"block_id": "samsung",  "entity_id": "amazon_003", "source": "amazon"},
        {"block_id": "samsung",  "entity_id": "google_003", "source": "google"},
    ]
    return pd.DataFrame(rows, columns=["block_id", "entity_id", "source"])


def _amazon_google_candidate_pairs() -> pd.DataFrame:
    rows = [
        {"id_A": "amazon_001", "id_B": "google_001"},
        {"id_A": "amazon_001", "id_B": "google_002"},
        {"id_A": "amazon_002", "id_B": "google_001"},
        {"id_A": "amazon_002", "id_B": "google_002"},
        {"id_A": "amazon_003", "id_B": "google_003"},
        {"id_A": "amazon_003", "id_B": "google_004"},
    ]
    return pd.DataFrame(rows, columns=["id_A", "id_B"])


_AMAZON_GOOGLE_BASE_SCORES = [0.82, 0.18, 0.14, 0.75, 0.91, 0.10]
_AMAZON_GOOGLE_PAIRS = [
    ("amazon_001", "google_001"), ("amazon_001", "google_002"),
    ("amazon_002", "google_001"), ("amazon_002", "google_002"),
    ("amazon_003", "google_003"), ("amazon_003", "google_004"),
]


def _amazon_google_match_results(variant: str) -> pd.DataFrame:
    offset = SCORE_OFFSETS.get(variant, 0.0)
    scores = [min(1.0, max(0.0, s + offset)) for s in _AMAZON_GOOGLE_BASE_SCORES]
    return _match_results(_AMAZON_GOOGLE_PAIRS, scores)


def _amazon_google_clusters() -> pd.DataFrame:
    rows = [
        {"cluster_id": 0, "entity_id": "amazon_001", "source": "amazon"},
        {"cluster_id": 0, "entity_id": "google_001", "source": "google"},
        {"cluster_id": 1, "entity_id": "amazon_002", "source": "amazon"},
        {"cluster_id": 1, "entity_id": "google_002", "source": "google"},
        {"cluster_id": 2, "entity_id": "amazon_003", "source": "amazon"},
        {"cluster_id": 2, "entity_id": "google_003", "source": "google"},
        {"cluster_id": 3, "entity_id": "google_004",  "source": "google"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "entity_id", "source"])


def _amazon_google_merged_entities() -> pd.DataFrame:
    rows = [
        {"cluster_id": 0, "attribute_name": "title", "merged_value": "dell inspiron 15 laptop"},
        {"cluster_id": 0, "attribute_name": "price", "merged_value": "499.99"},
        {"cluster_id": 1, "attribute_name": "title", "merged_value": "hp pavilion 15"},
        {"cluster_id": 1, "attribute_name": "price", "merged_value": "549.00"},
        {"cluster_id": 2, "attribute_name": "title", "merged_value": "samsung galaxy tab s7"},
        {"cluster_id": 2, "attribute_name": "price", "merged_value": "649.99"},
        {"cluster_id": 3, "attribute_name": "title", "merged_value": "google pixel 6 pro"},
        {"cluster_id": 3, "attribute_name": "price", "merged_value": "899.00"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "attribute_name", "merged_value"])


# ══════════════════════════════════════════════════════════════════════════════
# SPIMBENCH mocks
# ══════════════════════════════════════════════════════════════════════════════

def _spimbench_blocks() -> pd.DataFrame:
    rows = [
        {"block_id": "kubrick",  "entity_id": "spim_a_001", "source": "spimbench_a"},
        {"block_id": "kubrick",  "entity_id": "spim_b_001", "source": "spimbench_b"},
        {"block_id": "director", "entity_id": "spim_a_001", "source": "spimbench_a"},
        {"block_id": "director", "entity_id": "spim_b_002", "source": "spimbench_b"},
        {"block_id": "london",   "entity_id": "spim_a_002", "source": "spimbench_a"},
        {"block_id": "london",   "entity_id": "spim_b_003", "source": "spimbench_b"},
        {"block_id": "formula1", "entity_id": "spim_a_003", "source": "spimbench_a"},
        {"block_id": "formula1", "entity_id": "spim_b_004", "source": "spimbench_b"},
    ]
    return pd.DataFrame(rows, columns=["block_id", "entity_id", "source"])


def _spimbench_candidate_pairs() -> pd.DataFrame:
    rows = [
        {"id_A": "spim_a_001", "id_B": "spim_b_001"},
        {"id_A": "spim_a_001", "id_B": "spim_b_002"},
        {"id_A": "spim_a_002", "id_B": "spim_b_003"},
        {"id_A": "spim_a_002", "id_B": "spim_b_004"},
        {"id_A": "spim_a_003", "id_B": "spim_b_004"},
        {"id_A": "spim_a_003", "id_B": "spim_b_001"},
    ]
    return pd.DataFrame(rows, columns=["id_A", "id_B"])


_SPIMBENCH_BASE_SCORES = [0.88, 0.22, 0.86, 0.17, 0.79, 0.13]
_SPIMBENCH_PAIRS = [
    ("spim_a_001", "spim_b_001"), ("spim_a_001", "spim_b_002"),
    ("spim_a_002", "spim_b_003"), ("spim_a_002", "spim_b_004"),
    ("spim_a_003", "spim_b_004"), ("spim_a_003", "spim_b_001"),
]


def _spimbench_match_results(variant: str) -> pd.DataFrame:
    offset = SCORE_OFFSETS.get(variant, 0.0)
    scores = [min(1.0, max(0.0, s + offset)) for s in _SPIMBENCH_BASE_SCORES]
    return _match_results(_SPIMBENCH_PAIRS, scores)


def _spimbench_clusters() -> pd.DataFrame:
    rows = [
        {"cluster_id": 0, "entity_id": "spim_a_001", "source": "spimbench_a"},
        {"cluster_id": 0, "entity_id": "spim_b_001", "source": "spimbench_b"},
        {"cluster_id": 1, "entity_id": "spim_a_002", "source": "spimbench_a"},
        {"cluster_id": 1, "entity_id": "spim_b_003", "source": "spimbench_b"},
        {"cluster_id": 2, "entity_id": "spim_a_003", "source": "spimbench_a"},
        {"cluster_id": 2, "entity_id": "spim_b_004", "source": "spimbench_b"},
        {"cluster_id": 3, "entity_id": "spim_b_002",  "source": "spimbench_b"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "entity_id", "source"])


def _spimbench_merged_entities() -> pd.DataFrame:
    rows = [
        {"cluster_id": 0, "attribute_name": "label",    "merged_value": "stanley kubrick"},
        {"cluster_id": 0, "attribute_name": "birthdate","merged_value": "1928 07 26"},
        {"cluster_id": 1, "attribute_name": "label",    "merged_value": "city of london"},
        {"cluster_id": 1, "attribute_name": "country",  "merged_value": "united kingdom"},
        {"cluster_id": 2, "attribute_name": "label",    "merged_value": "formula one world championship"},
        {"cluster_id": 2, "attribute_name": "sport",    "merged_value": "formula 1"},
        {"cluster_id": 3, "attribute_name": "label",    "merged_value": "arsenal fc"},
        {"cluster_id": 3, "attribute_name": "sport",    "merged_value": "football"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "attribute_name", "merged_value"])


# ══════════════════════════════════════════════════════════════════════════════
# Generator — one subfolder per dataset
# ══════════════════════════════════════════════════════════════════════════════

DATASET_MOCKS = {
    "abt_buy": {
        "blocks":          _abt_buy_blocks,
        "candidate_pairs": _abt_buy_candidate_pairs,
        "match_results":   _abt_buy_match_results,
        "clusters":        _abt_buy_clusters,
        "merged_entities": _abt_buy_merged_entities,
    },
    "amazon_google": {
        "blocks":          _amazon_google_blocks,
        "candidate_pairs": _amazon_google_candidate_pairs,
        "match_results":   _amazon_google_match_results,
        "clusters":        _amazon_google_clusters,
        "merged_entities": _amazon_google_merged_entities,
    },
    "spimbench": {
        "blocks":          _spimbench_blocks,
        "candidate_pairs": _spimbench_candidate_pairs,
        "match_results":   _spimbench_match_results,
        "clusters":        _spimbench_clusters,
        "merged_entities": _spimbench_merged_entities,
    },
}


def generate_all_mocks(root_dir: str = MOCK_DIR) -> None:
    """
    Generate mock files for all three datasets.
    Output layout:
      output/mock/abt_buy/blocks.csv
      output/mock/abt_buy/candidate_pairs.csv
      output/mock/abt_buy/match_results_jaccard.csv
      ...  (same structure for amazon_google and spimbench)
    """
    for dataset, fns in DATASET_MOCKS.items():
        ds_dir = os.path.join(root_dir, dataset)
        os.makedirs(ds_dir, exist_ok=True)

        # blocks.csv
        df = fns["blocks"]()
        p  = os.path.join(ds_dir, "blocks.csv")
        df.to_csv(p, index=False)
        print(f"[MOCK] {p}  ({len(df)} rows)")

        # candidate_pairs.csv
        df = fns["candidate_pairs"]()
        p  = os.path.join(ds_dir, "candidate_pairs.csv")
        df.to_csv(p, index=False)
        print(f"[MOCK] {p}  ({len(df)} rows)")

        # match_results_<variant>.csv
        for variant in ["jaccard", "tfidf", "sbert", "combined"]:
            df = fns["match_results"](variant)
            p  = os.path.join(ds_dir, f"match_results_{variant}.csv")
            df.to_csv(p, index=False)
            print(f"[MOCK] {p}  ({len(df)} rows)")

        # clusters.csv
        df = fns["clusters"]()
        p  = os.path.join(ds_dir, "clusters.csv")
        df.to_csv(p, index=False)
        print(f"[MOCK] {p}  ({len(df)} rows)")

        # merged_entities.csv
        df = fns["merged_entities"]()
        p  = os.path.join(ds_dir, "merged_entities.csv")
        df.to_csv(p, index=False)
        print(f"[MOCK] {p}  ({len(df)} rows)")

        print()

    print(f"[MOCK] All mock files saved under: {root_dir}/")
    print("[MOCK] Push output/mock/ to Git so your teammates can start now.")


if __name__ == "__main__":
    generate_all_mocks()