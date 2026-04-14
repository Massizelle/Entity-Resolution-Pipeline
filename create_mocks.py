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


def _match_results_collective(pairs: list[tuple], scores: list[float],
                               threshold: float = 0.50) -> pd.DataFrame:
    """Collective match results: final_score column instead of similarity_score."""
    return pd.DataFrame(
        [{"id_A": a, "id_B": b,
          "final_score": s,
          "is_match": 1 if s >= threshold else 0}
         for (a, b), s in zip(pairs, scores)],
        columns=["id_A", "id_B", "final_score", "is_match"]
    )


SCORE_OFFSETS = {
    "jaccard":    0.00,
    "tfidf":     -0.03,
    "sbert":     +0.05,
    "combined":  +0.01,
    "collective": +0.02,
}


# ══════════════════════════════════════════════════════════════════════════════
# ABT-BUY mocks
# ══════════════════════════════════════════════════════════════════════════════

def _abt_buy_blocks() -> pd.DataFrame:
    # FIX: buy_004 added to "lenovo" block so it can legitimately appear
    # in candidate_pairs (entities must share at least one block).
    rows = [
        {"block_id": "laptop",   "entity_id": "abt_001", "source": "abt"},
        {"block_id": "laptop",   "entity_id": "abt_002", "source": "abt"},
        {"block_id": "laptop",   "entity_id": "buy_001", "source": "buy"},
        {"block_id": "laptop",   "entity_id": "buy_002", "source": "buy"},
        {"block_id": "dell",     "entity_id": "abt_001", "source": "abt"},
        {"block_id": "dell",     "entity_id": "buy_001", "source": "buy"},
        {"block_id": "inspiron", "entity_id": "abt_001", "source": "abt"},
        {"block_id": "inspiron", "entity_id": "buy_001", "source": "buy"},
        {"block_id": "samsung",  "entity_id": "abt_003", "source": "abt"},
        {"block_id": "samsung",  "entity_id": "buy_003", "source": "buy"},
        {"block_id": "lenovo",   "entity_id": "abt_003", "source": "abt"},  # FIX
        {"block_id": "lenovo",   "entity_id": "buy_004", "source": "buy"},  # FIX
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
    if variant == "collective":
        return _match_results_collective(_ABT_BUY_PAIRS, scores)
    return _match_results(_ABT_BUY_PAIRS, scores)


def _abt_buy_clusters() -> pd.DataFrame:
    # Matches: (abt_001,buy_001)=0.85, (abt_002,buy_002)=0.72, (abt_003,buy_003)=0.90
    # Isolated: buy_004 (score 0.12 → no match)
    rows = [
        {"cluster_id": 0, "entity_id": "abt_001"},
        {"cluster_id": 0, "entity_id": "buy_001"},
        {"cluster_id": 1, "entity_id": "abt_002"},
        {"cluster_id": 1, "entity_id": "buy_002"},
        {"cluster_id": 2, "entity_id": "abt_003"},
        {"cluster_id": 2, "entity_id": "buy_003"},
        {"cluster_id": 3, "entity_id": "buy_004"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "entity_id"])


def _abt_buy_source1() -> pd.DataFrame:
    """Cleaned Abt.com entities — mirrors real cleaned_source1.csv schema."""
    rows = [
        {"id": "abt_001", "name": "dell inspiron 15 laptop 8gb ram",
         "description": "dell inspiron 15 3000 series laptop intel core i5",
         "price": 499.99, "source": "abt"},
        {"id": "abt_002", "name": "hp pavilion 15 laptop",
         "description": "hp pavilion 15 inch laptop amd ryzen 5 processor",
         "price": 549.00, "source": "abt"},
        {"id": "abt_003", "name": "samsung galaxy tab s7 wifi",
         "description": "samsung galaxy tab s7 11 inch android tablet 128gb",
         "price": 649.99, "source": "abt"},
    ]
    return pd.DataFrame(rows, columns=["id", "name", "description", "price", "source"])


def _abt_buy_source2() -> pd.DataFrame:
    """Cleaned Buy.com entities — mirrors real cleaned_source2.csv schema."""
    rows = [
        {"id": "buy_001", "name": "dell inspiron 15 3000",
         "description": "dell inspiron 15 laptop intel i5 8gb 256gb ssd",
         "price": 489.00, "manufacturer": "dell",    "source": "buy"},
        {"id": "buy_002", "name": "dell inspiron laptop 15 inch",
         "description": "dell inspiron laptop 15 series windows 11",
         "price": 510.00, "manufacturer": "dell",    "source": "buy"},
        {"id": "buy_003", "name": "samsung tab s7 android tablet",
         "description": "samsung galaxy tab s7 128gb wifi snapdragon 865",
         "price": 639.00, "manufacturer": "samsung", "source": "buy"},
        {"id": "buy_004", "name": "lenovo ideapad 3 laptop",
         "description": "lenovo ideapad 3 15 inch laptop amd ryzen 3",
         "price": 379.00, "manufacturer": "lenovo",  "source": "buy"},
    ]
    return pd.DataFrame(rows, columns=["id", "name", "description", "price", "manufacturer", "source"])


def _abt_buy_merged_entities() -> pd.DataFrame:
    # FIX: wide format matching clustering.merge_cluster_attributes() output.
    # abt_buy has no 'title' column → title is empty; manufacturer comes from buy source.
    rows = [
        {"cluster_id": 0, "title": "",
         "description": "dell inspiron 15 laptop intel i5 8gb 256gb ssd",
         "manufacturer": "dell",    "price": 499.99},
        {"cluster_id": 1, "title": "",
         "description": "hp pavilion 15 inch laptop amd ryzen 5 processor",
         "manufacturer": "dell",    "price": 549.00},
        {"cluster_id": 2, "title": "",
         "description": "samsung galaxy tab s7 128gb wifi snapdragon 865",
         "manufacturer": "samsung", "price": 649.99},
        {"cluster_id": 3, "title": "",
         "description": "lenovo ideapad 3 15 inch laptop amd ryzen 3",
         "manufacturer": "lenovo",  "price": 379.00},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "title", "description", "manufacturer", "price"])


def _abt_buy_ground_truth() -> pd.DataFrame:
    rows = [
        {"id_A": "abt_001", "id_B": "buy_001"},
        {"id_A": "abt_002", "id_B": "buy_002"},
        {"id_A": "abt_003", "id_B": "buy_003"},
    ]
    return pd.DataFrame(rows, columns=["id_A", "id_B"])


# ══════════════════════════════════════════════════════════════════════════════
# AMAZON-GOOGLE mocks
# ══════════════════════════════════════════════════════════════════════════════

def _amazon_google_blocks() -> pd.DataFrame:
    # FIX: google_002 and google_004 added to blocks so they can legitimately
    # appear in candidate_pairs.
    rows = [
        {"block_id": "laptop",   "entity_id": "amazon_001", "source": "amazon"},
        {"block_id": "laptop",   "entity_id": "amazon_002", "source": "amazon"},
        {"block_id": "laptop",   "entity_id": "google_001", "source": "google"},
        {"block_id": "laptop",   "entity_id": "google_002", "source": "google"},  # FIX
        {"block_id": "dell",     "entity_id": "amazon_001", "source": "amazon"},
        {"block_id": "dell",     "entity_id": "google_001", "source": "google"},
        {"block_id": "inspiron", "entity_id": "amazon_001", "source": "amazon"},
        {"block_id": "inspiron", "entity_id": "google_001", "source": "google"},
        {"block_id": "samsung",  "entity_id": "amazon_003", "source": "amazon"},
        {"block_id": "samsung",  "entity_id": "google_003", "source": "google"},
        {"block_id": "pixel",    "entity_id": "amazon_003", "source": "amazon"},  # FIX
        {"block_id": "pixel",    "entity_id": "google_004", "source": "google"},  # FIX
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
    if variant == "collective":
        return _match_results_collective(_AMAZON_GOOGLE_PAIRS, scores)
    return _match_results(_AMAZON_GOOGLE_PAIRS, scores)


def _amazon_google_clusters() -> pd.DataFrame:
    # Matches: (amazon_001,google_001)=0.82, (amazon_002,google_002)=0.75,
    #          (amazon_003,google_003)=0.91
    # Isolated: google_004 (score 0.10 → no match)
    rows = [
        {"cluster_id": 0, "entity_id": "amazon_001"},
        {"cluster_id": 0, "entity_id": "google_001"},
        {"cluster_id": 1, "entity_id": "amazon_002"},
        {"cluster_id": 1, "entity_id": "google_002"},
        {"cluster_id": 2, "entity_id": "amazon_003"},
        {"cluster_id": 2, "entity_id": "google_003"},
        {"cluster_id": 3, "entity_id": "google_004"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "entity_id"])


def _amazon_google_source1() -> pd.DataFrame:
    """Cleaned Amazon entities — mirrors real cleaned_source1.csv schema."""
    rows = [
        {"id": "amazon_001", "title": "dell inspiron 15 laptop 8gb ram 256gb ssd",
         "description": "dell inspiron 15 3000 series laptop intel core i5 processor",
         "manufacturer": "dell",    "price": 499.99, "source": "amazon"},
        {"id": "amazon_002", "title": "hp pavilion 15 laptop amd ryzen 5",
         "description": "hp pavilion 15 inch laptop amd ryzen 5 8gb ram 512gb",
         "manufacturer": "hp",      "price": 549.00, "source": "amazon"},
        {"id": "amazon_003", "title": "samsung galaxy tab s7 11 inch wifi",
         "description": "samsung galaxy tab s7 android tablet 128gb snapdragon",
         "manufacturer": "samsung", "price": 649.99, "source": "amazon"},
    ]
    return pd.DataFrame(rows, columns=["id", "title", "description", "manufacturer", "price", "source"])


def _amazon_google_source2() -> pd.DataFrame:
    """Cleaned Google Shopping entities — mirrors real cleaned_source2.csv schema."""
    rows = [
        {"id": "google_001", "name": "dell inspiron 15 3000 laptop",
         "description": "dell inspiron 15 laptop intel i5 8gb 256gb ssd windows 11",
         "manufacturer": "dell",    "price": 489.00, "source": "google"},
        {"id": "google_002", "name": "dell inspiron laptop 15 inch i5",
         "description": "dell inspiron 15 series laptop windows 11 home",
         "manufacturer": "dell",    "price": 510.00, "source": "google"},
        {"id": "google_003", "name": "samsung galaxy tab s7 wifi 128gb",
         "description": "samsung tab s7 android 128gb snapdragon 865 plus",
         "manufacturer": "samsung", "price": 639.00, "source": "google"},
        {"id": "google_004", "name": "google pixel 6 pro 128gb",
         "description": "google pixel 6 pro android smartphone 128gb 5g",
         "manufacturer": "google",  "price": 899.00, "source": "google"},
    ]
    return pd.DataFrame(rows, columns=["id", "name", "description", "manufacturer", "price", "source"])


def _amazon_google_merged_entities() -> pd.DataFrame:
    # FIX: wide format matching clustering.merge_cluster_attributes() output.
    # title comes from amazon (source1), description picks the longest.
    rows = [
        {"cluster_id": 0, "title": "dell inspiron 15 laptop 8gb ram 256gb ssd",
         "description": "dell inspiron 15 laptop intel i5 8gb 256gb ssd windows 11",
         "manufacturer": "dell",    "price": 499.99},
        {"cluster_id": 1, "title": "hp pavilion 15 laptop amd ryzen 5",
         "description": "hp pavilion 15 inch laptop amd ryzen 5 8gb ram 512gb",
         "manufacturer": "hp",      "price": 549.00},
        {"cluster_id": 2, "title": "samsung galaxy tab s7 11 inch wifi",
         "description": "samsung tab s7 android 128gb snapdragon 865 plus",
         "manufacturer": "samsung", "price": 649.99},
        {"cluster_id": 3, "title": "",
         "description": "google pixel 6 pro android smartphone 128gb 5g",
         "manufacturer": "google",  "price": 899.00},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "title", "description", "manufacturer", "price"])


def _amazon_google_ground_truth() -> pd.DataFrame:
    rows = [
        {"id_A": "amazon_001", "id_B": "google_001"},
        {"id_A": "amazon_002", "id_B": "google_002"},
        {"id_A": "amazon_003", "id_B": "google_003"},
    ]
    return pd.DataFrame(rows, columns=["id_A", "id_B"])


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
        {"block_id": "arsenal",  "entity_id": "spim_a_003", "source": "spimbench_a"},  # FIX: spim_b_001 reachable
        {"block_id": "arsenal",  "entity_id": "spim_b_001", "source": "spimbench_b"},  # FIX
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
    if variant == "collective":
        return _match_results_collective(_SPIMBENCH_PAIRS, scores)
    return _match_results(_SPIMBENCH_PAIRS, scores)


def _spimbench_clusters() -> pd.DataFrame:
    # Matches: (spim_a_001,spim_b_001)=0.88, (spim_a_002,spim_b_003)=0.86,
    #          (spim_a_003,spim_b_004)=0.79
    # Isolated: spim_b_002 (score 0.22 → no match)
    rows = [
        {"cluster_id": 0, "entity_id": "spim_a_001"},
        {"cluster_id": 0, "entity_id": "spim_b_001"},
        {"cluster_id": 1, "entity_id": "spim_a_002"},
        {"cluster_id": 1, "entity_id": "spim_b_003"},
        {"cluster_id": 2, "entity_id": "spim_a_003"},
        {"cluster_id": 2, "entity_id": "spim_b_004"},
        {"cluster_id": 3, "entity_id": "spim_b_002"},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "entity_id"])


def _spimbench_source1() -> pd.DataFrame:
    """Cleaned SPIMBench source_a entities (odd-indexed TTL rows)."""
    rows = [
        {"id": "spim_a_001", "label": "stanley kubrick",
         "description": "american film director producer screenwriter",
         "source": "spimbench_a"},
        {"id": "spim_a_002", "label": "city of london",
         "description": "historic city square mile financial district england",
         "source": "spimbench_a"},
        {"id": "spim_a_003", "label": "formula one world championship",
         "description": "international single seater auto racing series fia",
         "source": "spimbench_a"},
    ]
    return pd.DataFrame(rows, columns=["id", "label", "description", "source"])


def _spimbench_source2() -> pd.DataFrame:
    """Cleaned SPIMBench source_b entities (even-indexed TTL rows)."""
    rows = [
        {"id": "spim_b_001", "label": "kubrick stanley",
         "description": "film director new york born cinema auteur",
         "source": "spimbench_b"},
        {"id": "spim_b_002", "label": "stanley kubrick director",
         "description": "director of 2001 space odyssey and the shining",
         "source": "spimbench_b"},
        {"id": "spim_b_003", "label": "london city",
         "description": "city of london uk financial centre one square mile",
         "source": "spimbench_b"},
        {"id": "spim_b_004", "label": "formula 1 championship",
         "description": "fia formula one season grand prix racing world title",
         "source": "spimbench_b"},
    ]
    return pd.DataFrame(rows, columns=["id", "label", "description", "source"])


def _spimbench_merged_entities() -> pd.DataFrame:
    # FIX: wide format matching clustering.merge_cluster_attributes() output.
    # SPIMBench TTL data has no title/manufacturer/price → those fields are empty.
    rows = [
        {"cluster_id": 0, "title": "", "description": "american film director producer screenwriter",
         "manufacturer": "", "price": 0.0},
        {"cluster_id": 1, "title": "", "description": "historic city square mile financial district england",
         "manufacturer": "", "price": 0.0},
        {"cluster_id": 2, "title": "", "description": "international single seater auto racing series fia",
         "manufacturer": "", "price": 0.0},
        {"cluster_id": 3, "title": "", "description": "director of 2001 space odyssey and the shining",
         "manufacturer": "", "price": 0.0},
    ]
    return pd.DataFrame(rows, columns=["cluster_id", "title", "description", "manufacturer", "price"])


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
        "ground_truth":    _abt_buy_ground_truth,
        "source1":         _abt_buy_source1,
        "source2":         _abt_buy_source2,
    },
    "amazon_google": {
        "blocks":          _amazon_google_blocks,
        "candidate_pairs": _amazon_google_candidate_pairs,
        "match_results":   _amazon_google_match_results,
        "clusters":        _amazon_google_clusters,
        "merged_entities": _amazon_google_merged_entities,
        "ground_truth":    _amazon_google_ground_truth,
        "source1":         _amazon_google_source1,
        "source2":         _amazon_google_source2,
    },
    "spimbench": {
        "blocks":          _spimbench_blocks,
        "candidate_pairs": _spimbench_candidate_pairs,
        "match_results":   _spimbench_match_results,
        "clusters":        _spimbench_clusters,
        "merged_entities": _spimbench_merged_entities,
        "source1":         _spimbench_source1,
        "source2":         _spimbench_source2,
    },
}


def generate_all_mocks(root_dir: str = MOCK_DIR) -> None:
    """
    Generate mock files for all three datasets.
    Output layout per dataset:
      output/mock/<dataset>/blocks.csv
      output/mock/<dataset>/candidate_pairs.csv
      output/mock/<dataset>/cleaned_source1.csv
      output/mock/<dataset>/cleaned_source2.csv
      output/mock/<dataset>/ground_truth.csv  (when the dataset has one)
      output/mock/<dataset>/match_results_{jaccard,tfidf,sbert,combined,collective}.csv
      output/mock/<dataset>/clusters.csv
      output/mock/<dataset>/merged_entities.csv
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

        # cleaned_source1.csv + cleaned_source2.csv
        df = fns["source1"]()
        p  = os.path.join(ds_dir, "cleaned_source1.csv")
        df.to_csv(p, index=False)
        print(f"[MOCK] {p}  ({len(df)} rows)")

        df = fns["source2"]()
        p  = os.path.join(ds_dir, "cleaned_source2.csv")
        df.to_csv(p, index=False)
        print(f"[MOCK] {p}  ({len(df)} rows)")

        if "ground_truth" in fns:
            df = fns["ground_truth"]()
            p  = os.path.join(ds_dir, "ground_truth.csv")
            df.to_csv(p, index=False)
            print(f"[MOCK] {p}  ({len(df)} rows)")

        # match_results_<variant>.csv  (jaccard, tfidf, sbert, combined, collective)
        for variant in ["jaccard", "tfidf", "sbert", "combined", "collective"]:
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
