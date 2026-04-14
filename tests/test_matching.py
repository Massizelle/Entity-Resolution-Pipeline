"""Tests for Member 3 entity matching."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from pipeline.matching import (
    jaccard_sim,
    tfidf_sim,
    sbert_sim,
    compute_value_sim,
    build_neighbor_index,
    compute_neighbor_sim,
    combine_scores,
    evaluate,
    load_and_prepare,
    _load_candidate_pairs,
)
from pipeline.adaptive_rescue import profile_matching_surface


# ─── ÉTAPE 1 — load_and_prepare ───────────────────────────────────────────────

def test_load_and_prepare_returns_text_columns():
    with tempfile.TemporaryDirectory() as tmp:
        pairs = pd.DataFrame({"id_A": ["1"], "id_B": ["2"]})
        s1 = pd.DataFrame({"id": ["1"], "name": ["laptop dell"], "source": ["abt"]})
        s2 = pd.DataFrame({"id": ["2"], "name": ["dell laptop"], "source": ["buy"]})

        pairs_path = os.path.join(tmp, "pairs.csv")
        s1_path = os.path.join(tmp, "s1.csv")
        s2_path = os.path.join(tmp, "s2.csv")
        pairs.to_csv(pairs_path, index=False)
        s1.to_csv(s1_path, index=False)
        s2.to_csv(s2_path, index=False)

        result = load_and_prepare(pairs_path, s1_path, s2_path)
        assert "text_A" in result.columns
        assert "text_B" in result.columns
        assert len(result) == 1


# ─── ÉTAPE 2 — Value Similarity ───────────────────────────────────────────────

def test_jaccard_sim_identical():
    assert jaccard_sim("hello world", "hello world") == 1.0


def test_jaccard_sim_disjoint():
    assert jaccard_sim("hello world", "foo bar") == 0.0


def test_jaccard_sim_partial():
    score = jaccard_sim("hello world", "hello python")
    assert 0.0 < score < 1.0


def test_jaccard_sim_empty():
    assert jaccard_sim("", "") == 0


def test_tfidf_sim_identical():
    texts = ["laptop dell inspiron"]
    scores = tfidf_sim(texts, texts)
    assert len(scores) == 1
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_tfidf_sim_length():
    texts_a = ["laptop dell", "apple iphone", "samsung tv"]
    texts_b = ["dell laptop", "iphone apple", "tv samsung"]
    scores = tfidf_sim(texts_a, texts_b)
    assert len(scores) == 3


def test_sbert_sim_identical():
    texts = ["laptop dell inspiron"]
    scores = sbert_sim(texts, texts)
    assert len(scores) == 1
    assert scores[0] == pytest.approx(1.0, abs=1e-3)


def test_sbert_sim_range():
    texts_a = ["laptop dell"]
    texts_b = ["completely different text about cooking"]
    scores = sbert_sim(texts_a, texts_b)
    assert 0.0 <= scores[0] <= 1.0


# ─── ÉTAPE 3 — compute_value_sim ──────────────────────────────────────────────

def test_compute_value_sim_columns():
    df = pd.DataFrame({
        "id_A": ["1", "2"],
        "id_B": ["3", "4"],
        "text_A": ["laptop dell", "apple iphone"],
        "text_B": ["dell laptop", "samsung galaxy"],
    })
    result = compute_value_sim(df)
    for col in ["id_A", "id_B", "jaccard_score", "tfidf_score", "sbert_score", "value_score", "is_match"]:
        assert col in result.columns


def test_compute_value_sim_is_match_is_int():
    df = pd.DataFrame({
        "id_A": ["1"],
        "id_B": ["2"],
        "text_A": ["laptop dell"],
        "text_B": ["dell laptop"],
    })
    result = compute_value_sim(df)
    assert result["is_match"].dtype in [int, "int64", "int32"]
    assert result["is_match"].iloc[0] in [0, 1]


# ─── ÉTAPE 4 — Neighbor Similarity ────────────────────────────────────────────

def test_build_neighbor_index_cross_source_only():
    blocks_df = pd.DataFrame([
        {"block_id": "b1", "entity_id": "a1", "source": "s1"},
        {"block_id": "b1", "entity_id": "b1", "source": "s2"},
        {"block_id": "b1", "entity_id": "a2", "source": "s1"},
    ])
    index = build_neighbor_index(blocks_df)
    # a1 et a2 sont dans s1 → pas voisins entre eux
    assert "b1" in index.get("a1", set())
    assert "a2" not in index.get("a1", set())


def test_compute_neighbor_sim_returns_dataframe():
    blocks_df = pd.DataFrame([
        {"block_id": "b1", "entity_id": "a1", "source": "s1"},
        {"block_id": "b1", "entity_id": "b1", "source": "s2"},
    ])
    provisional = pd.DataFrame({"id_A": ["a1"], "id_B": ["b1"]})
    result = compute_neighbor_sim(blocks_df, provisional)
    assert list(result.columns) == ["id_A", "id_B", "neighbor_score"]
    assert len(result) == 1
    assert 0.0 <= result["neighbor_score"].iloc[0] <= 1.0


# ─── ÉTAPE 5 — combine_scores ─────────────────────────────────────────────────

def test_combine_scores_columns():
    value_df = pd.DataFrame({
        "id_A": ["a1", "a2"],
        "id_B": ["b1", "b2"],
        "jaccard_score": [0.8, 0.1],
        "tfidf_score": [0.7, 0.1],
        "sbert_score": [0.9, 0.1],
        "value_score": [0.8, 0.1],
        "is_match": [1, 0],
    })
    blocks_df = pd.DataFrame([
        {"block_id": "b1", "entity_id": "a1", "source": "s1"},
        {"block_id": "b1", "entity_id": "b1", "source": "s2"},
        {"block_id": "b1", "entity_id": "a2", "source": "s1"},
        {"block_id": "b1", "entity_id": "b2", "source": "s2"},
    ])
    result = combine_scores(value_df, blocks_df)
    for col in ["id_A", "id_B", "final_score", "is_match"]:
        assert col in result.columns


def test_combine_scores_is_match_is_int():
    value_df = pd.DataFrame({
        "id_A": ["a1"],
        "id_B": ["b1"],
        "jaccard_score": [0.9],
        "tfidf_score": [0.9],
        "sbert_score": [0.9],
        "value_score": [0.9],
        "is_match": [1],
    })
    blocks_df = pd.DataFrame([
        {"block_id": "b1", "entity_id": "a1", "source": "s1"},
        {"block_id": "b1", "entity_id": "b1", "source": "s2"},
    ])
    result = combine_scores(value_df, blocks_df)
    assert result["is_match"].iloc[0] in [0, 1]


def test_combine_scores_does_not_lower_strong_value_match():
    value_df = pd.DataFrame({
        "id_A": ["a1", "a1", "a2"],
        "id_B": ["b1", "b2", "b3"],
        "certificate_score": [4.0, 0.2, 0.1],
        "jaccard_score": [0.9, 0.2, 0.1],
        "tfidf_score": [0.9, 0.2, 0.1],
        "sbert_score": [0.9, 0.2, 0.1],
        "value_score": [0.9, 0.2, 0.1],
        "is_match": [1, 0, 0],
    })
    result = combine_scores(value_df, pd.DataFrame())
    row = result[result["id_B"] == "b1"].iloc[0]
    assert row["final_score"] >= 0.9
    assert row["is_match"] == 1


def test_combine_scores_uses_mutual_best_and_certificate_support():
    value_df = pd.DataFrame({
        "id_A": ["a1", "a1", "a2"],
        "id_B": ["b1", "b2", "b2"],
        "certificate_score": [3.0, 0.1, 0.1],
        "jaccard_score": [0.48, 0.47, 0.30],
        "tfidf_score": [0.48, 0.47, 0.30],
        "sbert_score": [0.48, 0.47, 0.30],
        "value_score": [0.48, 0.47, 0.30],
        "is_match": [0, 0, 0],
    })
    result = combine_scores(value_df, pd.DataFrame())
    best = result[result["id_B"] == "b1"].iloc[0]
    assert best["mutual_best_bonus"] == pytest.approx(1.0)
    assert best["final_score"] > best["value_score"]


def test_profile_matching_surface_activates_on_ambiguous_surface():
    df = pd.DataFrame({
        "id_A": [f"a{i}" for i in range(10)],
        "id_B": [f"b{i}" for i in range(10)],
        "value_score": [0.41, 0.43, 0.47, 0.44, 0.49, 0.38, 0.40, 0.46, 0.52, 0.30],
        "final_score": [0.45, 0.48, 0.51, 0.49, 0.54, 0.42, 0.44, 0.53, 0.60, 0.20],
        "is_match": [0] * 10,
    })
    profile = profile_matching_surface(df)
    assert profile["activate"] is True


def test_profile_matching_surface_stays_off_on_cleanly_separated_surface():
    df = pd.DataFrame({
        "id_A": [f"a{i}" for i in range(6)],
        "id_B": [f"b{i}" for i in range(6)],
        "value_score": [0.95, 0.92, 0.88, 0.05, 0.08, 0.10],
        "final_score": [0.99, 0.97, 0.94, 0.06, 0.08, 0.10],
        "is_match": [1, 1, 1, 0, 0, 0],
    })
    profile = profile_matching_surface(df)
    assert profile["activate"] is False


def test_combine_scores_adds_adaptive_bonus_on_ambiguous_supported_pair():
    value_df = pd.DataFrame({
        "id_A": ["a1", "a1", "a2", "a2", "a3", "a3"],
        "id_B": ["b1", "b2", "b1", "b2", "b3", "b4"],
        "certificate_score": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "evidence_count": [4, 1, 4, 1, 1, 1],
        "witness_path": [
            "rare_pair:x|digit_signature:123|token_skeleton:abc",
            "block_id:z",
            "rare_pair:x|digit_signature:123|token_skeleton:abc",
            "block_id:z",
            "block_id:q",
            "block_id:r",
        ],
        "region_size": [2, 20, 2, 20, 25, 25],
        "jaccard_score": [0.39, 0.37, 0.44, 0.25, 0.36, 0.35],
        "tfidf_score": [0.40, 0.37, 0.46, 0.25, 0.38, 0.36],
        "sbert_score": [0.41, 0.37, 0.47, 0.25, 0.41, 0.38],
        "value_score": [0.40, 0.37, 0.456667, 0.25, 0.383333, 0.363333],
        "is_match": [0, 0, 0, 0, 0, 0],
    })
    result = combine_scores(value_df, pd.DataFrame())
    rescued = result[(result["id_A"] == "a1") & (result["id_B"] == "b1")].iloc[0]
    weak = result[(result["id_A"] == "a1") & (result["id_B"] == "b2")].iloc[0]

    assert bool(rescued["adaptive_rescue_active"]) is True
    assert rescued["adaptive_rescue_bonus"] > 0.0
    assert weak["adaptive_rescue_bonus"] == pytest.approx(0.0)



# ─── ÉTAPE 6 — evaluate ───────────────────────────────────────────────────────

def test_evaluate_returns_none_if_no_ground_truth():
    matches_df = pd.DataFrame({"id_A": ["a1"], "id_B": ["b1"], "is_match": [1]})
    result = evaluate(matches_df, "/nonexistent/path/ground_truth.csv")
    assert result is None


def test_evaluate_perfect_match():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("id_A,id_B\na1,b1\na2,b2\n")
        path = f.name
    try:
        matches_df = pd.DataFrame({
            "id_A": ["a1", "a2"],
            "id_B": ["b1", "b2"],
            "is_match": [1, 1],
        })
        precision, recall, f1 = evaluate(matches_df, path)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)
    finally:
        os.unlink(path)


def test_evaluate_casts_ground_truth_ids_to_strings():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("id_A,id_B\n1,2\n")
        path = f.name
    try:
        matches_df = pd.DataFrame({
            "id_A": ["1"],
            "id_B": ["2"],
            "is_match": [1],
        })
        precision, recall, f1 = evaluate(matches_df, path)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)
    finally:
        os.unlink(path)


def test_evaluate_no_match():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("id_A,id_B\na1,b1\n")
        path = f.name
    try:
        matches_df = pd.DataFrame({
            "id_A": ["a2"],
            "id_B": ["b2"],
            "is_match": [1],
        })
        precision, recall, f1 = evaluate(matches_df, path)
        assert precision == pytest.approx(0.0)
        assert recall == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)
    finally:
        os.unlink(path)


def test_evaluate_uses_only_rows_marked_as_match():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("id_A,id_B\na1,b1\n")
        path = f.name
    try:
        matches_df = pd.DataFrame(
            {
                "id_A": ["a1", "a2"],
                "id_B": ["b1", "b2"],
                "is_match": [1, 0],
            }
        )
        precision, recall, f1 = evaluate(matches_df, path)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)
    finally:
        os.unlink(path)


def test_load_candidate_pairs_sorts_by_certificate_score_for_cw_semantic_predictive():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("id_A,id_B,certificate_score\n")
        f.write("a1,b1,2.1\n")
        f.write("a2,b2,3.9\n")
        f.write("a3,b3,3.5\n")
        path = f.name
    try:
        df = _load_candidate_pairs(path, candidate_strategy="cw_semantic_predictive", limit=2)
        assert list(df["id_A"]) == ["a2", "a3"]
    finally:
        os.unlink(path)
