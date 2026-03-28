"""Tests for Member 2 block processing."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from block_processing import (
    load_blocks_csv,
    meta_blocking_candidate_pairs,
    purge_oversized_blocks,
    run_block_processing,
)


def test_load_blocks_csv_requires_columns():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("a,b\n1,2\n")
        path = f.name
    try:
        with pytest.raises(ValueError, match="missing columns"):
            load_blocks_csv(path)
    finally:
        os.unlink(path)


def test_purge_removes_oversized_block():
    # Block "A" has 6 rows (>5); block "B" has 2 rows — only B survives.
    df = pd.DataFrame(
        {
            "block_id": ["A"] * 6 + ["B", "B"],
            "entity_id": [f"e{i}" for i in range(8)],
            "source": ["s1" if i % 2 == 0 else "s2" for i in range(8)],
        }
    )
    out, stats = purge_oversized_blocks(df, max_block_size=5, verbose=False)
    assert set(out["block_id"].unique()) == {"B"}
    assert stats["oversized_blocks_removed"] == 1
    assert stats["blocks_after"] < stats["blocks_before"]


def test_meta_blocking_jaccard_filters():
    # Two blocks: (a1,b1) only in block x; (a1,b2) in x and y -> higher Jaccard for a1-b1?
    # a1 in {x}, b1 in {x} -> Jaccard 1.0
    # a1 in {x}, b2 in {x,y} if b2 only in y: actually craft explicit sets
    df = pd.DataFrame(
        [
            {"block_id": "b1", "entity_id": "a1", "source": "s1"},
            {"block_id": "b1", "entity_id": "x1", "source": "s2"},
            {"block_id": "b2", "entity_id": "a1", "source": "s1"},
            {"block_id": "b2", "entity_id": "x2", "source": "s2"},
            {"block_id": "b3", "entity_id": "x1", "source": "s2"},
            {"block_id": "b3", "entity_id": "x2", "source": "s2"},
        ]
    )
    pairs, G, st = meta_blocking_candidate_pairs(
        df, "s1", "s2", min_jaccard=0.0, verbose=False
    )
    assert list(pairs.columns) == ["id_A", "id_B"]
    assert st["pairs_after_blocking_before_jaccard"] >= 1

    pairs_hi, _, st2 = meta_blocking_candidate_pairs(
        df, "s1", "s2", min_jaccard=0.99, verbose=False
    )
    assert st2["pairs_after_jaccard"] <= st["pairs_after_jaccard"]


def test_run_block_processing_writes_csv():
    df = pd.DataFrame(
        [
            {"block_id": "z", "entity_id": "p1", "source": "s1"},
            {"block_id": "z", "entity_id": "q1", "source": "s2"},
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        blocks = os.path.join(tmp, "blocks.csv")
        out_csv = os.path.join(tmp, "candidate_pairs.csv")
        stats_json = os.path.join(tmp, "member2_stats.json")
        df.to_csv(blocks, index=False)
        result, stats = run_block_processing(
            blocks,
            out_csv,
            "s1",
            "s2",
            max_block_size=100,
            min_jaccard=0.0,
            verbose=False,
            write_stats_json=stats_json,
        )
        assert os.path.isfile(out_csv)
        assert os.path.isfile(stats_json)
        assert len(result) == 1
        assert stats["candidate_pairs"] == 1
        assert stats["cartesian_pairs"] == 1
