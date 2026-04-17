"""Tests for clustering helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.clustering import materialize_incremental_clusters


def test_materialize_incremental_clusters_writes_checkpoint_files(tmp_path: Path) -> None:
    matches_df = pd.DataFrame([
        {"id_A": "a1", "id_B": "b1", "final_score": 0.9, "is_match": 1},
        {"id_A": "a2", "id_B": "b2", "final_score": 0.8, "is_match": 1},
        {"id_A": "a3", "id_B": "b3", "final_score": 0.2, "is_match": 0},
    ])
    source1 = pd.DataFrame([
        {"id": "a1", "title": "paper one", "source": "left"},
        {"id": "a2", "title": "paper two", "source": "left"},
    ])
    source2 = pd.DataFrame([
        {"id": "b1", "title": "paper one", "source": "right"},
        {"id": "b2", "title": "paper two", "source": "right"},
    ])
    source1_path = tmp_path / "source1.csv"
    source2_path = tmp_path / "source2.csv"
    source1.to_csv(source1_path, index=False)
    source2.to_csv(source2_path, index=False)

    payload = materialize_incremental_clusters(
        matches_df=matches_df,
        source1_path=str(source1_path),
        source2_path=str(source2_path),
        output_dir=str(tmp_path),
        completed=False,
        processed_rows=2,
        total_rows=3,
        promote_final=False,
    )

    assert payload["confirmed_matches"] == 2
    assert payload["n_clusters"] == 2
    assert (tmp_path / "clusters_incremental.csv").exists()
    assert (tmp_path / "merged_entities_incremental.csv").exists()
    assert (tmp_path / "cluster_status.json").exists()
    assert not (tmp_path / "clusters.csv").exists()

    status = json.loads((tmp_path / "cluster_status.json").read_text(encoding="utf-8"))
    assert status["completed"] is False
    assert status["processed_rows"] == 2
    assert status["total_rows"] == 3


def test_materialize_incremental_clusters_promotes_final_outputs(tmp_path: Path) -> None:
    matches_df = pd.DataFrame([
        {"id_A": "a1", "id_B": "b1", "final_score": 0.9, "is_match": 1},
    ])
    source = pd.DataFrame([{"id": "a1", "title": "x", "source": "left"}])
    source_other = pd.DataFrame([{"id": "b1", "title": "x", "source": "right"}])
    source1_path = tmp_path / "source1.csv"
    source2_path = tmp_path / "source2.csv"
    source.to_csv(source1_path, index=False)
    source_other.to_csv(source2_path, index=False)

    materialize_incremental_clusters(
        matches_df=matches_df,
        source1_path=str(source1_path),
        source2_path=str(source2_path),
        output_dir=str(tmp_path),
        completed=True,
        processed_rows=1,
        total_rows=1,
        promote_final=True,
    )

    assert (tmp_path / "clusters.csv").exists()
    assert (tmp_path / "merged_entities.csv").exists()
