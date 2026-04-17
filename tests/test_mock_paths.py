"""Regression tests for mock-mode path handling."""

from __future__ import annotations

from pathlib import Path

from create_mocks import generate_all_mocks
from cli.run_member4 import _resolve_cleaned_source_paths
from pipeline.matching import _resolve_ground_truth_path


def test_generate_all_mocks_writes_ground_truth_for_supported_datasets(tmp_path: Path):
    generate_all_mocks(root_dir=str(tmp_path))

    assert (tmp_path / "abt_buy" / "ground_truth.csv").is_file()
    assert (tmp_path / "amazon_google" / "ground_truth.csv").is_file()
    assert not (tmp_path / "spimbench" / "ground_truth.csv").exists()


def test_resolve_ground_truth_path_prefers_mock_fixture_when_present(tmp_path: Path):
    mock_dataset_dir = tmp_path / "abt_buy"
    mock_dataset_dir.mkdir()
    truth_path = mock_dataset_dir / "ground_truth.csv"
    truth_path.write_text("id_A,id_B\nabt_001,buy_001\n", encoding="utf-8")

    resolved = _resolve_ground_truth_path("abt_buy", mock=True, prefix=str(tmp_path))

    assert resolved == str(truth_path)


def test_run_member4_mock_source_paths_use_mock_directory():
    source1, source2 = _resolve_cleaned_source_paths(
        "abt_buy",
        use_mock=True,
        prefix="output/mock/abt_buy",
    )

    assert source1 == "output/mock/abt_buy/cleaned_source1.csv"
    assert source2 == "output/mock/abt_buy/cleaned_source2.csv"
