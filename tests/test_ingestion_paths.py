"""Regression tests for dataset-specific ingestion path handling."""

from __future__ import annotations

from pathlib import Path

from pipeline.data_ingestion import load_abt_buy


def test_load_abt_buy_accepts_lowercase_underscore_mapping_filename(tmp_path: Path):
    raw_dir = tmp_path / "Abt-Buy"
    raw_dir.mkdir()

    (raw_dir / "Abt.csv").write_text("id,name,description,price\n1,a,b,1.0\n", encoding="latin-1")
    (raw_dir / "Buy.csv").write_text(
        "id,name,description,price,manufacturer\n2,c,d,2.0,m\n",
        encoding="latin-1",
    )
    (raw_dir / "abt_buy_perfectMapping.csv").write_text(
        "idAbt,idBuy\n1,2\n",
        encoding="latin-1",
    )

    _, _, df_truth = load_abt_buy(str(raw_dir))

    assert len(df_truth) == 1
    assert list(df_truth.columns) == ["idAbt", "idBuy"]
