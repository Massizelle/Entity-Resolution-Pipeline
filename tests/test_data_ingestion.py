"""Tests for dataset ingestion helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.data_ingestion import load_rexa_dblp, load_dblp_acm, run_ingestion, DATASET_REGISTRY


DBLP_RDF = """<?xml version="1.0"?>
<rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:foaf="http://xmlns.com/foaf/0.1/">
  <rdf:Description rdf:about="http://dblp.org/rec/conf/test/paper1">
    <dc:title>Entity Resolution Benchmark</dc:title>
    <foaf:maker rdf:resource="http://dblp.org/person/Alice_Smith"/>
  </rdf:Description>
</rdf:RDF>
"""


REXA_RDF = """<?xml version="1.0"?>
<rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:foaf="http://xmlns.com/foaf/0.1/">
  <rdf:Description rdf:about="http://rexa.info/paper/1">
    <dc:title>Entity Resolution Benchmark</dc:title>
    <foaf:maker rdf:resource="http://rexa.info/person/Alice_Smith"/>
  </rdf:Description>
</rdf:RDF>
"""


ALIGNMENT_RDF = """<?xml version="1.0"?>
<rdf:RDF
    xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <Alignment>
    <map>
      <Cell>
        <entity1 rdf:resource="http://rexa.info/paper/1"/>
        <entity2 rdf:resource="http://dblp.org/rec/conf/test/paper1"/>
        <relation>=</relation>
        <measure rdf:datatype="http://www.w3.org/2001/XMLSchema#float">1.0</measure>
      </Cell>
    </map>
  </Alignment>
</rdf:RDF>
"""


def test_load_rexa_dblp_orients_oaei_alignment(tmp_path: Path) -> None:
    (tmp_path / "dblp.rdf").write_text(DBLP_RDF, encoding="utf-8")
    (tmp_path / "rexa.rdf").write_text(REXA_RDF, encoding="utf-8")
    (tmp_path / "refalign.rdf").write_text(ALIGNMENT_RDF, encoding="utf-8")

    df_dblp, df_rexa, df_truth = load_rexa_dblp(str(tmp_path))

    assert len(df_dblp) == 1
    assert len(df_rexa) == 1
    assert {"idDBLP", "idRexa"} <= set(df_truth.columns)
    assert df_truth.iloc[0]["idDBLP"] == "http://dblp.org/rec/conf/test/paper1"
    assert df_truth.iloc[0]["idRexa"] == "http://rexa.info/paper/1"
    assert "maker" in df_dblp.columns
    assert "alice smith" in df_dblp.iloc[0]["maker"]


def test_run_ingestion_rexa_dblp_writes_cleaned_files(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "dblp.rdf").write_text(DBLP_RDF, encoding="utf-8")
    (tmp_path / "rexa.rdf").write_text(REXA_RDF, encoding="utf-8")
    (tmp_path / "refalign.rdf").write_text(ALIGNMENT_RDF, encoding="utf-8")

    monkeypatch.setitem(DATASET_REGISTRY["rexa_dblp"], "raw_dir", str(tmp_path))

    df1, df2, df_truth = run_ingestion(dataset="rexa_dblp")

    assert len(df1) == 1
    assert len(df2) == 1
    assert len(df_truth) == 1
    assert df1.iloc[0]["source"] == "dblp"
    assert df2.iloc[0]["source"] == "rexa"
    assert "alice smith" in df1.iloc[0]["maker"]
    assert Path("data/cleaned/rexa_dblp/cleaned_source1.csv").exists()
    assert Path("data/cleaned/rexa_dblp/cleaned_source2.csv").exists()
    assert Path("data/cleaned/rexa_dblp/ground_truth.csv").exists()


def test_load_dblp_acm_uses_positive_pairs_from_all_splits(tmp_path: Path) -> None:
    pd.DataFrame([
        {"id": 0, "title": "paper a", "authors": "alice", "venue": "vldb", "year": 2000},
        {"id": 1, "title": "paper b", "authors": "bob", "venue": "sigmod", "year": 2001},
    ]).to_csv(tmp_path / "tableA.csv", index=False)
    pd.DataFrame([
        {"id": 10, "title": "paper a", "authors": "alice", "venue": "vldb", "year": 2000},
        {"id": 11, "title": "paper c", "authors": "cara", "venue": "icde", "year": 2003},
    ]).to_csv(tmp_path / "tableB.csv", index=False)
    pd.DataFrame([
        {"ltable_id": 0, "rtable_id": 10, "label": 1},
        {"ltable_id": 1, "rtable_id": 11, "label": 0},
    ]).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame([
        {"ltable_id": 0, "rtable_id": 10, "label": 1},
    ]).to_csv(tmp_path / "valid.csv", index=False)
    pd.DataFrame([
        {"ltable_id": 1, "rtable_id": 11, "label": 0},
    ]).to_csv(tmp_path / "test.csv", index=False)

    df_dblp, df_acm, df_truth = load_dblp_acm(str(tmp_path))

    assert len(df_dblp) == 2
    assert len(df_acm) == 2
    assert list(df_truth.columns) == ["idDBLP", "idACM"]
    assert len(df_truth) == 1
    assert int(df_truth.iloc[0]["idDBLP"]) == 0
    assert int(df_truth.iloc[0]["idACM"]) == 10
