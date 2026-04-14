"""Tests for the witness-first region-collapse prototype."""

from __future__ import annotations

import pandas as pd

from pipeline.constraint_witness import (
    asymmetric_text_rescue_pairs,
    build_witness_regions,
    collapse_witness_regions,
    extract_witnesses,
    facet_rescue_pairs,
    predictive_rescue_pairs,
    run_constraint_witness_resolution,
    semantic_rescue_pairs,
    strong_witness_rescue_pairs,
)


def _toy_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    source1 = pd.DataFrame(
        [
            {
                "id": "a1",
                "title": "Dell Inspiron 15",
                "description": "15 inch laptop 16GB",
                "manufacturer": "Dell",
                "price": "499",
            },
            {
                "id": "a2",
                "title": "HP Pavilion 14",
                "description": "14 inch laptop 8GB",
                "manufacturer": "HP",
                "price": "599",
            },
        ]
    )
    source2 = pd.DataFrame(
        [
            {
                "id": "b1",
                "title": "Dell Inspiron 15 Notebook",
                "description": "Laptop 16GB RAM",
                "manufacturer": "Dell",
                "price": "505",
            },
            {
                "id": "b2",
                "title": "HP Pavilion 14 Notebook",
                "description": "Laptop 8GB RAM",
                "manufacturer": "HP",
                "price": "610",
            },
            {
                "id": "b3",
                "title": "Dell Optical Mouse",
                "description": "USB office mouse",
                "manufacturer": "Dell",
                "price": "20",
            },
        ]
    )
    return source1, source2


def test_extract_witnesses_emits_multiple_witness_families():
    source1, source2 = _toy_sources()

    witnesses = extract_witnesses(
        source1,
        source2,
        source1="s1",
        source2="s2",
        enable_semantic_bands=True,
    )

    assert not witnesses.empty
    assert {"rare_token", "anchor_bigram", "categorical_value", "numeric_bucket"} <= set(
        witnesses["witness_type"]
    )
    assert set(witnesses["source"]) == {"s1", "s2"}



def test_build_witness_regions_only_keeps_cross_source_regions():
    source1, source2 = _toy_sources()
    witnesses = extract_witnesses(
        source1,
        source2,
        source1="s1",
        source2="s2",
        enable_semantic_bands=True,
    )

    regions, entity_witnesses = build_witness_regions(
        witnesses,
        source1="s1",
        source2="s2",
    )

    assert regions
    assert "a1" in entity_witnesses
    assert all(state.left_ids and state.right_ids for state in regions.values())


def test_collapse_witness_regions_materializes_true_pairs_from_small_regions():
    source1, source2 = _toy_sources()
    witnesses = extract_witnesses(
        source1,
        source2,
        source1="s1",
        source2="s2",
        enable_semantic_bands=True,
    )

    candidates, stats = collapse_witness_regions(
        witnesses,
        source1="s1",
        source2="s2",
        max_cartesian_size=1,
    )

    produced = set(zip(candidates["id_A"], candidates["id_B"]))
    assert ("a1", "b1") in produced
    assert ("a2", "b2") in produced
    assert stats["candidate_pairs"] <= 4
    assert stats["region_visits"] >= stats["materialized_regions"]


def test_run_constraint_witness_resolution_reduces_against_full_cartesian():
    source1, source2 = _toy_sources()

    witnesses, candidates, stats = run_constraint_witness_resolution(
        source1,
        source2,
        source1="s1",
        source2="s2",
        max_cartesian_size=1,
    )

    naive_pairs = len(source1) * len(source2)
    assert len(witnesses) > 0
    assert stats["candidate_pairs"] < naive_pairs
    assert list(candidates.columns) == [
        "id_A",
        "id_B",
        "certificate_score",
        "evidence_count",
        "witness_path",
        "region_size",
    ]


def test_extract_witnesses_emits_model_code_witnesses_for_reformatted_references():
    source1 = pd.DataFrame(
        [
            {
                "id": "a1",
                "name": "Canon CLI-8M Ink Cartridge",
                "description": "Part 0622B002",
                "manufacturer": "Canon",
            }
        ]
    )
    source2 = pd.DataFrame(
        [
            {
                "id": "b1",
                "name": "Canon CLI8M Ink Cartridge",
                "description": "0622B002 Magenta",
                "manufacturer": "Canon",
            }
        ]
    )

    witnesses = extract_witnesses(
        source1,
        source2,
        source1="s1",
        source2="s2",
        enable_semantic_bands=True,
    )
    emitted = set(
        zip(
            witnesses["witness_type"].astype(str),
            witnesses["witness_value"].astype(str),
        )
    )

    assert ("model_code", "cli8m") in emitted
    assert ("model_code", "0622b002") in emitted
    assert ("categorical_model", "canon::cli8m") in emitted


def test_extract_witnesses_emits_free_title_signatures_for_rephrased_titles():
    source1 = pd.DataFrame(
        [
            {
                "id": "a1",
                "title": "Production Premium CS3 Mac Upgrade",
                "description": "Adobe creative suite production premium",
            }
        ]
    )
    source2 = pd.DataFrame(
        [
            {
                "id": "b1",
                "name": "Adobe CS3 Production Premium Mac Upgrade",
                "description": "production premium cs3 mac software",
            }
        ]
    )

    witnesses = extract_witnesses(
        source1,
        source2,
        source1="s1",
        source2="s2",
        enable_semantic_bands=True,
    )
    emitted = set(
        zip(
            witnesses["witness_type"].astype(str),
            witnesses["witness_value"].astype(str),
        )
    )

    assert any(wtype == "rare_pair" for wtype, _ in emitted)
    assert any(wtype == "title_prefix3" for wtype, _ in emitted)


def test_extract_witnesses_emits_shared_semantic_bands_for_close_titles():
    source1 = pd.DataFrame(
        [
            {
                "id": "a1",
                "title": "Adobe CS3 Production Premium Mac Upgrade",
                "description": "creative suite production premium",
            }
        ]
    )
    source2 = pd.DataFrame(
        [
            {
                "id": "b1",
                "name": "Production Premium CS3 Mac Upgrade by Adobe",
                "description": "mac production premium cs3 software",
            }
        ]
    )

    witnesses = extract_witnesses(
        source1,
        source2,
        source1="s1",
        source2="s2",
        enable_semantic_bands=True,
    )
    left = set(
        witnesses.loc[
            (witnesses["entity_id"] == "a1")
            & (witnesses["witness_type"] == "semantic_band"),
            "witness_value",
        ].astype(str)
    )
    right = set(
        witnesses.loc[
            (witnesses["entity_id"] == "b1")
            & (witnesses["witness_type"] == "semantic_band"),
            "witness_value",
        ].astype(str)
    )

    assert left
    assert right
    assert left & right


def test_semantic_rescue_pairs_only_targets_symbolically_weak_left_entities():
    source1 = pd.DataFrame(
        [
            {"id": "a1", "title": "Adobe CS3 Production Premium Mac Upgrade"},
            {"id": "a2", "title": "Canon CLI-8M Ink Cartridge"},
        ]
    )
    source2 = pd.DataFrame(
        [
            {"id": "b1", "name": "Production Premium CS3 Mac Upgrade by Adobe"},
            {"id": "b2", "name": "Canon CLI8M Ink Cartridge"},
        ]
    )
    symbolic = pd.DataFrame(
        [
            {
                "id_A": "a2",
                "id_B": "b2",
                "certificate_score": 3.5,
                "evidence_count": 3,
                "witness_path": "symbolic",
                "region_size": 1,
            }
        ]
    )

    rescue, stats = semantic_rescue_pairs(
        source1,
        source2,
        symbolic,
        top_k_per_left=2,
        min_similarity=0.2,
        symbolic_confidence_threshold=2.2,
    )

    produced = set(zip(rescue["id_A"], rescue["id_B"]))
    assert ("a1", "b1") in produced
    assert all(left_id != "a2" for left_id in rescue["id_A"])
    assert stats["semantic_rescue_left_entities"] == 1


def test_semantic_rescue_pairs_reports_scored_pairs_and_finds_reformatted_titles():
    source1 = pd.DataFrame(
        [
            {"id": "a1", "title": "Canon PIXMA MP500 photo all in one printer"},
        ]
    )
    source2 = pd.DataFrame(
        [
            {"id": "b1", "name": "Canon Pixma MP-500 all-in-one photo printer"},
            {"id": "b2", "name": "Epson Stylus C88 color printer"},
            {"id": "b3", "name": "Logitech cordless optical mouse"},
            {"id": "b4", "name": "Sony noise cancelling headphones"},
            {"id": "b5", "name": "KitchenAid artisan mixer"},
            {"id": "b6", "name": "Dell ultrasharp monitor 24 inch"},
        ]
    )

    rescue, stats = semantic_rescue_pairs(
        source1,
        source2,
        pd.DataFrame(columns=["id_A", "id_B", "certificate_score", "evidence_count", "witness_path", "region_size"]),
        top_k_per_left=2,
        min_similarity=0.15,
        symbolic_confidence_threshold=2.2,
    )

    produced = set(zip(rescue["id_A"], rescue["id_B"]))
    assert ("a1", "b1") in produced
    assert stats["semantic_rescue_left_entities"] == 1
    assert stats["semantic_rescue_scored_pairs"] == len(source2)


def test_predictive_rescue_pairs_uses_pseudo_positives_to_recover_weak_pairs():
    source1 = pd.DataFrame(
        [
            {"id": "a1", "title": "Dell Inspiron 15 laptop"},
            {"id": "a2", "title": "HP Pavilion 14 laptop"},
            {"id": "a3", "title": "Canon Pixma MP500 all in one printer"},
        ]
    )
    source2 = pd.DataFrame(
        [
            {"id": "b1", "name": "Dell Inspiron 15 notebook"},
            {"id": "b2", "name": "HP Pavilion 14 notebook"},
            {"id": "b3", "name": "Canon Pixma MP-500 all-in-one printer"},
            {"id": "b4", "name": "Epson Stylus C88 printer"},
        ]
    )
    witnesses = extract_witnesses(source1, source2, source1="s1", source2="s2")
    base = pd.DataFrame(
        [
            {
                "id_A": "a1",
                "id_B": "b1",
                "certificate_score": 3.6,
                "evidence_count": 3,
                "witness_path": "symbolic",
                "region_size": 1,
            },
            {
                "id_A": "a2",
                "id_B": "b2",
                "certificate_score": 3.4,
                "evidence_count": 3,
                "witness_path": "symbolic",
                "region_size": 1,
            },
        ]
    )

    rescue, stats = predictive_rescue_pairs(
        source1,
        source2,
        witnesses,
        base,
        probability_threshold=0.55,
        semantic_shortlist_k=3,
        min_local_prototype_size=1,
    )

    produced = set(zip(rescue["id_A"], rescue["id_B"]))
    assert ("a3", "b3") in produced
    assert stats["predictive_pseudo_positives"] == 2
    assert stats["predictive_rescue_left_entities"] >= 1
    assert stats["predictive_local_prototypes"] >= 1


def test_strong_witness_rescue_pairs_recovers_block_backed_reference_pair():
    source1 = pd.DataFrame(
        [
            {"id": "a1", "title": "Canon CLI-8M ink cartridge magenta"},
            {"id": "a2", "title": "Dell Inspiron 15 laptop"},
        ]
    )
    source2 = pd.DataFrame(
        [
            {"id": "b1", "name": "Canon CLI8M magenta ink cartridge"},
            {"id": "b2", "name": "Dell Inspiron 15 notebook"},
        ]
    )
    blocks = pd.DataFrame(
        [
            {"block_id": "canon_cli", "entity_id": "a1", "source": "s1"},
            {"block_id": "canon_cli", "entity_id": "b1", "source": "s2"},
            {"block_id": "dell_15", "entity_id": "a2", "source": "s1"},
            {"block_id": "dell_15", "entity_id": "b2", "source": "s2"},
        ]
    )
    witnesses = extract_witnesses(
        source1,
        source2,
        source1="s1",
        source2="s2",
        blocks_df=blocks,
    )
    base = pd.DataFrame(
        [
            {
                "id_A": "a2",
                "id_B": "b2",
                "certificate_score": 3.4,
                "evidence_count": 3,
                "witness_path": "symbolic",
                "region_size": 1,
            },
        ]
    )

    rescue, stats = strong_witness_rescue_pairs(
        source1,
        source2,
        witnesses,
        base,
        symbolic_confidence_threshold=2.4,
        min_bridge_score=1.2,
    )

    produced = set(zip(rescue["id_A"], rescue["id_B"]))
    assert ("a1", "b1") in produced
    assert all(left_id != "a2" for left_id in rescue["id_A"])
    assert stats["strong_rescue_left_entities"] == 1
    assert stats["strong_rescue_pairs"] >= 1


def test_asymmetric_text_rescue_pairs_recovers_short_vs_long_title_match():
    source1 = pd.DataFrame(
        [
            {"id": "a1", "title": "Adobe Photoshop Elements 8"},
            {"id": "a2", "title": "Canon CLI-8M ink cartridge"},
        ]
    )
    source2 = pd.DataFrame(
        [
            {"id": "b1", "name": "Adobe Photoshop Elements 8 for Mac and Windows retail package"},
            {"id": "b2", "name": "Canon CLI8M magenta ink cartridge"},
        ]
    )
    blocks = pd.DataFrame(
        [
            {"block_id": "canon", "entity_id": "a2", "source": "s1"},
            {"block_id": "canon", "entity_id": "b2", "source": "s2"},
        ]
    )
    witnesses = extract_witnesses(
        source1,
        source2,
        source1="s1",
        source2="s2",
        blocks_df=blocks,
    )
    base = pd.DataFrame(
        [
            {
                "id_A": "a2",
                "id_B": "b2",
                "certificate_score": 3.6,
                "evidence_count": 3,
                "witness_path": "symbolic",
                "region_size": 1,
            },
        ]
    )

    rescue, stats = asymmetric_text_rescue_pairs(
        source1,
        source2,
        witnesses,
        base,
        symbolic_confidence_threshold=2.9,
        min_rescue_score=0.65,
    )

    produced = set(zip(rescue["id_A"], rescue["id_B"]))
    assert ("a1", "b1") in produced
    assert stats["asymmetric_rescue_left_entities"] == 1
    assert stats["asymmetric_rescue_pairs"] >= 1


def test_facet_rescue_pairs_recovers_family_version_edition_match():
    source1 = pd.DataFrame(
        [
            {"id": "a1", "title": "Adobe Photoshop Elements 8 upgrade"},
            {"id": "a2", "title": "Canon CLI-8M ink cartridge"},
        ]
    )
    source2 = pd.DataFrame(
        [
            {"id": "b1", "name": "Adobe Photoshop Elements 8 software upgrade retail"},
            {"id": "b2", "name": "Canon CLI8M magenta ink cartridge"},
        ]
    )
    witnesses = extract_witnesses(source1, source2, source1="s1", source2="s2")
    base = pd.DataFrame(
        [
            {
                "id_A": "a2",
                "id_B": "b2",
                "certificate_score": 3.6,
                "evidence_count": 3,
                "witness_path": "symbolic",
                "region_size": 1,
            },
        ]
    )

    rescue, stats = facet_rescue_pairs(
        source1,
        source2,
        witnesses,
        base,
        symbolic_confidence_threshold=3.1,
        min_facet_score=0.7,
    )

    produced = set(zip(rescue["id_A"], rescue["id_B"]))
    assert ("a1", "b1") in produced
    assert stats["facet_rescue_left_entities"] == 1
    assert stats["facet_rescue_pairs"] >= 1
