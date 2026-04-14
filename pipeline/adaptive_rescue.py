from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _series_or_default(df: pd.DataFrame, column: str, default: float | str = 0.0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _parse_witness_families(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    families = set()
    for part in text.split("|"):
        token = part.strip()
        if not token:
            continue
        families.add(token.split(":", 1)[0] if ":" in token else token)
    return families


def profile_matching_surface(final_df: pd.DataFrame) -> dict[str, float | bool]:
    if final_df.empty:
        return {
            "near_threshold_ratio": 0.0,
            "score_std": 0.0,
            "mid_band_density": 0.0,
            "activate": False,
        }

    near_threshold_ratio = float(final_df["final_score"].between(0.35, 0.65, inclusive="both").mean())
    mid_band_density = float(final_df["value_score"].between(0.30, 0.60, inclusive="both").mean())
    score_std = float(final_df["final_score"].std() or 0.0)

    # Activate only on broad ambiguous surfaces.
    activate = (
        near_threshold_ratio >= 0.15
        and mid_band_density >= 0.20
        and score_std <= 0.28
    )
    return {
        "near_threshold_ratio": near_threshold_ratio,
        "score_std": score_std,
        "mid_band_density": mid_band_density,
        "activate": bool(activate),
    }


def apply_adaptive_rescue(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unified adaptive rescue layer.

    It consolidates the useful part of previous post-reranking experiments:
    - local completion pressure
    - compact symbolic support
    - witness-family diversity

    Crucially, it activates only on globally ambiguous matching surfaces and
    only for locally dominant negative pairs in the direct-score ambiguity band.
    """
    df = final_df.copy()
    profile = profile_matching_surface(df)
    df["adaptive_rescue_active"] = bool(profile["activate"])
    df["adaptive_rescue_bonus"] = 0.0
    df["adaptive_rescue_score"] = df["final_score"].astype(float) if not df.empty else pd.Series(dtype=float)

    if df.empty or not profile["activate"]:
        return df

    df["rank_final_A"] = df.groupby("id_A")["final_score"].rank(method="first", ascending=False)
    df["rank_final_B"] = df.groupby("id_B")["final_score"].rank(method="first", ascending=False)

    best_final_a = df.groupby("id_A")["final_score"].transform("max")
    second_final_a = (
        df[df["rank_final_A"] > 1.0]
        .groupby("id_A")["final_score"]
        .transform("max")
        .reindex(df.index)
        .fillna(0.0)
    )
    best_final_b = df.groupby("id_B")["final_score"].transform("max")
    second_final_b = (
        df[df["rank_final_B"] > 1.0]
        .groupby("id_B")["final_score"]
        .transform("max")
        .reindex(df.index)
        .fillna(0.0)
    )

    df["completion_margin_A"] = np.where(
        df["rank_final_A"] == 1.0,
        np.clip(best_final_a - second_final_a, 0.0, 1.0),
        0.0,
    )
    df["completion_margin_B"] = np.where(
        df["rank_final_B"] == 1.0,
        np.clip(best_final_b - second_final_b, 0.0, 1.0),
        0.0,
    )

    witness_families = _series_or_default(df, "witness_path", "").apply(_parse_witness_families)
    df["witness_family_count"] = witness_families.apply(len).astype(float)
    df["completion_family_norm"] = np.clip(df["witness_family_count"] / 4.0, 0.0, 1.0)

    evidence = pd.to_numeric(_series_or_default(df, "evidence_count", 0.0), errors="coerce").fillna(0.0)
    region = pd.to_numeric(_series_or_default(df, "region_size", 0.0), errors="coerce").fillna(0.0)
    compact_support = evidence / np.log1p(region.clip(lower=1.0))
    if compact_support.max() > compact_support.min():
        compact_support = (compact_support - compact_support.min()) / (compact_support.max() - compact_support.min())
    else:
        compact_support = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    df["completion_compact_support"] = compact_support.astype(float)

    ambiguous_window = (
        df["value_score"].between(0.35, 0.55, inclusive="both")
        & (df["final_score"] < 0.85)
        & (df["is_match"].astype(int) == 0)
    )
    local_dominance = (
        (df["rank_final_A"] == 1.0) & (df["rank_final_B"] <= 2.0)
    ) | (
        (df["rank_final_B"] == 1.0) & (df["rank_final_A"] <= 2.0)
    )
    structured_support = (
        (df["completion_family_norm"] >= 0.50)
        | (df["completion_compact_support"] >= 0.55)
    )

    completion_pressure = np.clip(
        0.45 * ((df["completion_margin_A"] + df["completion_margin_B"]) / 2.0)
        + 0.35 * df["completion_family_norm"]
        + 0.20 * df["completion_compact_support"],
        0.0,
        1.0,
    )
    gated = ambiguous_window & local_dominance & structured_support
    bonus = np.where(gated, 0.18 * completion_pressure, 0.0)

    df["adaptive_rescue_bonus"] = bonus.astype(float)
    df["adaptive_rescue_score"] = np.clip(df["final_score"] + df["adaptive_rescue_bonus"], 0.0, 1.0)
    df["final_score"] = df["adaptive_rescue_score"]
    df["is_match"] = (df["final_score"] > 0.5).astype(int)

    return df.drop(
        columns=[
            "rank_final_A",
            "rank_final_B",
            "completion_margin_A",
            "completion_margin_B",
            "witness_family_count",
        ]
    )
