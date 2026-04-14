"""Witness-first entity resolution prototype.

This module implements a concrete MVP of the design captured in
``tasks/novel_er_design.md``.  The key shift is that we do not enumerate
pairwise candidates inside coarse blocks.  Instead, we:

1. extract heterogeneous witnesses per entity
2. build cross-source witness regions
3. progressively collapse those regions by intersecting additional witnesses
4. materialize explicit pairs only when the cartesian region becomes small
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import hashlib
import math
import re
from typing import Any

import pandas as pd


_TOKEN_RE = re.compile(r"[^a-z0-9\s]")
_SPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")
_VOWEL_RE = re.compile(r"[aeiou]")
_MODEL_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_/]*[a-z0-9]")
_YEAR_OR_VERSION_RE = re.compile(r"^(?:\d{1,4}(?:\.\d+)?)$")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "it",
    "on", "with", "as", "at", "by", "from", "that", "this", "was", "are",
    "be", "been", "have", "has", "had", "not", "but", "they", "we", "you",
    "he", "she", "do", "did", "will", "would", "can", "could", "may",
    "might", "should", "shall", "its", "their", "our", "your", "his", "her",
    "new", "one", "all", "more", "also", "so", "if", "then", "than",
}

EDITION_TERMS = {
    "academic", "student", "teacher", "teachered", "upgrade", "full", "retail",
    "professional", "pro", "premium", "standard", "basic", "home", "business",
    "enterprise", "ultimate", "deluxe", "suite", "trial", "subscription",
    "license", "digital", "download", "oem", "boxed",
}



@dataclass(frozen=True)
class RegionState:
    """Implicit candidate region before pair materialization."""

    left_ids: frozenset[str]
    right_ids: frozenset[str]
    evidence: tuple[str, ...]
    evidence_mass: float

    @property
    def cartesian_size(self) -> int:
        return len(self.left_ids) * len(self.right_ids)


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower()
    text = _TOKEN_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _tokenize(text: str, min_len: int = 2) -> list[str]:
    if not text:
        return []
    return [
        tok for tok in _normalize_text(text).split()
        if len(tok) >= min_len and tok not in STOPWORDS
    ]


def _text_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in {"id", "source"}:
            continue
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            cols.append(col)
    return cols


def _infer_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols: list[str] = []
    for col in df.columns:
        if col in {"id", "source"}:
            continue
        series = df[col]
        non_empty = series.astype(str).str.strip() != ""
        if not non_empty.any():
            continue
        parsed = pd.to_numeric(series.where(non_empty, None), errors="coerce")
        coverage = float(parsed.notna().sum()) / float(non_empty.sum())
        if coverage >= 0.6:
            numeric_cols.append(col)
    return numeric_cols


def _infer_short_categorical_columns(df: pd.DataFrame, text_columns: list[str]) -> list[str]:
    categorical_cols: list[str] = []
    n_rows = max(1, len(df))
    for col in text_columns:
        series = df[col].fillna("").astype(str).str.strip()
        series = series[series != ""]
        if series.empty:
            continue
        avg_tokens = series.map(lambda value: len(_tokenize(value)) or 1).mean()
        unique_ratio = series.nunique() / n_rows
        if avg_tokens <= 3.0 and unique_ratio <= 0.5:
            categorical_cols.append(col)
    return categorical_cols


def _row_text(row: pd.Series, text_columns: list[str]) -> str:
    parts = []
    for col in text_columns:
        value = row.get(col, "")
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return " ".join(parts)


def _row_raw_text(row: pd.Series, text_columns: list[str]) -> str:
    parts = []
    for col in text_columns:
        value = row.get(col, "")
        if value is None or pd.isna(value):
            continue
        raw = str(value).strip()
        if raw:
            parts.append(raw.lower())
    return " ".join(parts)


def _token_skeleton(token: str) -> str:
    if len(token) <= 4:
        return ""
    head, tail = token[0], token[1:]
    collapsed = head + _VOWEL_RE.sub("", tail)
    return collapsed if len(collapsed) >= 4 and collapsed != token else ""


def _price_bucket(row: pd.Series) -> str:
    raw = row.get("price", "")
    price = pd.to_numeric(raw, errors="coerce")
    if pd.isna(price):
        return ""
    bucket = int(math.floor(float(price) / 50.0))
    return f"{bucket * 50:05d}_{bucket * 50 + 49:05d}"


def _numeric_bucket(raw: Any, width: float = 50.0) -> str:
    value = pd.to_numeric(raw, errors="coerce")
    if pd.isna(value):
        return ""
    bucket = int(math.floor(float(value) / width))
    return f"{bucket}"


def _digit_signature(tokens: list[str]) -> list[str]:
    signatures = []
    for token in tokens:
        digits = "".join(_DIGIT_RE.findall(token))
        if len(digits) >= 2:
            signatures.append(digits)
    return signatures[:2]


def _extract_model_codes(raw_text: str) -> set[str]:
    codes: set[str] = set()
    if not raw_text:
        return codes

    for match in _MODEL_TOKEN_RE.findall(raw_text):
        compact = re.sub(r"[^a-z0-9]", "", match.lower())
        if len(compact) < 4:
            continue
        has_alpha = any(ch.isalpha() for ch in compact)
        has_digit = any(ch.isdigit() for ch in compact)
        if has_alpha and has_digit:
            codes.add(compact)

    digit_runs = _DIGIT_RE.findall(raw_text)
    for idx in range(len(digit_runs) - 1):
        merged = digit_runs[idx] + digit_runs[idx + 1]
        if len(merged) >= 6:
            codes.add(merged)
    for idx in range(len(digit_runs) - 2):
        merged = digit_runs[idx] + digit_runs[idx + 1] + digit_runs[idx + 2]
        if len(merged) >= 6:
            codes.add(merged)

    return codes


def _title_like_columns(df: pd.DataFrame, text_columns: list[str]) -> list[str]:
    preferred = []
    fallback = []
    for col in text_columns:
        lowered = col.lower()
        if any(key in lowered for key in ("title", "name", "label")):
            preferred.append(col)
        else:
            fallback.append(col)
    return preferred or fallback[:1]


def _entity_title_texts(df: pd.DataFrame) -> dict[str, str]:
    text_columns = _text_columns(df)
    title_columns = _title_like_columns(df, text_columns)
    texts: dict[str, str] = {}
    for _, row in df.iterrows():
        texts[str(row["id"])] = _row_text(row, title_columns)
    return texts


def _sparse_tfidf_vectors(texts: dict[str, str]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    doc_features: dict[str, list[str]] = {}
    document_frequency: Counter[str] = Counter()
    for entity_id, text in texts.items():
        tokens = _tokenize(text)
        features = [f"tok:{token}" for token in tokens[:12]]
        features.extend(f"cg:{gram}" for gram in _char_ngrams(text, n=3)[:32])
        dedup = list(dict.fromkeys(features))
        doc_features[entity_id] = dedup
        document_frequency.update(set(dedup))

    n_docs = max(1, len(doc_features))
    idf = {
        feature: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        for feature, df in document_frequency.items()
    }

    vectors: dict[str, dict[str, float]] = {}
    for entity_id, features in doc_features.items():
        tf = Counter(features)
        vector = {feature: count * idf.get(feature, 1.0) for feature, count in tf.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors[entity_id] = {feature: value / norm for feature, value in vector.items()}
    return vectors, idf


def _sparse_char_tfidf_vectors(texts: dict[str, str]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    doc_features: dict[str, list[str]] = {}
    document_frequency: Counter[str] = Counter()
    for entity_id, text in texts.items():
        features = [f"cg:{gram}" for gram in _char_ngrams(text, n=3)[:64]]
        dedup = list(dict.fromkeys(features))
        doc_features[entity_id] = dedup
        document_frequency.update(set(dedup))

    n_docs = max(1, len(doc_features))
    idf = {
        feature: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        for feature, df in document_frequency.items()
    }

    vectors: dict[str, dict[str, float]] = {}
    for entity_id, features in doc_features.items():
        tf = Counter(features)
        vector = {feature: count * idf.get(feature, 1.0) for feature, count in tf.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors[entity_id] = {feature: value / norm for feature, value in vector.items()}
    return vectors, idf


def _sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(feature, 0.0) for feature, value in left.items())


def _jaccard_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _informative_tokens(text: str) -> set[str]:
    return {
        tok for tok in _tokenize(text, min_len=3)
        if not tok.isdigit() and len(tok) >= 3
    }


def _containment_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    small, big = (left, right) if len(left) <= len(right) else (right, left)
    if not small:
        return 0.0
    return len(small & big) / float(len(small))


def _catalog_facets(text: str) -> dict[str, set[str]]:
    tokens = _tokenize(text, min_len=2)
    facets: dict[str, set[str]] = defaultdict(set)
    if not tokens:
        return facets

    alpha_tokens = [tok for tok in tokens if tok.isalpha() and len(tok) >= 3]
    if alpha_tokens:
        facets["lead"].add(alpha_tokens[0])
        for tok in alpha_tokens[:4]:
            if tok not in STOPWORDS:
                facets["family"].add(tok)

    for tok in tokens:
        compact = re.sub(r"[^a-z0-9]", "", tok.lower())
        if not compact:
            continue
        if tok in EDITION_TERMS:
            facets["edition"].add(tok)
        if any(ch.isalpha() for ch in compact) and any(ch.isdigit() for ch in compact):
            facets["version"].add(compact)
        elif _YEAR_OR_VERSION_RE.match(compact):
            facets["version"].add(compact)
        elif tok.isalpha() and len(tok) >= 4 and tok not in STOPWORDS and tok not in EDITION_TERMS:
            facets["descriptor"].add(tok)

    if len(alpha_tokens) >= 2:
        facets["lead_pair"].add("::".join(alpha_tokens[:2]))
    if len(alpha_tokens) >= 3:
        facets["lead_triple"].add("::".join(alpha_tokens[:3]))
    return facets


def _witness_sets_by_entity(witnesses_df: pd.DataFrame) -> dict[str, set[str]]:
    entity_witnesses: dict[str, set[str]] = defaultdict(set)
    if witnesses_df.empty:
        return {}
    for _, row in witnesses_df.iterrows():
        entity_id = str(row["entity_id"])
        entity_witnesses[entity_id].add(f"{row['witness_type']}::{row['witness_value']}")
    return dict(entity_witnesses)


def _typed_witness_values(witness_keys: set[str], witness_type: str) -> set[str]:
    prefix = f"{witness_type}::"
    return {key[len(prefix):] for key in witness_keys if key.startswith(prefix)}


def _pair_feature_vector(
    left_id: str,
    right_id: str,
    *,
    vectors: dict[str, dict[str, float]],
    texts_left: dict[str, str],
    texts_right: dict[str, str],
    entity_witnesses: dict[str, set[str]],
) -> list[float]:
    left_vector = vectors.get(left_id, {})
    right_vector = vectors.get(right_id, {})
    semantic_cosine = _sparse_cosine(left_vector, right_vector)

    left_tokens = set(_tokenize(texts_left.get(left_id, "")))
    right_tokens = set(_tokenize(texts_right.get(right_id, "")))
    token_jaccard = _jaccard_overlap(left_tokens, right_tokens)

    left_witnesses = entity_witnesses.get(left_id, set())
    right_witnesses = entity_witnesses.get(right_id, set())
    witness_jaccard = _jaccard_overlap(left_witnesses, right_witnesses)

    strong_types = {
        "model_code",
        "categorical_model",
        "digit_signature",
        "anchor_bigram",
        "rare_pair",
        "rare_triple",
    }
    shared_strong = 0.0
    for witness_type in strong_types:
        left_values = _typed_witness_values(left_witnesses, witness_type)
        right_values = _typed_witness_values(right_witnesses, witness_type)
        if left_values & right_values:
            shared_strong += 1.0
    shared_strong /= float(len(strong_types))

    block_overlap = 1.0 if (
        _typed_witness_values(left_witnesses, "block_id")
        & _typed_witness_values(right_witnesses, "block_id")
    ) else 0.0
    first_token_match = (
        1.0
        if left_tokens and right_tokens and next(iter(sorted(left_tokens))) == next(iter(sorted(right_tokens)))
        else 0.0
    )

    return [
        semantic_cosine,
        token_jaccard,
        witness_jaccard,
        shared_strong,
        block_overlap,
        first_token_match,
    ]


def _centroid(values: list[list[float]]) -> list[float]:
    if not values:
        return []
    dims = len(values[0])
    return [
        sum(row[idx] for row in values) / float(len(values))
        for idx in range(dims)
    ]


def _pair_prototype_keys(
    left_id: str,
    right_id: str,
    *,
    texts_left: dict[str, str],
    texts_right: dict[str, str],
    entity_witnesses: dict[str, set[str]],
) -> list[str]:
    left_witnesses = entity_witnesses.get(left_id, set())
    right_witnesses = entity_witnesses.get(right_id, set())
    keys: list[str] = []

    for witness_type in (
        "categorical_model",
        "model_code",
        "anchor_bigram",
        "rare_pair",
        "rare_triple",
        "title_prefix3",
        "digit_signature",
        "categorical_value",
        "block_id",
    ):
        shared = sorted(
            _typed_witness_values(left_witnesses, witness_type)
            & _typed_witness_values(right_witnesses, witness_type)
        )
        for value in shared[:2]:
            keys.append(f"{witness_type}::{value}")

    left_tokens = _tokenize(texts_left.get(left_id, ""))
    right_tokens = _tokenize(texts_right.get(right_id, ""))
    if left_tokens and right_tokens:
        if left_tokens[0] == right_tokens[0]:
            keys.append(f"lead_token::{left_tokens[0]}")
        prefix_overlap = sorted({tok[:4] for tok in left_tokens if len(tok) >= 4} & {tok[:4] for tok in right_tokens if len(tok) >= 4})
        for value in prefix_overlap[:2]:
            keys.append(f"lead_prefix::{value}")

    return list(dict.fromkeys(keys))


def _prototype_probability(
    features: list[float],
    positive_centroid: list[float],
    negative_centroid: list[float],
) -> float:
    if not positive_centroid:
        return 0.0
    if not negative_centroid:
        return max(0.0, min(1.0, sum(features) / max(1.0, len(features))))

    weights = [p - n for p, n in zip(positive_centroid, negative_centroid)]
    midpoint = [(p + n) / 2.0 for p, n in zip(positive_centroid, negative_centroid)]
    raw = sum(weight * (value - center) for weight, value, center in zip(weights, features, midpoint))
    return 1.0 / (1.0 + math.exp(-4.0 * raw))


def _free_title_signatures(
    tokens: list[str],
    token_frequency: Counter[str],
) -> dict[str, set[str]]:
    signatures: dict[str, set[str]] = defaultdict(set)
    if not tokens:
        return signatures

    ranked = sorted(set(tokens), key=lambda tok: (token_frequency[tok], tok))
    top = ranked[:4]
    if len(top) >= 2:
        signatures["rare_pair"].add("::".join(sorted(top[:2])))
    if len(top) >= 3:
        signatures["rare_triple"].add("::".join(sorted(top[:3])))

    if len(tokens) >= 3:
        signatures["title_prefix3"].add("::".join(tokens[:3]))

    long_tokens = [tok for tok in tokens if len(tok) >= 4]
    if len(long_tokens) >= 2:
        signatures["long_pair"].add("::".join(sorted(long_tokens[:2])))

    prefix4 = sorted({tok[:4] for tok in long_tokens if len(tok) >= 4})
    if len(prefix4) >= 2:
        signatures["prefix_pair"].add("::".join(prefix4[:2]))

    return signatures


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    compact = re.sub(r"\s+", " ", _normalize_text(text)).strip()
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[idx: idx + n] for idx in range(len(compact) - n + 1)]


def _stable_hash64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _simhash_signature(features: list[str]) -> int:
    if not features:
        return 0
    weights = [0] * 64
    for feature in features:
        hashed = _stable_hash64(feature)
        for bit in range(64):
            if hashed & (1 << bit):
                weights[bit] += 1
            else:
                weights[bit] -= 1
    signature = 0
    for bit, value in enumerate(weights):
        if value >= 0:
            signature |= (1 << bit)
    return signature


def _semantic_band_signatures(title_text: str, title_tokens: list[str]) -> set[str]:
    bands: set[str] = set()
    feature_sets: list[tuple[str, list[str]]] = []

    ordered_features: list[str] = []
    ordered_features.extend(f"tok:{token}" for token in title_tokens[:8])
    ordered_features.extend(
        f"sk:{skeleton}"
        for skeleton in (_token_skeleton(token) for token in title_tokens[:8])
        if skeleton
    )
    ordered_features.extend(f"cg:{gram}" for gram in _char_ngrams(title_text, n=3)[:24])
    feature_sets.append(("o", ordered_features))

    unordered_tokens = sorted(set(title_tokens))
    unordered_features: list[str] = []
    unordered_features.extend(f"tok:{token}" for token in unordered_tokens[:8])
    unordered_features.extend(
        f"sk:{skeleton}"
        for skeleton in (_token_skeleton(token) for token in unordered_tokens[:8])
        if skeleton
    )
    unordered_features.extend(
        f"pf:{token[:4]}" for token in unordered_tokens[:8] if len(token) >= 4
    )
    feature_sets.append(("u", unordered_features))

    for prefix, features in feature_sets:
        signature = _simhash_signature(features)
        for idx in range(4):
            band = (signature >> (idx * 16)) & 0xFFFF
            bands.add(f"{prefix}{idx}:{band:04x}")
            bands.add(f"{prefix}{idx}:c{(band >> 4):03x}")
    return bands


def extract_witnesses(
    df_source1: pd.DataFrame,
    df_source2: pd.DataFrame,
    *,
    source1: str,
    source2: str,
    rare_token_cap: int = 5,
    blocks_df: pd.DataFrame | None = None,
    enable_semantic_bands: bool = False,
) -> pd.DataFrame:
    """Return one row per extracted witness."""
    frames = [
        df_source1.copy().fillna("").assign(source=source1),
        df_source2.copy().fillna("").assign(source=source2),
    ]
    combined = pd.concat(frames, ignore_index=True)
    text_columns = _text_columns(combined)
    title_columns = _title_like_columns(combined, text_columns)
    numeric_columns = _infer_numeric_columns(combined)
    short_categorical_columns = _infer_short_categorical_columns(combined, text_columns)

    token_frequency: Counter[str] = Counter()
    entity_tokens: dict[str, list[str]] = {}
    for _, row in combined.iterrows():
        entity_id = str(row["id"])
        tokens = _tokenize(_row_text(row, text_columns))
        entity_tokens[entity_id] = tokens
        token_frequency.update(set(tokens))

    rows: list[dict[str, Any]] = []
    for _, row in combined.iterrows():
        entity_id = str(row["id"])
        source = str(row["source"])
        tokens = entity_tokens[entity_id]
        raw_text = _row_raw_text(row, text_columns)
        title_text = _row_text(row, title_columns)
        title_tokens = _tokenize(title_text)
        unique_tokens = sorted(set(tokens), key=lambda tok: (token_frequency[tok], tok))
        rare_tokens = unique_tokens[:rare_token_cap]

        for token in rare_tokens:
            weight = 1.0 / max(1, token_frequency[token])
            rows.append(
                {
                    "entity_id": entity_id,
                    "source": source,
                    "witness_type": "rare_token",
                    "witness_value": token,
                    "witness_weight": round(weight, 6),
                }
            )

        anchor_pairs: set[tuple[str, str]] = set()
        if len(tokens) >= 2:
            anchor_pairs.add((tokens[0], tokens[1]))
        if len(rare_tokens) >= 2:
            anchor_pairs.add(tuple(sorted(rare_tokens[:2])))
        for left_tok, right_tok in anchor_pairs:
            anchor = f"{left_tok}::{right_tok}"
            rows.append(
                {
                    "entity_id": entity_id,
                    "source": source,
                    "witness_type": "anchor_bigram",
                    "witness_value": anchor,
                    "witness_weight": 1.2,
                }
            )

        categorical_values: list[str] = []
        for col in short_categorical_columns:
            value = _normalize_text(row.get(col, ""))
            if not value:
                continue
            categorical_values.append(value)
            rows.append(
                {
                    "entity_id": entity_id,
                    "source": source,
                    "witness_type": "categorical_value",
                    "witness_value": f"{col}::{value}",
                    "witness_weight": 1.1,
                }
            )

        for col in numeric_columns:
            bucket_value = _numeric_bucket(row.get(col, ""), width=50.0)
            if not bucket_value:
                continue
            base_bucket = int(bucket_value)
            for delta, weight in ((0, 0.9), (-1, 0.45), (1, 0.45)):
                bucket = base_bucket + delta
                if bucket < 0:
                    continue
                rows.append(
                    {
                        "entity_id": entity_id,
                        "source": source,
                        "witness_type": "numeric_bucket",
                        "witness_value": f"{col}::{bucket}",
                        "witness_weight": weight,
                    }
                )

        for token in tokens[:6]:
            skeleton = _token_skeleton(token)
            if not skeleton:
                continue
            rows.append(
                {
                    "entity_id": entity_id,
                    "source": source,
                    "witness_type": "token_skeleton",
                    "witness_value": skeleton,
                    "witness_weight": 0.6,
                }
            )

        for digits in _digit_signature(tokens):
            rows.append(
                {
                    "entity_id": entity_id,
                    "source": source,
                    "witness_type": "digit_signature",
                    "witness_value": digits,
                    "witness_weight": 1.0,
                }
            )

        free_title = _free_title_signatures(title_tokens, token_frequency)
        for witness_type, values in free_title.items():
            base_weight = {
                "rare_pair": 1.25,
                "rare_triple": 1.35,
                "title_prefix3": 1.1,
                "long_pair": 0.95,
                "prefix_pair": 0.75,
            }.get(witness_type, 1.0)
            for value in values:
                rows.append(
                    {
                        "entity_id": entity_id,
                        "source": source,
                        "witness_type": witness_type,
                        "witness_value": value,
                        "witness_weight": base_weight,
                    }
                )

        if enable_semantic_bands:
            for value in _semantic_band_signatures(title_text, title_tokens):
                rows.append(
                    {
                        "entity_id": entity_id,
                        "source": source,
                        "witness_type": "semantic_band",
                        "witness_value": value,
                        "witness_weight": 0.85,
                    }
                )

        model_codes = sorted(_extract_model_codes(raw_text))
        for code in model_codes[:6]:
            rows.append(
                {
                    "entity_id": entity_id,
                    "source": source,
                    "witness_type": "model_code",
                    "witness_value": code,
                    "witness_weight": 1.35,
                }
            )

        if categorical_values and model_codes:
            for category in categorical_values[:2]:
                for code in model_codes[:4]:
                    rows.append(
                        {
                            "entity_id": entity_id,
                            "source": source,
                            "witness_type": "categorical_model",
                            "witness_value": f"{category}::{code}",
                            "witness_weight": 1.55,
                        }
                    )

        non_empty_cols = sorted(
            col for col in combined.columns
            if col not in {"id", "source"} and str(row.get(col, "")).strip()
        )
        if non_empty_cols:
            rows.append(
                {
                    "entity_id": entity_id,
                    "source": source,
                    "witness_type": "field_presence",
                    "witness_value": "|".join(non_empty_cols),
                    "witness_weight": 0.35,
                }
            )

    if blocks_df is not None and not blocks_df.empty:
        for _, row in blocks_df.iterrows():
            entity_id = str(row["entity_id"])
            source = str(row["source"])
            block_id = str(row["block_id"])
            rows.append(
                {
                    "entity_id": entity_id,
                    "source": source,
                    "witness_type": "block_id",
                    "witness_value": block_id,
                    "witness_weight": 0.7,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "source",
                "witness_type",
                "witness_value",
                "witness_weight",
            ]
        )
    return out.drop_duplicates().reset_index(drop=True)


def build_witness_regions(
    witnesses_df: pd.DataFrame,
    *,
    source1: str,
    source2: str,
) -> tuple[dict[str, RegionState], dict[str, set[str]]]:
    """Build initial cross-source witness regions and per-entity witness sets."""
    postings: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    entity_witnesses: dict[str, set[str]] = defaultdict(set)
    witness_weights: dict[str, float] = {}

    for _, row in witnesses_df.iterrows():
        witness_key = f"{row['witness_type']}::{row['witness_value']}"
        entity_id = str(row["entity_id"])
        source = str(row["source"])
        postings[witness_key][source].add(entity_id)
        entity_witnesses[entity_id].add(witness_key)
        witness_weights[witness_key] = float(row["witness_weight"])

    regions: dict[str, RegionState] = {}
    for witness_key, by_source in postings.items():
        left = frozenset(by_source.get(source1, set()))
        right = frozenset(by_source.get(source2, set()))
        if not left or not right:
            continue
        regions[witness_key] = RegionState(
            left_ids=left,
            right_ids=right,
            evidence=(witness_key,),
            evidence_mass=witness_weights.get(witness_key, 0.0),
        )

    return regions, dict(entity_witnesses)


def _shared_refinement_candidates(
    state: RegionState,
    entity_witnesses: dict[str, set[str]],
) -> set[str]:
    left_union: set[str] = set()
    for entity_id in state.left_ids:
        left_union |= entity_witnesses.get(entity_id, set())

    right_union: set[str] = set()
    for entity_id in state.right_ids:
        right_union |= entity_witnesses.get(entity_id, set())

    shared = left_union & right_union
    return {w for w in shared if w not in state.evidence}


def _refine_state(
    state: RegionState,
    witness_key: str,
    regions: dict[str, RegionState],
) -> RegionState | None:
    seed = regions.get(witness_key)
    if seed is None:
        return None

    new_left = state.left_ids & seed.left_ids
    new_right = state.right_ids & seed.right_ids
    if not new_left or not new_right:
        return None

    old_size = state.cartesian_size
    new_size = len(new_left) * len(new_right)
    if new_size >= old_size:
        return None

    evidence = tuple(sorted((*state.evidence, witness_key)))
    added_mass = max(0.0, seed.evidence_mass)
    return RegionState(
        left_ids=frozenset(new_left),
        right_ids=frozenset(new_right),
        evidence=evidence,
        evidence_mass=state.evidence_mass + added_mass,
    )


def collapse_witness_regions(
    witnesses_df: pd.DataFrame,
    *,
    source1: str,
    source2: str,
    max_cartesian_size: int = 8,
    max_expansions_per_state: int = 4,
    max_region_visits: int = 10000,
    rescue_cartesian_size: int = 64,
    rescue_top_k_per_left: int = 3,
    rescue_min_support: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse witness regions until they become small enough to materialize."""
    regions, entity_witnesses = build_witness_regions(
        witnesses_df,
        source1=source1,
        source2=source2,
    )
    heap: list[tuple[int, float, int, RegionState]] = []
    seen_signatures: set[tuple[frozenset[str], frozenset[str], tuple[str, ...]]] = set()
    emitted_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    rescue_scores: dict[tuple[str, str], dict[str, Any]] = {}

    counter = 0
    for state in regions.values():
        signature = (state.left_ids, state.right_ids, state.evidence)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        heapq.heappush(heap, (state.cartesian_size, -state.evidence_mass, counter, state))
        counter += 1

    region_visits = 0
    materialized_regions = 0

    while heap and region_visits < max_region_visits:
        _, _, _, state = heapq.heappop(heap)
        region_visits += 1

        if state.cartesian_size <= max_cartesian_size:
            materialized_regions += 1
            evidence_count = len(state.evidence)
            certificate_score = round(
                state.evidence_mass + evidence_count / (1.0 + state.cartesian_size),
                6,
            )
            for left_id in sorted(state.left_ids):
                for right_id in sorted(state.right_ids):
                    key = (left_id, right_id)
                    current = emitted_pairs.get(key)
                    if current and current["certificate_score"] >= certificate_score:
                        continue
                    emitted_pairs[key] = {
                        "id_A": left_id,
                        "id_B": right_id,
                        "certificate_score": certificate_score,
                        "evidence_count": evidence_count,
                        "witness_path": " | ".join(state.evidence),
                        "region_size": state.cartesian_size,
                    }
            continue

        if state.cartesian_size <= rescue_cartesian_size:
            evidence_count = len(state.evidence)
            support = round(
                state.evidence_mass + evidence_count / (1.0 + math.log1p(state.cartesian_size)),
                6,
            )
            for left_id in sorted(state.left_ids):
                for right_id in sorted(state.right_ids):
                    key = (left_id, right_id)
                    bucket = rescue_scores.setdefault(
                        key,
                        {
                            "id_A": left_id,
                            "id_B": right_id,
                            "certificate_score": 0.0,
                            "evidence_count": 0,
                            "witness_path": "",
                            "region_size": state.cartesian_size,
                        },
                    )
                    if support > bucket["certificate_score"]:
                        bucket["certificate_score"] = support
                        bucket["evidence_count"] = evidence_count
                        bucket["witness_path"] = " | ".join(state.evidence)
                        bucket["region_size"] = state.cartesian_size

        candidates = []
        for witness_key in _shared_refinement_candidates(state, entity_witnesses):
            refined = _refine_state(state, witness_key, regions)
            if refined is None:
                continue
            gain = state.cartesian_size - refined.cartesian_size
            candidates.append((gain, refined.evidence_mass, refined))

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2].cartesian_size))
        for _, _, refined in candidates[:max_expansions_per_state]:
            signature = (refined.left_ids, refined.right_ids, refined.evidence)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            heapq.heappush(
                heap,
                (refined.cartesian_size, -refined.evidence_mass, counter, refined),
            )
            counter += 1

    rows = sorted(
        emitted_pairs.values(),
        key=lambda row: (-row["certificate_score"], row["id_A"], row["id_B"]),
    )

    if rescue_scores:
        by_left: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rescue_scores.values():
            if row["certificate_score"] < rescue_min_support:
                continue
            by_left[row["id_A"]].append(row)
        for left_id, candidates in by_left.items():
            candidates.sort(
                key=lambda row: (
                    -row["certificate_score"],
                    -row["evidence_count"],
                    row["id_B"],
                )
            )
            for row in candidates[:rescue_top_k_per_left]:
                key = (row["id_A"], row["id_B"])
                if key in emitted_pairs:
                    continue
                emitted_pairs[key] = row

        rows = sorted(
            emitted_pairs.values(),
            key=lambda row: (-row["certificate_score"], row["id_A"], row["id_B"]),
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ],
    )
    stats = {
        "seed_regions": len(regions),
        "region_visits": region_visits,
        "materialized_regions": materialized_regions,
        "candidate_pairs": len(out),
        "rescue_pairs": sum(1 for key in emitted_pairs if key in rescue_scores),
    }
    return out, stats


def semantic_rescue_pairs(
    df_source1: pd.DataFrame,
    df_source2: pd.DataFrame,
    base_candidates_df: pd.DataFrame,
    *,
    top_k_per_left: int = 3,
    min_similarity: float = 0.35,
    symbolic_confidence_threshold: float = 2.2,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Run a semantic rescue only for source1 entities weakly covered by symbolic collapse."""
    if df_source1.empty or df_source2.empty:
        empty = pd.DataFrame(
            columns=[
                "id_A",
                "id_B",
                "certificate_score",
                "evidence_count",
                "witness_path",
                "region_size",
            ]
        )
        return empty, {"semantic_rescue_left_entities": 0, "semantic_rescue_pairs": 0}

    working = base_candidates_df.copy()
    if not working.empty:
        working["id_A"] = working["id_A"].astype(str)
        working["id_B"] = working["id_B"].astype(str)
    left_ids = df_source1["id"].astype(str).tolist()
    right_ids = df_source2["id"].astype(str).tolist()

    weak_left_ids: list[str] = []
    if working.empty:
        weak_left_ids = left_ids
    else:
        grouped = working.groupby("id_A")["certificate_score"].max().to_dict()
        weak_left_ids = [
            entity_id for entity_id in left_ids
            if grouped.get(entity_id, 0.0) < symbolic_confidence_threshold
        ]

    if not weak_left_ids:
        empty = pd.DataFrame(
            columns=[
                "id_A",
                "id_B",
                "certificate_score",
                "evidence_count",
                "witness_path",
                "region_size",
            ]
        )
        return empty, {"semantic_rescue_left_entities": 0, "semantic_rescue_pairs": 0}

    texts_left = _entity_title_texts(df_source1)
    texts_right = _entity_title_texts(df_source2)
    weak_left_ids = [entity_id for entity_id in weak_left_ids if texts_left.get(entity_id, "").strip()]
    right_ids = [entity_id for entity_id in right_ids if texts_right.get(entity_id, "").strip()]
    if not weak_left_ids or not right_ids:
        empty = pd.DataFrame(
            columns=[
                "id_A",
                "id_B",
                "certificate_score",
                "evidence_count",
                "witness_path",
                "region_size",
            ]
        )
        return empty, {"semantic_rescue_left_entities": 0, "semantic_rescue_pairs": 0}

    combined_texts = {entity_id: texts_left[entity_id] for entity_id in weak_left_ids}
    combined_texts.update({entity_id: texts_right[entity_id] for entity_id in right_ids})
    vectors, _idf = _sparse_tfidf_vectors(combined_texts)

    existing_pairs = set(zip(working.get("id_A", pd.Series(dtype=str)).astype(str),
                             working.get("id_B", pd.Series(dtype=str)).astype(str)))
    rows: list[dict[str, Any]] = []
    scored_pairs = 0
    for left_id in weak_left_ids:
        left_vector = vectors.get(left_id, {})
        if not left_vector:
            continue
        ranked = sorted(
            (
                (_sparse_cosine(left_vector, vectors.get(right_id, {})), right_id)
                for right_id in right_ids
            ),
            key=lambda item: (-item[0], item[1]),
        )
        scored_pairs += len(right_ids)
        kept = 0
        for score, right_id in ranked:
            if kept >= top_k_per_left:
                break
            if score < min_similarity:
                break
            if (left_id, right_id) in existing_pairs:
                continue
            kept += 1
            rows.append(
                {
                    "id_A": left_id,
                    "id_B": right_id,
                    "certificate_score": round(1.5 + score, 6),
                    "evidence_count": 1,
                    "witness_path": f"semantic_rescue::{score:.4f}",
                    "region_size": 1,
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ],
    )
    stats = {
        "semantic_rescue_left_entities": len(weak_left_ids),
        "semantic_rescue_pairs": len(out),
        "semantic_rescue_scored_pairs": scored_pairs,
    }
    return out, stats


def strong_witness_rescue_pairs(
    df_source1: pd.DataFrame,
    df_source2: pd.DataFrame,
    witnesses_df: pd.DataFrame,
    base_candidates_df: pd.DataFrame,
    *,
    top_k_per_left: int = 2,
    symbolic_confidence_threshold: float = 2.4,
    min_bridge_score: float = 2.05,
    semantic_bonus_weight: float = 0.35,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Rescue weak entities using strong shared witnesses before predictive scoring."""
    empty = pd.DataFrame(
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ]
    )
    if df_source1.empty or df_source2.empty or witnesses_df.empty:
        return empty, {
            "strong_rescue_left_entities": 0,
            "strong_rescue_pairs": 0,
            "strong_rescue_scored_pairs": 0,
        }

    working = base_candidates_df.copy()
    if not working.empty:
        working["id_A"] = working["id_A"].astype(str)
        working["id_B"] = working["id_B"].astype(str)
        best_by_left = working.groupby("id_A")["certificate_score"].max().to_dict()
        existing_pairs = set(zip(working["id_A"], working["id_B"]))
    else:
        best_by_left = {}
        existing_pairs: set[tuple[str, str]] = set()

    left_ids = df_source1["id"].astype(str).tolist()
    right_ids = df_source2["id"].astype(str).tolist()
    weak_left_ids = [
        entity_id
        for entity_id in left_ids
        if best_by_left.get(entity_id, 0.0) < symbolic_confidence_threshold
    ]
    if not weak_left_ids:
        return empty, {
            "strong_rescue_left_entities": 0,
            "strong_rescue_pairs": 0,
            "strong_rescue_scored_pairs": 0,
        }

    texts_left = _entity_title_texts(df_source1)
    texts_right = _entity_title_texts(df_source2)
    combined_texts = {
        entity_id: text
        for entity_id, text in texts_left.items()
        if entity_id in weak_left_ids and text.strip()
    }
    combined_texts.update(
        {
            entity_id: text
            for entity_id, text in texts_right.items()
            if entity_id in right_ids and text.strip()
        }
    )
    vectors, _idf = _sparse_tfidf_vectors(combined_texts)

    entity_witnesses = _witness_sets_by_entity(witnesses_df)
    right_id_set = set(right_ids)
    right_postings: dict[str, set[str]] = defaultdict(set)
    for right_id in right_id_set:
        for witness_key in entity_witnesses.get(right_id, set()):
            right_postings[witness_key].add(right_id)

    generator_weights = {
        "model_code": 2.2,
        "categorical_model": 1.9,
        "digit_signature": 1.35,
        "rare_triple": 1.4,
        "rare_pair": 1.2,
        "anchor_bigram": 1.15,
        "title_prefix3": 1.0,
        "token_skeleton": 0.95,
        "long_pair": 0.85,
        "prefix_pair": 0.75,
        "categorical_value": 0.85,
        "rare_token": 0.75,
    }
    support_bonus = {
        "block_id": 0.45,
        "field_presence": 0.15,
    }
    high_value_types = {"model_code", "categorical_model", "digit_signature"}

    rows: list[dict[str, Any]] = []
    scored_pairs = 0
    for left_id in weak_left_ids:
        left_witnesses = entity_witnesses.get(left_id, set())
        if not left_witnesses:
            continue

        candidate_support: dict[str, dict[str, Any]] = {}
        for witness_key in left_witnesses:
            witness_type, _witness_value = witness_key.split("::", 1)
            weight = generator_weights.get(witness_type)
            if weight is None:
                continue
            for right_id in right_postings.get(witness_key, set()):
                if (left_id, right_id) in existing_pairs:
                    continue
                bucket = candidate_support.setdefault(
                    right_id,
                    {
                        "type_scores": {},
                        "shared_witnesses": set(),
                    },
                )
                bucket["type_scores"][witness_type] = max(
                    float(weight),
                    float(bucket["type_scores"].get(witness_type, 0.0)),
                )
                bucket["shared_witnesses"].add(witness_key)

        if not candidate_support:
            continue

        scored: list[tuple[float, str, list[str]]] = []
        for right_id, bucket in candidate_support.items():
            support_types = set(bucket["type_scores"])
            if not support_types:
                continue

            scored_pairs += 1
            left_vector = vectors.get(left_id, {})
            right_vector = vectors.get(right_id, {})
            semantic_score = _sparse_cosine(left_vector, right_vector)

            all_shared = left_witnesses & entity_witnesses.get(right_id, set())
            bridge_score = sum(float(bucket["type_scores"][witness_type]) for witness_type in support_types)
            present_bonus_types = {
                witness_key.split("::", 1)[0]
                for witness_key in all_shared
                if witness_key.split("::", 1)[0] in support_bonus
            }
            bridge_score += sum(support_bonus[witness_type] for witness_type in present_bonus_types)
            if present_bonus_types and support_types:
                bridge_score += 0.25
            if len(support_types) >= 2:
                bridge_score += 0.45
            if high_value_types & support_types:
                bridge_score += 0.55
            if {"digit_signature", "token_skeleton"} <= support_types:
                bridge_score += 0.25
            bridge_score += semantic_bonus_weight * semantic_score

            if bridge_score < min_bridge_score:
                continue

            path_types = sorted(support_types | present_bonus_types)
            scored.append((bridge_score, right_id, path_types))

        scored.sort(key=lambda item: (-item[0], item[1]))
        for bridge_score, right_id, path_types in scored[:top_k_per_left]:
            rows.append(
                {
                    "id_A": left_id,
                    "id_B": right_id,
                    "certificate_score": round(1.7 + bridge_score, 6),
                    "evidence_count": len(path_types),
                    "witness_path": "strong_rescue::" + "|".join(path_types),
                    "region_size": 1,
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ],
    )
    return out, {
        "strong_rescue_left_entities": len(weak_left_ids),
        "strong_rescue_pairs": len(out),
        "strong_rescue_scored_pairs": scored_pairs,
    }


def asymmetric_text_rescue_pairs(
    df_source1: pd.DataFrame,
    df_source2: pd.DataFrame,
    witnesses_df: pd.DataFrame,
    base_candidates_df: pd.DataFrame,
    *,
    top_k_per_left: int = 2,
    symbolic_confidence_threshold: float = 2.9,
    shortlist_per_view: int = 6,
    min_rescue_score: float = 0.78,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Rescue short-vs-long title matches missed by earlier symbolic and predictive stages."""
    empty = pd.DataFrame(
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ]
    )
    if df_source1.empty or df_source2.empty:
        return empty, {
            "asymmetric_rescue_left_entities": 0,
            "asymmetric_rescue_pairs": 0,
            "asymmetric_rescue_scored_pairs": 0,
        }

    working = base_candidates_df.copy()
    if not working.empty:
        working["id_A"] = working["id_A"].astype(str)
        working["id_B"] = working["id_B"].astype(str)
        best_by_left = working.groupby("id_A")["certificate_score"].max().to_dict()
        existing_pairs = set(zip(working["id_A"], working["id_B"]))
    else:
        best_by_left = {}
        existing_pairs: set[tuple[str, str]] = set()

    texts_left = _entity_title_texts(df_source1)
    texts_right = _entity_title_texts(df_source2)
    weak_left_ids = [
        str(entity_id)
        for entity_id in df_source1["id"].astype(str).tolist()
        if best_by_left.get(str(entity_id), 0.0) < symbolic_confidence_threshold
        and texts_left.get(str(entity_id), "").strip()
    ]
    right_ids = [
        str(entity_id)
        for entity_id in df_source2["id"].astype(str).tolist()
        if texts_right.get(str(entity_id), "").strip()
    ]
    if not weak_left_ids or not right_ids:
        return empty, {
            "asymmetric_rescue_left_entities": 0,
            "asymmetric_rescue_pairs": 0,
            "asymmetric_rescue_scored_pairs": 0,
        }

    combined_texts = {entity_id: texts_left[entity_id] for entity_id in weak_left_ids}
    combined_texts.update({entity_id: texts_right[entity_id] for entity_id in right_ids})
    token_vectors, _token_idf = _sparse_tfidf_vectors(combined_texts)
    char_vectors, _char_idf = _sparse_char_tfidf_vectors(combined_texts)
    entity_witnesses = _witness_sets_by_entity(witnesses_df)

    right_info_tokens = {
        right_id: _informative_tokens(texts_right.get(right_id, ""))
        for right_id in right_ids
    }

    rows: list[dict[str, Any]] = []
    scored_pairs = 0
    for left_id in weak_left_ids:
        left_token_vector = token_vectors.get(left_id, {})
        left_char_vector = char_vectors.get(left_id, {})
        left_info_tokens = _informative_tokens(texts_left.get(left_id, ""))
        if not left_token_vector and not left_char_vector and not left_info_tokens:
            continue

        token_ranked = sorted(
            (
                (_sparse_cosine(left_token_vector, token_vectors.get(right_id, {})), right_id)
                for right_id in right_ids
                if (left_id, right_id) not in existing_pairs
            ),
            key=lambda item: (-item[0], item[1]),
        )[:shortlist_per_view]
        char_ranked = sorted(
            (
                (_sparse_cosine(left_char_vector, char_vectors.get(right_id, {})), right_id)
                for right_id in right_ids
                if (left_id, right_id) not in existing_pairs
            ),
            key=lambda item: (-item[0], item[1]),
        )[:shortlist_per_view]
        containment_ranked = sorted(
            (
                (_containment_score(left_info_tokens, right_info_tokens.get(right_id, set())), right_id)
                for right_id in right_ids
                if (left_id, right_id) not in existing_pairs
            ),
            key=lambda item: (-item[0], item[1]),
        )[:shortlist_per_view]

        shortlist = {
            right_id
            for score, right_id in [*token_ranked, *char_ranked, *containment_ranked]
            if score > 0.0
        }
        if not shortlist:
            continue

        left_witnesses = entity_witnesses.get(left_id, set())
        scored: list[tuple[float, str, list[str]]] = []
        for right_id in shortlist:
            right_witnesses = entity_witnesses.get(right_id, set())
            token_score = _sparse_cosine(left_token_vector, token_vectors.get(right_id, {}))
            char_score = _sparse_cosine(left_char_vector, char_vectors.get(right_id, {}))
            containment = _containment_score(left_info_tokens, right_info_tokens.get(right_id, set()))
            shared_model = _typed_witness_values(left_witnesses, "model_code") & _typed_witness_values(right_witnesses, "model_code")
            shared_digits = _typed_witness_values(left_witnesses, "digit_signature") & _typed_witness_values(right_witnesses, "digit_signature")
            shared_skeleton = _typed_witness_values(left_witnesses, "token_skeleton") & _typed_witness_values(right_witnesses, "token_skeleton")
            shared_block = _typed_witness_values(left_witnesses, "block_id") & _typed_witness_values(right_witnesses, "block_id")
            shared_anchor = _typed_witness_values(left_witnesses, "anchor_bigram") & _typed_witness_values(right_witnesses, "anchor_bigram")

            support = 0.0
            if shared_model:
                support += 0.18
            if shared_digits:
                support += 0.10
            if shared_skeleton:
                support += 0.08
            if shared_anchor:
                support += 0.08
            if shared_block:
                support += 0.05

            base_score = max(
                token_score,
                char_score,
                0.85 * containment + 0.15 * max(token_score, char_score),
            )
            rescue_score = base_score + support
            scored_pairs += 1
            if rescue_score < min_rescue_score:
                continue

            path: list[str] = []
            if containment >= 0.75:
                path.append("containment")
            if token_score >= 0.45:
                path.append("token")
            if char_score >= 0.45:
                path.append("char")
            if shared_model:
                path.append("model_code")
            if shared_digits:
                path.append("digit_signature")
            if shared_skeleton:
                path.append("token_skeleton")
            if shared_anchor:
                path.append("anchor_bigram")
            if shared_block:
                path.append("block_id")

            scored.append((rescue_score, right_id, path or ["asymmetric"]))

        scored.sort(key=lambda item: (-item[0], item[1]))
        for rescue_score, right_id, path in scored[:top_k_per_left]:
            rows.append(
                {
                    "id_A": left_id,
                    "id_B": right_id,
                    "certificate_score": round(1.65 + rescue_score, 6),
                    "evidence_count": len(path),
                    "witness_path": "asymmetric_rescue::" + "|".join(path),
                    "region_size": 1,
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ],
    )
    return out, {
        "asymmetric_rescue_left_entities": len(weak_left_ids),
        "asymmetric_rescue_pairs": len(out),
        "asymmetric_rescue_scored_pairs": scored_pairs,
    }


def facet_rescue_pairs(
    df_source1: pd.DataFrame,
    df_source2: pd.DataFrame,
    witnesses_df: pd.DataFrame,
    base_candidates_df: pd.DataFrame,
    *,
    top_k_per_left: int = 2,
    symbolic_confidence_threshold: float = 3.1,
    min_facet_score: float = 1.2,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Rescue catalog-like matches via vendor/family/version/edition facets."""
    empty = pd.DataFrame(
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ]
    )
    if df_source1.empty or df_source2.empty:
        return empty, {
            "facet_rescue_left_entities": 0,
            "facet_rescue_pairs": 0,
            "facet_rescue_scored_pairs": 0,
        }

    working = base_candidates_df.copy()
    if not working.empty:
        working["id_A"] = working["id_A"].astype(str)
        working["id_B"] = working["id_B"].astype(str)
        best_by_left = working.groupby("id_A")["certificate_score"].max().to_dict()
        existing_pairs = set(zip(working["id_A"], working["id_B"]))
    else:
        best_by_left = {}
        existing_pairs: set[tuple[str, str]] = set()

    texts_left = _entity_title_texts(df_source1)
    texts_right = _entity_title_texts(df_source2)
    weak_left_ids = [
        str(entity_id)
        for entity_id in df_source1["id"].astype(str).tolist()
        if best_by_left.get(str(entity_id), 0.0) < symbolic_confidence_threshold
        and texts_left.get(str(entity_id), "").strip()
    ]
    right_ids = [
        str(entity_id)
        for entity_id in df_source2["id"].astype(str).tolist()
        if texts_right.get(str(entity_id), "").strip()
    ]
    if not weak_left_ids or not right_ids:
        return empty, {
            "facet_rescue_left_entities": 0,
            "facet_rescue_pairs": 0,
            "facet_rescue_scored_pairs": 0,
        }

    combined_texts = {entity_id: texts_left[entity_id] for entity_id in weak_left_ids}
    combined_texts.update({entity_id: texts_right[entity_id] for entity_id in right_ids})
    token_vectors, _idf = _sparse_tfidf_vectors(combined_texts)
    entity_witnesses = _witness_sets_by_entity(witnesses_df)
    right_facets = {right_id: _catalog_facets(texts_right[right_id]) for right_id in right_ids}

    rows: list[dict[str, Any]] = []
    scored_pairs = 0
    for left_id in weak_left_ids:
        left_facets = _catalog_facets(texts_left[left_id])
        left_vector = token_vectors.get(left_id, {})
        if not left_facets and not left_vector:
            continue

        candidate_ids: set[str] = set()
        left_leads = left_facets.get("lead", set())
        left_versions = left_facets.get("version", set())
        left_lead_pair = left_facets.get("lead_pair", set())
        for right_id in right_ids:
            if (left_id, right_id) in existing_pairs:
                continue
            rf = right_facets[right_id]
            if left_leads & rf.get("lead", set()):
                candidate_ids.add(right_id)
            elif left_versions & rf.get("version", set()):
                candidate_ids.add(right_id)
            elif left_lead_pair & rf.get("lead_pair", set()):
                candidate_ids.add(right_id)

        if not candidate_ids:
            ranked = sorted(
                (
                    (_sparse_cosine(left_vector, token_vectors.get(right_id, {})), right_id)
                    for right_id in right_ids
                    if (left_id, right_id) not in existing_pairs
                ),
                key=lambda item: (-item[0], item[1]),
            )[:8]
            candidate_ids = {right_id for score, right_id in ranked if score > 0.0}

        scored: list[tuple[float, str, list[str]]] = []
        left_witnesses = entity_witnesses.get(left_id, set())
        for right_id in candidate_ids:
            rf = right_facets[right_id]
            right_witnesses = entity_witnesses.get(right_id, set())
            semantic = _sparse_cosine(left_vector, token_vectors.get(right_id, {}))

            lead_overlap = left_facets.get("lead", set()) & rf.get("lead", set())
            family_overlap = left_facets.get("family", set()) & rf.get("family", set())
            version_overlap = left_facets.get("version", set()) & rf.get("version", set())
            edition_overlap = left_facets.get("edition", set()) & rf.get("edition", set())
            pair_overlap = left_facets.get("lead_pair", set()) & rf.get("lead_pair", set())
            triple_overlap = left_facets.get("lead_triple", set()) & rf.get("lead_triple", set())
            descriptor_overlap = left_facets.get("descriptor", set()) & rf.get("descriptor", set())

            facet_score = 0.0
            path: list[str] = []
            if lead_overlap:
                facet_score += 0.35
                path.append("lead")
            if family_overlap:
                facet_score += 0.25 + 0.08 * min(2, len(family_overlap))
                path.append("family")
            if version_overlap:
                facet_score += 0.35
                path.append("version")
            if edition_overlap:
                facet_score += 0.20
                path.append("edition")
            if pair_overlap:
                facet_score += 0.30
                path.append("lead_pair")
            if triple_overlap:
                facet_score += 0.35
                path.append("lead_triple")
            if descriptor_overlap:
                facet_score += 0.12
                path.append("descriptor")

            shared_model = _typed_witness_values(left_witnesses, "model_code") & _typed_witness_values(right_witnesses, "model_code")
            shared_digits = _typed_witness_values(left_witnesses, "digit_signature") & _typed_witness_values(right_witnesses, "digit_signature")
            shared_anchor = _typed_witness_values(left_witnesses, "anchor_bigram") & _typed_witness_values(right_witnesses, "anchor_bigram")
            if shared_model:
                facet_score += 0.18
                path.append("model_code")
            if shared_digits:
                facet_score += 0.10
                path.append("digit_signature")
            if shared_anchor:
                facet_score += 0.08
                path.append("anchor_bigram")

            total_score = facet_score + 0.45 * semantic
            scored_pairs += 1
            if total_score < min_facet_score:
                continue
            scored.append((total_score, right_id, list(dict.fromkeys(path or ["facet"]))))

        scored.sort(key=lambda item: (-item[0], item[1]))
        for total_score, right_id, path in scored[:top_k_per_left]:
            rows.append(
                {
                    "id_A": left_id,
                    "id_B": right_id,
                    "certificate_score": round(1.6 + total_score, 6),
                    "evidence_count": len(path),
                    "witness_path": "facet_rescue::" + "|".join(path),
                    "region_size": 1,
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ],
    )
    return out, {
        "facet_rescue_left_entities": len(weak_left_ids),
        "facet_rescue_pairs": len(out),
        "facet_rescue_scored_pairs": scored_pairs,
    }


def predictive_rescue_pairs(
    df_source1: pd.DataFrame,
    df_source2: pd.DataFrame,
    witnesses_df: pd.DataFrame,
    base_candidates_df: pd.DataFrame,
    *,
    top_k_per_left: int = 2,
    pseudo_positive_threshold: float = 3.0,
    symbolic_confidence_threshold: float = 2.4,
    probability_threshold: float = 0.62,
    semantic_shortlist_k: int = 8,
    min_local_prototype_size: int = 3,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Use high-confidence pairs as pseudo-labels to rescue additional weak pairs."""
    empty = pd.DataFrame(
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ]
    )
    if df_source1.empty or df_source2.empty or base_candidates_df.empty:
        return empty, {
            "predictive_rescue_left_entities": 0,
            "predictive_rescue_pairs": 0,
            "predictive_pseudo_positives": 0,
            "predictive_pseudo_negatives": 0,
        }

    texts_left = _entity_title_texts(df_source1)
    texts_right = _entity_title_texts(df_source2)
    right_ids = [
        str(entity_id)
        for entity_id in df_source2["id"].astype(str).tolist()
        if texts_right.get(str(entity_id), "").strip()
    ]
    if not right_ids:
        return empty, {
            "predictive_rescue_left_entities": 0,
            "predictive_rescue_pairs": 0,
            "predictive_pseudo_positives": 0,
            "predictive_pseudo_negatives": 0,
        }

    entity_witnesses = _witness_sets_by_entity(witnesses_df)
    combined_texts = {entity_id: text for entity_id, text in texts_left.items() if text.strip()}
    combined_texts.update({entity_id: text for entity_id, text in texts_right.items() if text.strip()})
    vectors, _idf = _sparse_tfidf_vectors(combined_texts)

    working = base_candidates_df.copy()
    working["id_A"] = working["id_A"].astype(str)
    working["id_B"] = working["id_B"].astype(str)
    existing_pairs = set(zip(working["id_A"], working["id_B"]))
    best_by_left = working.groupby("id_A")["certificate_score"].max().to_dict()

    pseudo_positive_pairs = [
        (str(row["id_A"]), str(row["id_B"]))
        for _, row in working.iterrows()
        if float(row.get("certificate_score", 0.0)) >= pseudo_positive_threshold
    ]
    if not pseudo_positive_pairs:
        return empty, {
            "predictive_rescue_left_entities": 0,
            "predictive_rescue_pairs": 0,
            "predictive_pseudo_positives": 0,
            "predictive_pseudo_negatives": 0,
        }

    positive_vectors = [
        _pair_feature_vector(
            left_id,
            right_id,
            vectors=vectors,
            texts_left=texts_left,
            texts_right=texts_right,
            entity_witnesses=entity_witnesses,
        )
        for left_id, right_id in pseudo_positive_pairs
    ]
    positive_by_key: dict[str, list[list[float]]] = defaultdict(list)
    for left_id, right_id in pseudo_positive_pairs:
        features = _pair_feature_vector(
            left_id,
            right_id,
            vectors=vectors,
            texts_left=texts_left,
            texts_right=texts_right,
            entity_witnesses=entity_witnesses,
        )
        for key in _pair_prototype_keys(
            left_id,
            right_id,
            texts_left=texts_left,
            texts_right=texts_right,
            entity_witnesses=entity_witnesses,
        ):
            positive_by_key[key].append(features)

    negative_vectors: list[list[float]] = []
    for left_id, pos_right_id in pseudo_positive_pairs[: min(64, len(pseudo_positive_pairs))]:
        left_vector = vectors.get(left_id, {})
        if not left_vector:
            continue
        ranked = sorted(
            (
                (_sparse_cosine(left_vector, vectors.get(right_id, {})), right_id)
                for right_id in right_ids
                if right_id != pos_right_id and (left_id, right_id) not in existing_pairs
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for _score, right_id in ranked[:2]:
            negative_vectors.append(
                _pair_feature_vector(
                    left_id,
                    right_id,
                    vectors=vectors,
                    texts_left=texts_left,
                    texts_right=texts_right,
                    entity_witnesses=entity_witnesses,
                )
            )

    positive_centroid = _centroid(positive_vectors)
    negative_centroid = _centroid(negative_vectors)
    local_positive_centroids = {
        key: _centroid(rows)
        for key, rows in positive_by_key.items()
        if len(rows) >= min_local_prototype_size
    }

    weak_left_ids = [
        str(entity_id)
        for entity_id in df_source1["id"].astype(str).tolist()
        if best_by_left.get(str(entity_id), 0.0) < symbolic_confidence_threshold
        and texts_left.get(str(entity_id), "").strip()
    ]
    if not weak_left_ids:
        return empty, {
            "predictive_rescue_left_entities": 0,
            "predictive_rescue_pairs": 0,
            "predictive_pseudo_positives": len(pseudo_positive_pairs),
            "predictive_pseudo_negatives": len(negative_vectors),
        }

    rows: list[dict[str, Any]] = []
    scored_pairs = 0
    local_hits = 0
    for left_id in weak_left_ids:
        left_vector = vectors.get(left_id, {})
        if not left_vector:
            continue
        ranked_semantic = sorted(
            (
                (_sparse_cosine(left_vector, vectors.get(right_id, {})), right_id)
                for right_id in right_ids
                if (left_id, right_id) not in existing_pairs
            ),
            key=lambda item: (-item[0], item[1]),
        )[:semantic_shortlist_k]

        scored: list[tuple[float, str]] = []
        for semantic_score, right_id in ranked_semantic:
            features = _pair_feature_vector(
                left_id,
                right_id,
                vectors=vectors,
                texts_left=texts_left,
                texts_right=texts_right,
                entity_witnesses=entity_witnesses,
            )
            global_probability = _prototype_probability(features, positive_centroid, negative_centroid)
            local_probabilities = [
                _prototype_probability(features, local_positive_centroids[key], negative_centroid)
                for key in _pair_prototype_keys(
                    left_id,
                    right_id,
                    texts_left=texts_left,
                    texts_right=texts_right,
                    entity_witnesses=entity_witnesses,
                )
                if key in local_positive_centroids
            ]
            if local_probabilities:
                local_hits += 1
            probability = max([global_probability, *local_probabilities])
            probability = max(probability, semantic_score)
            scored_pairs += 1
            scored.append((probability, right_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        kept = 0
        for probability, right_id in scored:
            if kept >= top_k_per_left:
                break
            if probability < probability_threshold:
                break
            kept += 1
            rows.append(
                {
                    "id_A": left_id,
                    "id_B": right_id,
                    "certificate_score": round(1.8 + probability, 6),
                    "evidence_count": 1,
                    "witness_path": f"predictive_rescue::{probability:.4f}",
                    "region_size": 1,
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "id_A",
            "id_B",
            "certificate_score",
            "evidence_count",
            "witness_path",
            "region_size",
        ],
    )
    return out, {
        "predictive_rescue_left_entities": len(weak_left_ids),
        "predictive_rescue_pairs": len(out),
        "predictive_rescue_scored_pairs": scored_pairs,
        "predictive_pseudo_positives": len(pseudo_positive_pairs),
        "predictive_pseudo_negatives": len(negative_vectors),
        "predictive_local_prototypes": len(local_positive_centroids),
        "predictive_local_scored_pairs": local_hits,
    }


def run_constraint_witness_resolution(
    df_source1: pd.DataFrame,
    df_source2: pd.DataFrame,
    *,
    source1: str,
    source2: str,
    blocks_df: pd.DataFrame | None = None,
    max_cartesian_size: int = 8,
    max_region_visits: int = 10000,
    enable_semantic_bands: bool = False,
    semantic_rescue: bool = False,
    predictive_rescue: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Convenience wrapper for the witness-first MVP."""
    witnesses_df = extract_witnesses(
        df_source1,
        df_source2,
        source1=source1,
        source2=source2,
        blocks_df=blocks_df,
        enable_semantic_bands=enable_semantic_bands,
    )
    candidates_df, stats = collapse_witness_regions(
        witnesses_df,
        source1=source1,
        source2=source2,
        max_cartesian_size=max_cartesian_size,
        max_region_visits=max_region_visits,
    )
    if semantic_rescue:
        semantic_df, semantic_stats = semantic_rescue_pairs(
            df_source1,
            df_source2,
            candidates_df,
        )
        if not semantic_df.empty:
            candidates_df = (
                pd.concat([candidates_df, semantic_df], ignore_index=True)
                .sort_values(["certificate_score", "id_A", "id_B"], ascending=[False, True, True])
                .drop_duplicates(subset=["id_A", "id_B"], keep="first")
                .reset_index(drop=True)
            )
        stats = {**stats, **semantic_stats, "candidate_pairs": len(candidates_df)}
    if predictive_rescue:
        predictive_df, predictive_stats = predictive_rescue_pairs(
            df_source1,
            df_source2,
            witnesses_df,
            candidates_df,
        )
        if not predictive_df.empty:
            candidates_df = (
                pd.concat([candidates_df, predictive_df], ignore_index=True)
                .sort_values(["certificate_score", "id_A", "id_B"], ascending=[False, True, True])
                .drop_duplicates(subset=["id_A", "id_B"], keep="first")
                .reset_index(drop=True)
            )
        stats = {**stats, **predictive_stats, "candidate_pairs": len(candidates_df)}
        strong_df, strong_stats = strong_witness_rescue_pairs(
            df_source1,
            df_source2,
            witnesses_df,
            candidates_df,
        )
        if not strong_df.empty:
            candidates_df = (
                pd.concat([candidates_df, strong_df], ignore_index=True)
                .sort_values(["certificate_score", "id_A", "id_B"], ascending=[False, True, True])
                .drop_duplicates(subset=["id_A", "id_B"], keep="first")
                .reset_index(drop=True)
            )
        stats = {**stats, **strong_stats, "candidate_pairs": len(candidates_df)}
        asymmetric_df, asymmetric_stats = asymmetric_text_rescue_pairs(
            df_source1,
            df_source2,
            witnesses_df,
            candidates_df,
        )
        if not asymmetric_df.empty:
            candidates_df = (
                pd.concat([candidates_df, asymmetric_df], ignore_index=True)
                .sort_values(["certificate_score", "id_A", "id_B"], ascending=[False, True, True])
                .drop_duplicates(subset=["id_A", "id_B"], keep="first")
                .reset_index(drop=True)
            )
        stats = {**stats, **asymmetric_stats, "candidate_pairs": len(candidates_df)}
        facet_df, facet_stats = facet_rescue_pairs(
            df_source1,
            df_source2,
            witnesses_df,
            candidates_df,
        )
        if not facet_df.empty:
            candidates_df = (
                pd.concat([candidates_df, facet_df], ignore_index=True)
                .sort_values(["certificate_score", "id_A", "id_B"], ascending=[False, True, True])
                .drop_duplicates(subset=["id_A", "id_B"], keep="first")
                .reset_index(drop=True)
            )
        stats = {**stats, **facet_stats, "candidate_pairs": len(candidates_df)}
    return witnesses_df, candidates_df, stats
