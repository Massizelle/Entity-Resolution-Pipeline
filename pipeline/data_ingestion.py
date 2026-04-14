"""
Member 1 - Data Ingestion & Cleaning
=====================================
Supports four tabular/RDF datasets plus one synthetic RDF benchmark:

  Dataset               Type                Files
  ─────────────────────────────────────────────────────────────────
  abt_buy               Semi-structured CSV  Abt.csv / Buy.csv
  amazon_google         Semi-structured CSV  Amazon.csv / GoogleProducts.csv
  dblp_acm              Structured CSV       tableA.csv / tableB.csv
  rexa_dblp             RDF / OAEI           DBLP RDF + Rexa RDF + alignment
  spimbench             RDF / TTL            dbpedia_*.ttl (multiple files)

Ground truth files (*_perfectMapping.csv) are loaded automatically.
Data is read from the RAW_DIR structure you already have on disk —
no download needed, just pass --no-download.

Output structure:
  data/cleaned/<dataset_key>/
    cleaned_source1.csv
    cleaned_source2.csv
    ground_truth.csv        (if a perfectMapping exists)
"""

from __future__ import annotations

import os
import re
import gzip
import xml.etree.ElementTree as ET
import pandas as pd


# ── Directory layout ──────────────────────────────────────────────────────────
DATA_DIR  = "data"
RAW_DIR   = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")

# Absolute paths to the raw folders (matching your Windows layout)
# The pipeline works with relative paths when run from the project root;
# override these via environment variables if needed.
ABT_BUY_RAW_DIR      = os.environ.get("ABT_BUY_DIR",
    os.path.join(RAW_DIR, "Abt-Buy"))
AMAZON_GOOGLE_RAW_DIR = os.environ.get("AMAZON_GOOGLE_DIR",
    os.path.join(RAW_DIR, "Amazon-GoogleProducts"))
DBLP_ACM_RAW_DIR      = os.environ.get("DBLP_ACM_DIR",
    os.path.join(RAW_DIR, "DBLP-ACM"))
SPIMBENCH_RAW_DIR     = os.environ.get("SPIMBENCH_DIR",
    os.path.join("data", "spimbench", "datasets_and_queries", "datasets"))
REXA_DBLP_RAW_DIR     = os.environ.get("REXA_DBLP_DIR",
    os.path.join(RAW_DIR, "Rexa-DBLP"))

RDF_EXTENSIONS = (".rdf", ".owl", ".xml", ".ttl", ".nt", ".n3")


# ── Text normalisation ────────────────────────────────────────────────────────

def normalize_text(text) -> str:
    """Lowercase, strip special characters, collapse whitespace."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_uri(uri) -> str:
    """
    Convert an RDF URI to a readable token string.
    e.g. 'http://dbpedia.org/resource/Stanley_Kubrick' → 'stanley kubrick'
    """
    if pd.isna(uri) or not isinstance(uri, str):
        return ""
    local = re.split(r"[/#]", uri.strip())[-1]
    return normalize_text(local.replace("_", " "))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique values while preserving their first-seen order."""
    return list(dict.fromkeys(v for v in values if v))


# ── General DataFrame cleaner ─────────────────────────────────────────────────

def clean_dataframe(
    df: pd.DataFrame,
    source_name: str,
    id_col: str | None = None,
    uri_columns: list | None = None,
) -> pd.DataFrame:
    """
    Standardise any entity DataFrame:
      - Ensure an 'id' column exists
      - Normalize all string columns (lowercase, strip specials)
      - Add a 'source' column
    """
    uri_columns = uri_columns or []
    print(f"  [CLEAN] '{source_name}' — before: {df.shape}")

    df = df.dropna(how="all").reset_index(drop=True)

    # ── Resolve id column ────────────────────────────────────────────────
    if id_col and id_col in df.columns:
        df = df.rename(columns={id_col: "id"})
    elif "id" not in df.columns:
        for alt in ["Id", "ID", "idABT", "idBuy", "idAmazon",
                    "idGoogleBase", "uri", "URI"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "id"})
                break
        else:
            df.insert(0, "id", [f"{source_name}_{i}" for i in range(len(df))])

    df["id"] = df["id"].astype(str).str.strip()

    # ── Normalize text columns ───────────────────────────────────────────
    for col in df.columns:
        if col in ("id", "source"):
            continue
        if col in uri_columns:
            df[col] = df[col].apply(normalize_uri)
        elif df[col].dtype == object:
            df[col] = df[col].apply(normalize_text)

    df = df.fillna("")
    df["source"] = source_name

    print(f"  [CLEAN] '{source_name}' — after:  {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Dataset-specific loaders
# ══════════════════════════════════════════════════════════════════════════════

# ── Abt-Buy ───────────────────────────────────────────────────────────────────

def load_abt_buy(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Abt.csv, Buy.csv and Abt-Buy_perfectMapping.csv.

    Expected columns:
      Abt.csv       : id, name, description, price
      Buy.csv       : id, name, description, price, manufacturer
      perfectMapping: idAbt, idBuy
    """
    abt_path     = os.path.join(raw_dir, "Abt.csv")
    buy_path     = os.path.join(raw_dir, "Buy.csv")
    mapping_candidates = [
        "Abt-Buy_perfectMapping.csv",
        "abt_buy_perfectMapping.csv",
        "Abt_Buy_perfectMapping.csv",
    ]
    mapping_path = None
    for candidate in mapping_candidates:
        p = os.path.join(raw_dir, candidate)
        if os.path.exists(p):
            mapping_path = p
            break

    for p in [abt_path, buy_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[abt_buy] Cannot find: {p}")

    # latin-1 handles the occasional accented character in product names
    df_abt = pd.read_csv(abt_path, encoding="latin-1")
    df_buy = pd.read_csv(buy_path, encoding="latin-1")

    df_truth = pd.DataFrame()
    if mapping_path:
        df_truth = pd.read_csv(mapping_path, encoding="latin-1")
        print(f"  [LOAD] Ground truth: {len(df_truth)} pairs from {mapping_path}")
    else:
        print(f"  [WARN] No ground truth found for abt_buy in {raw_dir}")

    return df_abt, df_buy, df_truth


# ── Amazon-GoogleProducts ─────────────────────────────────────────────────────

def load_amazon_google(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Amazon.csv, GoogleProducts.csv and Amzon_GoogleProducts_perfectMapping.csv.

    Expected columns:
      Amazon.csv        : id, title, description, manufacturer, price
      GoogleProducts.csv: id, name, description, manufacturer, price
      perfectMapping    : idAmazon, idGoogleBase
    """
    amazon_path  = os.path.join(raw_dir, "Amazon.csv")
    google_path  = os.path.join(raw_dir, "GoogleProducts.csv")

    # The filename has a typo in the original dataset ("Amzon" not "Amazon")
    mapping_candidates = [
        "Amzon_GoogleProducts_perfectMapping.csv",
        "Amazon_GoogleProducts_perfectMapping.csv",
        "Amazon-GoogleProducts_perfectMapping.csv",
    ]
    mapping_path = None
    for candidate in mapping_candidates:
        p = os.path.join(raw_dir, candidate)
        if os.path.exists(p):
            mapping_path = p
            break

    for p in [amazon_path, google_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[amazon_google] Cannot find: {p}")

    df_amazon = pd.read_csv(amazon_path, encoding="latin-1")
    df_google  = pd.read_csv(google_path, encoding="latin-1")

    df_truth = pd.DataFrame()
    if mapping_path:
        df_truth = pd.read_csv(mapping_path, encoding="latin-1")
        print(f"  [LOAD] Ground truth: {len(df_truth)} pairs from {mapping_path}")
    else:
        print("  [WARN] No ground truth found for amazon_google")

    return df_amazon, df_google, df_truth


# ── DBLP-ACM ──────────────────────────────────────────────────────────────────

def load_dblp_acm(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load DBLP-ACM from MatchBench-style CSV files.

    Expected files:
      tableA.csv: DBLP records
      tableB.csv: ACM records
      train.csv / valid.csv / test.csv: ltable_id, rtable_id, label

    The pipeline consumes the union of positive labeled pairs as ground truth.
    """
    table_a_path = os.path.join(raw_dir, "tableA.csv")
    table_b_path = os.path.join(raw_dir, "tableB.csv")
    split_paths = [
        os.path.join(raw_dir, "train.csv"),
        os.path.join(raw_dir, "valid.csv"),
        os.path.join(raw_dir, "test.csv"),
    ]

    for path in [table_a_path, table_b_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[dblp_acm] Cannot find: {path}")

    df_a = pd.read_csv(table_a_path, encoding="utf-8")
    df_b = pd.read_csv(table_b_path, encoding="utf-8")

    truth_frames: list[pd.DataFrame] = []
    for split_path in split_paths:
        if not os.path.exists(split_path):
            continue
        split_df = pd.read_csv(split_path, encoding="utf-8")
        if {"ltable_id", "rtable_id", "label"} <= set(split_df.columns):
            pos_df = split_df[split_df["label"].astype(int) == 1].copy()
            if not pos_df.empty:
                truth_frames.append(pos_df[["ltable_id", "rtable_id"]])

    if truth_frames:
        df_truth = pd.concat(truth_frames, ignore_index=True).drop_duplicates()
        df_truth = df_truth.rename(columns={
            "ltable_id": "idDBLP",
            "rtable_id": "idACM",
        })
        print(f"  [LOAD] Ground truth: {len(df_truth)} positive pairs from train/valid/test")
    else:
        df_truth = pd.DataFrame(columns=["idDBLP", "idACM"])
        print(f"  [WARN] No labeled positive pairs found for dblp_acm in {raw_dir}")

    return df_a, df_b, df_truth


# ── SPIMBENCH (RDF / TTL) ─────────────────────────────────────────────────────

# TTL files shipped with SPIMBENCH — we treat the *_ldbc variants as "source 1"
# and the plain variants as "source 2" when both exist.
# For the evaluation, SPIMBENCH uses its own validation queries; we expose
# the entity list so blocking/matching can run on it.

SPIMBENCH_TTL_FILES = [
    "dbpedia_Event.ttl",
    "dbpedia_Organisation.ttl",
    "dbpedia_Person_ldbc.ttl",
    "dbpedia_Place.ttl",
    "dbpedia_Sport.ttl",
    "english-football-competitions-1.ttl",
    "english-football-teams-2.ttl",
    "formula1-competitions-8.ttl",
    "formula1-teams-3.ttl",
    "geonames-GB.ttl",
    "international-football-competitions-3.ttl",
    "international-football-teams-2.ttl",
    "scottish-football-competitions-1.ttl",
    "scottish-football-teams-2.ttl",
    "travel_original_.ttl",
    "UK-Parliament-Identifiers-People-8.ttl",
]


def _parse_ttl_to_dataframe(ttl_path: str, source_name: str) -> pd.DataFrame:
    """
    Parse a TTL (Turtle RDF) file into a flat DataFrame using rdflib.

    Strategy: collect all (subject, predicate, object) triples where the
    object is a literal. Each subject becomes an entity; each predicate
    becomes an attribute column. The result is a wide DataFrame.
    """
    return _parse_rdf_to_dataframe(ttl_path, source_name=source_name)


def _infer_rdf_format(path: str) -> str:
    """Infer the rdflib parser format from a file extension."""
    base = path[:-3] if path.endswith(".gz") else path
    lower = base.lower()
    if lower.endswith((".rdf", ".owl", ".xml")):
        return "xml"
    if lower.endswith(".ttl"):
        return "turtle"
    if lower.endswith(".nt"):
        return "nt"
    if lower.endswith(".n3"):
        return "n3"
    raise ValueError(f"Unsupported RDF file extension: {path}")


def _parse_rdf_to_dataframe(rdf_path: str, source_name: str) -> pd.DataFrame:
    """
    Parse an RDF file into a flat DataFrame.

    We keep:
    - literal objects as text attributes
    - URI objects as normalized local-name text

    This makes benchmarks like Rexa-DBLP usable in the same flat pipeline as the
    CSV datasets, while still retaining some graph structure such as authors,
    venues, and rdf:type.
    """
    rdf_format = _infer_rdf_format(rdf_path)
    label = os.path.basename(rdf_path)
    print(f"  [RDF] Parsing {label} ({rdf_format}) ...")

    try:
        import rdflib
    except ImportError:
        if rdf_format == "xml":
            return _parse_rdf_xml_without_rdflib(rdf_path, label=label)
        raise ImportError(
            "rdflib is required to parse non-XML RDF files.\n"
            "Run: pip install rdflib"
        )

    graph = rdflib.Graph()
    if rdf_path.endswith(".gz"):
        with gzip.open(rdf_path, "rt", encoding="utf-8", errors="ignore") as fh:
            graph.parse(data=fh.read(), format=rdf_format)
    else:
        graph.parse(rdf_path, format=rdf_format)

    entity_data: dict[str, dict[str, list[str]]] = {}
    literal_triples = 0
    uri_object_triples = 0

    for subj, pred, obj in graph:
        subj_str = str(subj)
        pred_local = re.split(r"[/#]", str(pred))[-1].lower() or "attribute"
        if pred_local == "type":
            pred_local = "entity_type"

        value = ""
        if isinstance(obj, rdflib.term.Literal):
            value = str(obj).strip()
            literal_triples += 1
        elif isinstance(obj, rdflib.term.URIRef):
            value = normalize_uri(str(obj))
            uri_object_triples += 1
        else:
            continue

        if not value:
            continue

        entity_data.setdefault(subj_str, {}).setdefault(pred_local, []).append(value)

    if not entity_data:
        print(f"  [WARN] No usable triples found in {rdf_path}")
        return pd.DataFrame()

    rows = []
    for subj_uri, attrs in entity_data.items():
        row = {"id": subj_uri}
        for pred_local, values in attrs.items():
            row[pred_local] = " ".join(_dedupe_preserve_order(values))
        rows.append(row)

    df = pd.DataFrame(rows).fillna("")
    print(
        f"  [RDF] {label}: {len(df)} entities, {len(df.columns) - 1} attributes "
        f"(literal triples={literal_triples}, uri-object triples={uri_object_triples})"
    )
    return df


def _parse_rdf_xml_without_rdflib(rdf_path: str, *, label: str) -> pd.DataFrame:
    """
    Minimal RDF/XML parser used when rdflib is unavailable.

    It is intentionally simple:
    - reads subjects from rdf:about / rdf:ID / rdf:nodeID
    - keeps literal child text values
    - keeps URI child objects via rdf:resource, normalized to local names
    """
    if rdf_path.endswith(".gz"):
        with gzip.open(rdf_path, "rt", encoding="utf-8", errors="ignore") as fh:
            root = ET.fromstring(fh.read())
    else:
        root = ET.parse(rdf_path).getroot()

    entity_data: dict[str, dict[str, list[str]]] = {}
    literal_triples = 0
    uri_object_triples = 0

    for entity_elem in root:
        subj_uri = ""
        for attr_name, value in entity_elem.attrib.items():
            if attr_name.endswith(("about", "ID", "nodeID")) and value:
                subj_uri = value.strip()
                break
        if not subj_uri:
            continue

        for child in entity_elem:
            pred_local = re.split(r"[}/#]", child.tag)[-1].lower() or "attribute"
            if pred_local == "type":
                pred_local = "entity_type"

            value = ""
            resource = _extract_resource_uri(child)
            if resource:
                value = normalize_uri(resource)
                uri_object_triples += 1
            elif child.text and child.text.strip():
                value = child.text.strip()
                literal_triples += 1

            if not value:
                continue

            entity_data.setdefault(subj_uri, {}).setdefault(pred_local, []).append(value)

    if not entity_data:
        print(f"  [WARN] No usable triples found in {rdf_path}")
        return pd.DataFrame()

    rows = []
    for subj_uri, attrs in entity_data.items():
        row = {"id": subj_uri}
        for pred_local, values in attrs.items():
            row[pred_local] = " ".join(_dedupe_preserve_order(values))
        rows.append(row)

    df = pd.DataFrame(rows).fillna("")
    print(
        f"  [RDF] {label}: {len(df)} entities, {len(df.columns) - 1} attributes "
        f"(fallback XML parser; literal triples={literal_triples}, "
        f"uri-object triples={uri_object_triples})"
    )
    return df


def _find_existing_file(raw_dir: str, candidates: list[str]) -> str | None:
    """Return the first existing candidate path under raw_dir."""
    for candidate in candidates:
        path = os.path.join(raw_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def _discover_rdf_source_path(raw_dir: str, keywords: list[str]) -> str | None:
    """
    Discover an RDF source file by filename keywords when canonical names are absent.
    """
    for fname in sorted(os.listdir(raw_dir)):
        lower = fname.lower()
        if not lower.endswith(RDF_EXTENSIONS) and not any(
            lower.endswith(ext + ".gz") for ext in RDF_EXTENSIONS
        ):
            continue
        if all(keyword in lower for keyword in keywords):
            return os.path.join(raw_dir, fname)
    return None


def _extract_resource_uri(elem: ET.Element | None) -> str:
    """Extract rdf:resource or resource from an XML element."""
    if elem is None:
        return ""
    for attr_name, value in elem.attrib.items():
        if attr_name.endswith("resource") and value:
            return value.strip()
    if elem.text and elem.text.strip():
        return elem.text.strip()
    return ""


def _parse_oaei_alignment(alignment_path: str) -> pd.DataFrame:
    """
    Parse a standard OAEI alignment RDF file into a generic two-column DataFrame.
    """
    tree = ET.parse(alignment_path)
    root = tree.getroot()
    rows: list[dict[str, str]] = []

    for cell in root.iter():
        if not cell.tag.endswith("Cell"):
            continue

        entity1 = None
        entity2 = None
        relation = ""
        measure = ""

        for child in cell:
            if child.tag.endswith("entity1"):
                entity1 = _extract_resource_uri(child)
            elif child.tag.endswith("entity2"):
                entity2 = _extract_resource_uri(child)
            elif child.tag.endswith("relation"):
                relation = (child.text or "").strip()
            elif child.tag.endswith("measure"):
                measure = (child.text or "").strip()

        if entity1 and entity2:
            rows.append({
                "id_left": entity1,
                "id_right": entity2,
                "relation": relation,
                "measure": measure,
            })

    return pd.DataFrame(rows)


def _coerce_mapping_dataframe(df_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a CSV mapping file to two generic ID columns when possible.
    """
    if df_truth.empty:
        return df_truth

    preferred_pairs = [
        ("idDBLP", "idRexa"),
        ("id_dblp", "id_rexa"),
        ("id_A", "id_B"),
        ("entity1", "entity2"),
        ("left_id", "right_id"),
    ]
    for left_col, right_col in preferred_pairs:
        if left_col in df_truth.columns and right_col in df_truth.columns:
            return df_truth.rename(columns={left_col: "id_left", right_col: "id_right"})

    if len(df_truth.columns) >= 2:
        left_col, right_col = df_truth.columns[:2]
        return df_truth.rename(columns={left_col: "id_left", right_col: "id_right"})

    return pd.DataFrame()


def _orient_truth_pairs(
    df_truth: pd.DataFrame,
    source1_ids: set[str],
    source2_ids: set[str],
    source1_col: str,
    source2_col: str,
) -> pd.DataFrame:
    """
    Orient generic truth pairs so their columns match the chosen source order.
    """
    if df_truth.empty or "id_left" not in df_truth.columns or "id_right" not in df_truth.columns:
        return pd.DataFrame(columns=[source1_col, source2_col])

    left = df_truth["id_left"].astype(str).str.strip()
    right = df_truth["id_right"].astype(str).str.strip()

    direct_mask = left.isin(source1_ids) & right.isin(source2_ids)
    swapped_mask = left.isin(source2_ids) & right.isin(source1_ids)

    rows = []
    if direct_mask.any():
        rows.append(pd.DataFrame({
            source1_col: left[direct_mask],
            source2_col: right[direct_mask],
        }))
    if swapped_mask.any():
        rows.append(pd.DataFrame({
            source1_col: right[swapped_mask],
            source2_col: left[swapped_mask],
        }))

    if not rows:
        return pd.DataFrame(columns=[source1_col, source2_col])

    return pd.concat(rows, ignore_index=True).drop_duplicates()


def load_rexa_dblp(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the Rexa-DBLP/OAEI benchmark from RDF sources and an optional alignment.

    Expected raw directory contents:
    - one RDF file for DBLP
    - one RDF file for Rexa
    - optionally one OAEI alignment RDF or a two-column CSV mapping
    """
    dblp_path = _find_existing_file(raw_dir, [
        "dblp.rdf",
        "dblp.owl",
        "dblp.xml",
        "dblp.nt",
        "dblp.ttl",
        "dblp.n3",
        "dblp.rdf.gz",
        "swetodblp_april_2008.rdf",
        "swetodblp_april_2008.rdf.gz",
    ]) or _discover_rdf_source_path(raw_dir, ["dblp"])

    rexa_path = _find_existing_file(raw_dir, [
        "rexa.rdf",
        "rexa.owl",
        "rexa.xml",
        "rexa.nt",
        "rexa.ttl",
        "rexa.n3",
        "rexa.rdf.gz",
    ]) or _discover_rdf_source_path(raw_dir, ["rexa"])

    if not dblp_path or not rexa_path:
        raise FileNotFoundError(
            "[rexa_dblp] Could not locate both RDF sources in "
            f"{raw_dir}. Expected one DBLP file and one Rexa file."
        )

    df_dblp = _parse_rdf_to_dataframe(dblp_path, source_name="dblp")
    df_rexa = _parse_rdf_to_dataframe(rexa_path, source_name="rexa")

    mapping_path = _find_existing_file(raw_dir, [
        "refalign.rdf",
        "reference.rdf",
        "reference.xml",
        "alignment.rdf",
        "alignment.xml",
        "gold_standard.rdf",
        "gold_standard.xml",
        "ground_truth.csv",
        "perfectMapping.csv",
        "rexa_dblp_perfectMapping.csv",
    ])

    df_truth = pd.DataFrame(columns=["idDBLP", "idRexa"])
    if mapping_path:
        if mapping_path.lower().endswith(".csv"):
            raw_truth = _coerce_mapping_dataframe(pd.read_csv(mapping_path, encoding="utf-8"))
        else:
            raw_truth = _parse_oaei_alignment(mapping_path)
        df_truth = _orient_truth_pairs(
            raw_truth,
            source1_ids=set(df_dblp["id"].astype(str)),
            source2_ids=set(df_rexa["id"].astype(str)),
            source1_col="idDBLP",
            source2_col="idRexa",
        )
        print(f"  [LOAD] Ground truth: {len(df_truth)} pairs from {mapping_path}")
    else:
        print(f"  [WARN] No ground truth found for rexa_dblp in {raw_dir}")

    return df_dblp, df_rexa, df_truth


def load_spimbench(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all SPIMBENCH TTL files, split them into two virtual sources:
      - source 1 ("spimbench_a"): first half of TTL files alphabetically
      - source 2 ("spimbench_b"): second half

    SPIMBENCH is a synthetic benchmark — it does not ship a CSV ground truth.
    The ground truth / validation logic is embedded in its SPARQL queries.
    We return an empty df_truth and note this in the output.

    All TTL files present in raw_dir are parsed and concatenated.
    """
    available = [
        f for f in SPIMBENCH_TTL_FILES
        if os.path.exists(os.path.join(raw_dir, f))
    ]

    if not available:
        # Fallback: pick up any .ttl file in the directory
        available = [f for f in os.listdir(raw_dir) if f.endswith(".ttl")]

    if not available:
        raise FileNotFoundError(
            f"[spimbench] No TTL files found in {raw_dir}"
        )

    print(f"  [SPIMBENCH] Found {len(available)} TTL files")

    frames: list[pd.DataFrame] = []
    for fname in sorted(available):
        path = os.path.join(raw_dir, fname)
        df   = _parse_ttl_to_dataframe(path, source_name="spimbench")
        if not df.empty:
            # Tag each entity with its source file so we can trace it back
            df["ttl_source"] = fname
            frames.append(df)

    if not frames:
        raise ValueError("[spimbench] All TTL files were empty or unparsable.")

    df_all = pd.concat(frames, ignore_index=True).fillna("")

    # Split into two virtual sources: odd/even index — gives balanced halves
    # and ensures entities from different files can be compared.
    df_s1 = df_all.iloc[::2].copy().reset_index(drop=True)
    df_s2 = df_all.iloc[1::2].copy().reset_index(drop=True)

    print(f"  [SPIMBENCH] Total entities: {len(df_all)} "
          f"→ source_a={len(df_s1)}, source_b={len(df_s2)}")

    # SPIMBENCH ground truth is validated via its SPARQL engine — no CSV mapping
    df_truth = pd.DataFrame()

    return df_s1, df_s2, df_truth


# ══════════════════════════════════════════════════════════════════════════════
# Dataset registry
# ══════════════════════════════════════════════════════════════════════════════

DATASET_REGISTRY: dict[str, dict] = {
    "abt_buy": {
        "loader":       load_abt_buy,
        "raw_dir":      ABT_BUY_RAW_DIR,
        "source1":      "abt",
        "source2":      "buy",
        "data_type":    "Semi-structured CSV",
        "description":  "Product listings — Abt.com vs Buy.com",
        "truth_col_s1": "idAbt",
        "truth_col_s2": "idBuy",
        "uri_cols_s1":  [],
        "uri_cols_s2":  [],
    },
    "amazon_google": {
        "loader":       load_amazon_google,
        "raw_dir":      AMAZON_GOOGLE_RAW_DIR,
        "source1":      "amazon",
        "source2":      "google",
        "data_type":    "Semi-structured CSV",
        "description":  "Product listings — Amazon vs Google Shopping",
        "truth_col_s1": "idAmazon",
        "truth_col_s2": "idGoogleBase",
        "uri_cols_s1":  [],
        "uri_cols_s2":  [],
    },
    "dblp_acm": {
        "loader":       load_dblp_acm,
        "raw_dir":      DBLP_ACM_RAW_DIR,
        "source1":      "dblp",
        "source2":      "acm",
        "data_type":    "Structured CSV",
        "description":  "Bibliographic ER benchmark — DBLP vs ACM",
        "truth_col_s1": "idDBLP",
        "truth_col_s2": "idACM",
        "uri_cols_s1":  [],
        "uri_cols_s2":  [],
    },
    "spimbench": {
        "loader":       load_spimbench,
        "raw_dir":      SPIMBENCH_RAW_DIR,
        "source1":      "spimbench_a",
        "source2":      "spimbench_b",
        "data_type":    "RDF / TTL",
        "description":  "Synthetic ER benchmark — SPIMBENCH DBpedia TTL files",
        "truth_col_s1": None,   # no CSV ground truth for SPIMBENCH
        "truth_col_s2": None,
        "uri_cols_s1":  ["id"],
        "uri_cols_s2":  ["id"],
    },
    "rexa_dblp": {
        "loader":       load_rexa_dblp,
        "raw_dir":      REXA_DBLP_RAW_DIR,
        "source1":      "dblp",
        "source2":      "rexa",
        "data_type":    "RDF / OAEI alignment",
        "description":  "Bibliographic ER benchmark — DBLP vs Rexa",
        "truth_col_s1": "idDBLP",
        "truth_col_s2": "idRexa",
        "uri_cols_s1":  [],
        "uri_cols_s2":  [],
    },
}


# ── Generic ingestion pipeline ────────────────────────────────────────────────

def run_ingestion(
    dataset: str = "amazon_google",
    download: bool = False,          # all datasets are already local
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full ingestion pipeline for ONE dataset.

    Returns: (df_source1_clean, df_source2_clean, df_truth)
    """
    if dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}"
        )

    cfg     = DATASET_REGISTRY[dataset]
    raw_dir = cfg["raw_dir"]

    print(f"\n[INGEST] Dataset  : {dataset.upper()}")
    print(f"[INGEST] Type     : {cfg['data_type']}")
    print(f"[INGEST] Raw dir  : {raw_dir}")

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"Raw directory not found: {raw_dir}\n"
            f"Check that the path is correct or set the environment variable "
            f"for this dataset."
        )

    # Load raw DataFrames
    df1_raw, df2_raw, df_truth = cfg["loader"](raw_dir)

    # Clean
    df1 = clean_dataframe(df1_raw, source_name=cfg["source1"],
                          uri_columns=cfg["uri_cols_s1"])
    df2 = clean_dataframe(df2_raw, source_name=cfg["source2"],
                          uri_columns=cfg["uri_cols_s2"])

    # Save cleaned files
    out_dir = os.path.join(CLEAN_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)

    s1_out    = os.path.join(out_dir, "cleaned_source1.csv")
    s2_out    = os.path.join(out_dir, "cleaned_source2.csv")
    truth_out = os.path.join(out_dir, "ground_truth.csv")

    df1.to_csv(s1_out, index=False)
    df2.to_csv(s2_out, index=False)
    if not df_truth.empty:
        df_truth.to_csv(truth_out, index=False)
        print(f"  [SAVE] {truth_out}")

    print(f"  [SAVE] {s1_out}")
    print(f"  [SAVE] {s2_out}")

    return df1, df2, df_truth


def run_all_ingestions(download: bool = False) -> dict:
    """Run ingestion for all three datasets."""
    results = {}
    for key in DATASET_REGISTRY:
        results[key] = run_ingestion(dataset=key, download=download)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    valid = list(DATASET_REGISTRY.keys()) + ["all"]
    parser = argparse.ArgumentParser(
        description="Member 1 — Data Ingestion & Cleaning"
    )
    parser.add_argument("--dataset", choices=valid, default="all")
    parser.add_argument("--no-download", action="store_true",
                        help="(default) data is already local, no download")
    args = parser.parse_args()

    if args.dataset == "all":
        results = run_all_ingestions()
        print("\n" + "=" * 60)
        print("  INGESTION SUMMARY")
        print("=" * 60)
        for key, (df1, df2, df_t) in results.items():
            cfg = DATASET_REGISTRY[key]
            print(f"\n  [{cfg['data_type']}]  {key}")
            print(f"    {cfg['source1']:15s}: {len(df1):>6} entities")
            print(f"    {cfg['source2']:15s}: {len(df2):>6} entities")
            print(f"    ground truth   : {len(df_t):>6} pairs")
    else:
        df1, df2, df_t = run_ingestion(dataset=args.dataset)
        cfg = DATASET_REGISTRY[args.dataset]
        print(f"\n[DONE] {args.dataset}")
        print(f"  {cfg['source1']:15s}: {len(df1)} entities")
        print(f"  {cfg['source2']:15s}: {len(df2)} entities")
        print(f"  ground truth   : {len(df_t)} pairs")
        
