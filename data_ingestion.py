"""
Member 1 - Data Ingestion & Cleaning
=====================================
Supports three datasets:

  Dataset               Type                Files
  ─────────────────────────────────────────────────────────────────
  abt_buy               Semi-structured CSV  Abt.csv / Buy.csv
  amazon_google         Semi-structured CSV  Amazon.csv / GoogleProducts.csv
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
SPIMBENCH_RAW_DIR     = os.environ.get("SPIMBENCH_DIR",
    os.path.join("data", "spimbench", "datasets_and_queries", "datasets"))


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
    mapping_path = os.path.join(raw_dir, "Abt-Buy_perfectMapping.csv")

    for p in [abt_path, buy_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[abt_buy] Cannot find: {p}")

    # latin-1 handles the occasional accented character in product names
    df_abt = pd.read_csv(abt_path, encoding="latin-1")
    df_buy = pd.read_csv(buy_path, encoding="latin-1")

    df_truth = pd.DataFrame()
    if os.path.exists(mapping_path):
        df_truth = pd.read_csv(mapping_path, encoding="latin-1")
        print(f"  [LOAD] Ground truth: {len(df_truth)} pairs from {mapping_path}")
    else:
        print(f"  [WARN] No ground truth found at {mapping_path}")

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
    try:
        import rdflib
    except ImportError:
        raise ImportError(
            "rdflib is required to parse TTL files.\n"
            "Run: pip install rdflib"
        )

    print(f"  [TTL] Parsing {os.path.basename(ttl_path)} ...")
    g = rdflib.Graph()
    g.parse(ttl_path, format="turtle")

    # Collect literal triples  {subject_uri: {pred_local: [values]}}
    entity_data: dict[str, dict[str, list[str]]] = {}

    for subj, pred, obj in g:
        # Only keep literal objects (skip blank nodes and URIs as values)
        if not isinstance(obj, rdflib.term.Literal):
            continue

        subj_str = str(subj)
        # Use only the local name of the predicate (after last / or #)
        pred_local = re.split(r"[/#]", str(pred))[-1].lower()
        obj_str    = str(obj).strip()

        if subj_str not in entity_data:
            entity_data[subj_str] = {}
        entity_data[subj_str].setdefault(pred_local, [])
        entity_data[subj_str][pred_local].append(obj_str)

    if not entity_data:
        print(f"  [WARN] No literal triples found in {ttl_path}")
        return pd.DataFrame()

    # Build flat rows — multi-valued predicates joined with space
    rows = []
    for subj_uri, attrs in entity_data.items():
        row = {"id": subj_uri}
        for pred_local, values in attrs.items():
            row[pred_local] = " ".join(values)
        rows.append(row)

    df = pd.DataFrame(rows).fillna("")
    print(f"  [TTL] {os.path.basename(ttl_path)}: {len(df)} entities, "
          f"{len(df.columns) - 1} attributes")
    return df


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
        