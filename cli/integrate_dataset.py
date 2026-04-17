"""
cli/integrate_dataset.py — Generic Dataset Integrator
======================================================
Analyses a directory (or .zip) of raw entity-resolution data and integrates
it into the pipeline automatically, regardless of format.

Supported formats (auto-detected by extension + content):
  .csv / .tsv      — any separator (auto-sniffed), any encoding
  .parquet         — via pandas
  .json / .jsonl   — via pandas
  .ttl / .n3       — RDF Turtle  (rdflib or manual line-parser fallback)
  .nt              — RDF N-Triples
  .rdf / .owl / .xml — RDF-XML or OAEI Cell-alignment
  .zip             — extracted automatically before analysis

Usage:
    # Analyse only (dry-run)
    python cli/integrate_dataset.py --source data/raw/MyDataset --name my_dataset --dry-run

    # Full integration
    python cli/integrate_dataset.py --source data/raw/MyDataset --name my_dataset \\
        --source1-name left --source2-name right

    # Integrate + run pipeline immediately
    python cli/integrate_dataset.py --source data/raw/DBpedia-IMDb --name dbpedia_imdb --run

After integration the dataset is available in the pipeline as:
    ./venv/bin/python cli/run_pipeline.py --dataset <name>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXTRA_REGISTRY_PATH = REPO_ROOT / "pipeline" / "extra_datasets.json"
CLEAN_DIR = REPO_ROOT / "data" / "cleaned"
RAW_DIR   = REPO_ROOT / "data" / "raw"

# ── Extension groups ──────────────────────────────────────────────────────────

TABULAR_EXTS  = {".csv", ".tsv", ".txt", ".dat"}
PARQUET_EXTS  = {".parquet", ".pq"}
JSON_EXTS     = {".json", ".jsonl", ".ndjson"}
RDF_TURTLE    = {".ttl", ".n3"}
RDF_NT        = {".nt"}
RDF_XML       = {".rdf", ".owl", ".xml"}
ARCHIVE_EXTS  = {".zip"}
SKIP_EXTS     = {".md", ".pdf", ".png", ".jpg", ".py", ".sh", ".txt"}

# Role-detection keyword sets (lower-case, no separators)
GT_KEYWORDS       = {"perfectmapping", "groundtruth", "gt", "gold", "matches",
                     "alignment", "links", "mapping", "reference", "positives"}
SOURCE_A_KEYWORDS = {"tablea", "tablea", "source1", "sourcea", "left", "abt",
                     "amazon", "walmart", "dblp1", "dblp", "dbpedia", "rexa",
                     "tmdb", "entity1", "entitya", "rest1", "imdb"}
SOURCE_B_KEYWORDS = {"tableb", "source2", "sourceb", "right", "buy", "scholar",
                     "google", "gp", "acm", "tvdb", "imdb", "entity2", "entityb",
                     "rest2"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FILE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _iter_files(root: Path):
    """Yield all files under root, skipping hidden and unwanted extensions."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            if p.suffix.lower() not in SKIP_EXTS:
                yield p


def _extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a zip archive and return the top-level directory."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    # If all files are under a single sub-dir, return that sub-dir
    contents = list(dest.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        return contents[0]
    return dest


# ══════════════════════════════════════════════════════════════════════════════
# 2. FORMAT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TABULAR_EXTS:
        return "tabular"
    if ext in PARQUET_EXTS:
        return "parquet"
    if ext in JSON_EXTS:
        return "json"
    if ext in RDF_TURTLE:
        return "turtle"
    if ext in RDF_NT:
        return "ntriples"
    if ext in RDF_XML:
        return "rdfxml"
    # Peek at content for ambiguous extensions
    try:
        snippet = path.read_bytes()[:512]
        if b"@prefix" in snippet or b"@base" in snippet:
            return "turtle"
        if b"<rdf:" in snippet or b"<owl:" in snippet or b"<?xml" in snippet:
            return "rdfxml"
        if snippet.lstrip().startswith(b"{") or snippet.lstrip().startswith(b"["):
            return "json"
    except Exception:
        pass
    return "tabular"


def _detect_separator(path: Path, encoding: str = "utf-8") -> str:
    """Sniff the CSV separator from the first few lines."""
    try:
        with open(path, encoding=encoding, errors="replace") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        return dialect.delimiter
    except csv.Error:
        # Fall back: count occurrences
        counts = {sep: sample.count(sep) for sep in [",", "|", "\t", ";"]}
        return max(counts, key=counts.get)


def _detect_encoding(path: Path) -> str:
    """Try encodings in order and return the first that works."""
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            path.read_text(encoding=enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_tabular(path: Path) -> pd.DataFrame:
    enc = _detect_encoding(path)
    sep = _detect_separator(path, enc)
    df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str,
                     on_bad_lines="skip", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).astype(str)


def _load_json(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, dtype=str)
    except Exception:
        return pd.read_json(path, lines=True, dtype=str)


def _load_rdf_turtle(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse Turtle/N3.  Returns (entities_df, sameAs_df).
    Tries rdflib first, falls back to manual line-by-line parsing.
    """
    try:
        import rdflib
        g = rdflib.Graph()
        g.parse(str(path), format="turtle")
        return _graph_to_dfs(g)
    except Exception:
        return _parse_ntriples_manual(path)


def _load_ntriples(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import rdflib
        g = rdflib.Graph()
        g.parse(str(path), format="nt")
        return _graph_to_dfs(g)
    except Exception:
        return _parse_ntriples_manual(path)


def _load_rdfxml(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Handles both OAEI Cell-alignment files and general RDF/XML."""
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return pd.DataFrame(), pd.DataFrame()

    # OAEI alignment format: <Cell><entity1 rdf:resource="..."/><entity2 .../>
    ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
          "align": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"}
    cells = root.findall(".//{http://knowledgeweb.semanticweb.org/heterogeneity/alignment#}Cell")
    if cells:
        rows = []
        for cell in cells:
            e1 = (cell.find("{http://knowledgeweb.semanticweb.org/heterogeneity/alignment#}entity1")
                  or cell.find("entity1"))
            e2 = (cell.find("{http://knowledgeweb.semanticweb.org/heterogeneity/alignment#}entity2")
                  or cell.find("entity2"))
            if e1 is not None and e2 is not None:
                u1 = e1.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")
                u2 = e2.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")
                if u1 and u2:
                    rows.append({"id_A": u1, "id_B": u2})
        alignment_df = pd.DataFrame(rows)
        return pd.DataFrame(), alignment_df   # (entities empty, alignment present)

    # Generic RDF/XML — extract subjects + their literals
    try:
        import rdflib
        g = rdflib.Graph()
        g.parse(str(path), format="xml")
        return _graph_to_dfs(g)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _graph_to_dfs(g) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract entity triples and owl:sameAs from an rdflib graph."""
    import rdflib
    OWL_SAME = rdflib.URIRef("http://www.w3.org/2002/07/owl#sameAs")

    same_rows = []
    entity_rows: dict[str, dict] = {}
    for s, p, o in g:
        s_str = str(s)
        if p == OWL_SAME:
            same_rows.append({"id_A": s_str, "id_B": str(o)})
        elif isinstance(o, rdflib.Literal):
            prop = str(p).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            entity_rows.setdefault(s_str, {"id": s_str})[prop] = str(o)

    entities_df  = pd.DataFrame(list(entity_rows.values())) if entity_rows else pd.DataFrame()
    sameAs_df    = pd.DataFrame(same_rows) if same_rows else pd.DataFrame()
    return entities_df, sameAs_df


def _parse_ntriples_manual(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback manual N-Triples parser — no rdflib required."""
    enc = _detect_encoding(path)
    same_rows = []
    entity_rows: dict[str, dict] = {}
    same_uri = "http://www.w3.org/2002/07/owl#sameAs"
    uri_re = re.compile(r'<([^>]+)>')
    lit_re = re.compile(r'"((?:[^"\\]|\\.)*)"')

    with open(path, encoding=enc, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            uris = uri_re.findall(line)
            if len(uris) >= 2 and same_uri in uris:
                same_rows.append({"id_A": uris[0], "id_B": uris[-1]})
            elif len(uris) >= 2:
                s, p = uris[0], uris[1]
                lit = lit_re.search(line)
                if lit:
                    prop = p.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                    entity_rows.setdefault(s, {"id": s})[prop] = lit.group(1)

    return (pd.DataFrame(list(entity_rows.values())),
            pd.DataFrame(same_rows) if same_rows else pd.DataFrame())


def load_file(path: Path) -> dict:
    """
    Load a file and return a dict with keys:
      format, df (for tabular) or (entities_df, alignment_df) for RDF
    """
    fmt = _detect_format(path)
    if fmt == "tabular":
        return {"format": fmt, "df": _load_tabular(path)}
    if fmt == "parquet":
        return {"format": fmt, "df": _load_parquet(path)}
    if fmt == "json":
        return {"format": fmt, "df": _load_json(path)}
    if fmt == "turtle":
        entities, alignment = _load_rdf_turtle(path)
        return {"format": fmt, "entities": entities, "alignment": alignment}
    if fmt == "ntriples":
        entities, alignment = _load_ntriples(path)
        return {"format": fmt, "entities": entities, "alignment": alignment}
    if fmt == "rdfxml":
        entities, alignment = _load_rdfxml(path)
        return {"format": fmt, "entities": entities, "alignment": alignment}
    return {"format": "unknown", "df": pd.DataFrame()}


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROLE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _stem_tokens(name: str) -> set[str]:
    """Normalise a filename into a set of lower-case tokens."""
    clean = re.sub(r"[^a-z0-9]", "", name.lower())
    return {clean}


def _score_role(name: str, keyword_set: set[str]) -> int:
    tokens = _stem_tokens(name)
    return sum(1 for kw in keyword_set if any(kw in t for t in tokens))


def _is_likely_ground_truth(df: pd.DataFrame) -> bool:
    """Heuristic: ground truth usually has 2–3 columns of IDs."""
    if df is None or df.empty:
        return False
    if not (2 <= len(df.columns) <= 3):
        return False
    # All columns should look like IDs (short strings / integers)
    for col in df.columns:
        sample = df[col].dropna().head(20)
        if sample.empty:
            continue
        # If any value is suspiciously long, probably not an ID
        if sample.astype(str).str.len().mean() > 60:
            return False
    return True


def classify_files(files: list[Path], loaded: dict) -> dict[str, Optional[Path]]:
    """
    Return {"source_a": Path, "source_b": Path, "ground_truth": Path}.
    Uses filename heuristics + structural analysis.
    """
    scores: dict[Path, dict[str, int]] = {}

    for path in files:
        stem = path.stem
        fmt  = _detect_format(path)
        info = loaded.get(path, {})

        # RDF alignment files → ground truth
        if fmt in ("turtle", "ntriples", "rdfxml"):
            align = info.get("alignment", pd.DataFrame())
            if align is not None and not align.empty:
                scores[path] = {"ground_truth": 10, "source_a": 0, "source_b": 0}
                continue
            else:
                scores[path] = {"ground_truth": 0, "source_a": 5, "source_b": 0}
                continue

        df = info.get("df", pd.DataFrame())

        gt_score = _score_role(stem, GT_KEYWORDS)
        sa_score = _score_role(stem, SOURCE_A_KEYWORDS)
        sb_score = _score_role(stem, SOURCE_B_KEYWORDS)

        # Structural bonus
        if _is_likely_ground_truth(df):
            gt_score += 3
        elif df is not None and len(df.columns) > 3:
            sa_score += 1
            sb_score += 1

        scores[path] = {"ground_truth": gt_score, "source_a": sa_score, "source_b": sb_score}

    # Pick best candidate for each role
    def pick(role: str, exclude: set[Path]) -> Optional[Path]:
        candidates = [(p, s[role]) for p, s in scores.items() if p not in exclude]
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[1])
        if best[1] == 0:
            # No clear winner — fall back to size heuristic
            remaining = [p for p, _ in candidates]
            sizes = {p: p.stat().st_size for p in remaining}
            if role == "ground_truth":
                return min(sizes, key=sizes.get)
            elif role == "source_a":
                sorted_p = sorted(remaining, key=lambda p: sizes[p])
                return sorted_p[0] if sorted_p else None
            else:
                sorted_p = sorted(remaining, key=lambda p: sizes[p], reverse=True)
                return sorted_p[0] if sorted_p else None
        return best[0]

    used: set[Path] = set()
    gt = pick("ground_truth", used)
    if gt:
        used.add(gt)
    sa = pick("source_a", used)
    if sa:
        used.add(sa)
    sb = pick("source_b", used)
    if sb:
        used.add(sb)

    return {"source_a": sa, "source_b": sb, "ground_truth": gt}


# ══════════════════════════════════════════════════════════════════════════════
# 5. NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the DataFrame has an 'id' column."""
    if "id" in df.columns:
        return df
    # Try common ID column names
    for candidate in ["ID", "Id", "entity_id", "uri", "subject", "D1", "D2"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "id"})
            return df
    # Use first column
    df = df.copy()
    df.insert(0, "id", df.iloc[:, 0])
    return df


def normalise_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalise a source DataFrame to pipeline format."""
    df = df.copy().fillna("").astype(str)
    df = _ensure_id_column(df)
    df["source"] = source_name
    # Remove duplicate 'source' columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def normalise_ground_truth(df: pd.DataFrame,
                            col1: str, col2: str,
                            s1_ids: set, s2_ids: set) -> pd.DataFrame:
    """
    Normalise a ground truth DataFrame so columns are (col1, col2).
    Handles:
      - standard 2-col CSV  (idA, idB)
      - pyJedAI pipe-sep     (D1|D2 with integer indices)
      - OAEI alignment       (id_A, id_B)
    Also fixes swapped columns if IDs are on the wrong side.
    """
    df = df.copy().fillna("").astype(str)
    # Rename first two columns to canonical names
    cols = list(df.columns)
    if len(cols) >= 2:
        df = df.rename(columns={cols[0]: col1, cols[1]: col2})

    # Check if IDs are on the right side; swap if needed
    if s1_ids and s2_ids:
        sample_c1 = set(df[col1].head(20))
        sample_c2 = set(df[col2].head(20))
        overlap_c1_s1 = len(sample_c1 & s1_ids)
        overlap_c1_s2 = len(sample_c1 & s2_ids)
        if overlap_c1_s2 > overlap_c1_s1 and overlap_c1_s2 > 0:
            df = df.rename(columns={col1: col2, col2: col1})

    return df[[col1, col2]].drop_duplicates()


def _extract_rdf_entities_as_source(loaded_info: dict, source_name: str) -> pd.DataFrame:
    """Convert rdflib entity dict into a pipeline-ready source DataFrame."""
    entities = loaded_info.get("entities", pd.DataFrame())
    if entities is None or entities.empty:
        return pd.DataFrame(columns=["id", "source"])
    df = entities.copy()
    if "id" not in df.columns and len(df.columns) > 0:
        df.insert(0, "id", range(len(df)))
    df["source"] = source_name
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _load_extra_registry() -> dict:
    if EXTRA_REGISTRY_PATH.exists():
        with open(EXTRA_REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_extra_registry(reg: dict) -> None:
    EXTRA_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXTRA_REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def register_dataset(name: str, raw_dir: str, source1: str, source2: str,
                     truth_col_s1: str, truth_col_s2: str,
                     data_type: str = "Auto-detected") -> None:
    reg = _load_extra_registry()
    reg[name] = {
        "raw_dir":       raw_dir,
        "source1":       source1,
        "source2":       source2,
        "data_type":     data_type,
        "truth_col_s1":  truth_col_s1,
        "truth_col_s2":  truth_col_s2,
    }
    _save_extra_registry(reg)
    print(f"  [REGISTER] '{name}' added to pipeline/extra_datasets.json")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN INTEGRATION FLOW
# ══════════════════════════════════════════════════════════════════════════════

def integrate(source: Path, name: str, source1_name: str, source2_name: str,
              dry_run: bool = False,
              source1_file: Optional[str] = None,
              source2_file: Optional[str] = None,
              gt_file: Optional[str] = None) -> bool:
    """
    Full integration pipeline.
    Returns True on success, False on failure.
    """
    print(f"\n{'='*60}")
    print(f"  INTEGRATE: {source.name}  →  dataset '{name}'")
    print(f"{'='*60}")

    # ── Step A: Extract zip if needed ─────────────────────────────────────────
    if source.suffix.lower() == ".zip":
        extract_dir = RAW_DIR / name
        print(f"\n[1/6] Extracting {source.name} → {extract_dir}")
        source = _extract_zip(source, extract_dir)
    else:
        print(f"\n[1/6] Source directory: {source}")

    # ── Step B: Discover files ────────────────────────────────────────────────
    files = list(_iter_files(source))
    print(f"\n[2/6] Files found: {len(files)}")
    for f in files:
        fmt = _detect_format(f)
        size_kb = f.stat().st_size // 1024
        print(f"       {f.name:<40} format={fmt:<10} {size_kb:>6} KB")

    if not files:
        print("  [ERROR] No usable files found.")
        return False

    # ── Step C: Load all files ────────────────────────────────────────────────
    print(f"\n[3/6] Loading files...")
    loaded: dict[Path, dict] = {}
    for f in files:
        try:
            info = load_file(f)
            df_shape = ""
            if "df" in info and info["df"] is not None:
                df_shape = f"{info['df'].shape}"
            elif "entities" in info:
                ent = info.get("entities", pd.DataFrame())
                aln = info.get("alignment", pd.DataFrame())
                df_shape = f"entities={len(ent)} alignment={len(aln)}"
            print(f"       {f.name:<40} {df_shape}")
            loaded[f] = info
        except Exception as e:
            print(f"       {f.name:<40} [ERROR] {e}")

    # ── Step D: Classify roles ────────────────────────────────────────────────
    print(f"\n[4/6] Detecting roles...")
    roles = classify_files(files, loaded)

    # Apply explicit overrides if provided
    def _find_file(name_hint: str) -> Optional[Path]:
        for p in files:
            if p.name == name_hint or p.stem == name_hint:
                return p
        return None

    if source1_file:
        p = _find_file(source1_file)
        if p:
            roles["source_a"] = p
        else:
            print(f"  [WARN] --source1-file '{source1_file}' not found in discovered files")
    if source2_file:
        p = _find_file(source2_file)
        if p:
            roles["source_b"] = p
        else:
            print(f"  [WARN] --source2-file '{source2_file}' not found in discovered files")
    if gt_file:
        p = _find_file(gt_file)
        if p:
            roles["ground_truth"] = p
        else:
            print(f"  [WARN] --gt-file '{gt_file}' not found in discovered files")

    for role, path in roles.items():
        print(f"       {role:<15} → {path.name if path else '(not found)'}")

    if not roles["source_a"] or not roles["source_b"]:
        print("  [ERROR] Could not identify both source tables.")
        return False

    # ── Step E: Normalise ─────────────────────────────────────────────────────
    print(f"\n[5/6] Normalising...")

    path_a = roles["source_a"]
    path_b = roles["source_b"]
    path_gt = roles["ground_truth"]
    info_a = loaded[path_a]
    info_b = loaded[path_b]

    # Build source DataFrames
    fmt_a = info_a["format"]
    if fmt_a in ("turtle", "ntriples", "rdfxml"):
        df_a = _extract_rdf_entities_as_source(info_a, source1_name)
    else:
        raw_a = info_a.get("df", pd.DataFrame())
        df_a = normalise_source(raw_a, source1_name)

    fmt_b = info_b["format"]
    if fmt_b in ("turtle", "ntriples", "rdfxml"):
        df_b = _extract_rdf_entities_as_source(info_b, source2_name)
    else:
        raw_b = info_b.get("df", pd.DataFrame())
        df_b = normalise_source(raw_b, source2_name)

    col1 = f"id{source1_name.capitalize()}"
    col2 = f"id{source2_name.capitalize()}"

    # Build ground truth DataFrame
    df_gt = pd.DataFrame(columns=[col1, col2])
    if path_gt:
        info_gt = loaded[path_gt]
        fmt_gt = info_gt["format"]

        if fmt_gt in ("turtle", "ntriples", "rdfxml"):
            align = info_gt.get("alignment", pd.DataFrame())
            if align is not None and not align.empty:
                df_gt = align.rename(columns={align.columns[0]: col1,
                                               align.columns[1]: col2})
        else:
            raw_gt = info_gt.get("df", pd.DataFrame())
            if not raw_gt.empty:
                s1_ids = set(df_a["id"].astype(str).unique())
                s2_ids = set(df_b["id"].astype(str).unique())
                df_gt = normalise_ground_truth(raw_gt, col1, col2, s1_ids, s2_ids)

    print(f"       source1 ({source1_name}): {len(df_a):,} rows, "
          f"{len(df_a.columns)} cols — {list(df_a.columns[:5])}")
    print(f"       source2 ({source2_name}): {len(df_b):,} rows, "
          f"{len(df_b.columns)} cols — {list(df_b.columns[:5])}")
    print(f"       ground truth:    {len(df_gt):,} pairs  [{col1}, {col2}]")

    if dry_run:
        print("\n  [DRY-RUN] No files written.")
        return True

    # ── Step F: Save + register ───────────────────────────────────────────────
    print(f"\n[6/6] Saving & registering...")

    out_dir = CLEAN_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_path = out_dir / "cleaned_source1.csv"
    s2_path = out_dir / "cleaned_source2.csv"
    gt_path = out_dir / "ground_truth.csv"

    df_a.to_csv(s1_path, index=False)
    df_b.to_csv(s2_path, index=False)
    if not df_gt.empty:
        df_gt.to_csv(gt_path, index=False)

    print(f"       [SAVE] {s1_path}")
    print(f"       [SAVE] {s2_path}")
    if not df_gt.empty:
        print(f"       [SAVE] {gt_path}")
    else:
        print(f"       [WARN] No ground truth saved — evaluation will be skipped")

    raw_dir_str = str(source)
    register_dataset(
        name=name,
        raw_dir=raw_dir_str,
        source1=source1_name,
        source2=source2_name,
        truth_col_s1=col1,
        truth_col_s2=col2,
        data_type=f"Auto-detected ({', '.join(set(_detect_format(f) for f in files))})",
    )

    print(f"\n  ✓ Dataset '{name}' is ready.")
    print(f"    Run:  ./venv/bin/python cli/run_pipeline.py --dataset {name}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generic Entity Resolution Dataset Integrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", required=True,
                        help="Path to a directory or .zip file containing the dataset")
    parser.add_argument("--name", required=True,
                        help="Dataset name to register (e.g., 'dbpedia_imdb')")
    parser.add_argument("--source1-name", default=None,
                        help="Name for source table 1 (auto-inferred from filenames if omitted)")
    parser.add_argument("--source2-name", default=None,
                        help="Name for source table 2 (auto-inferred from filenames if omitted)")
    parser.add_argument("--source1-file", default=None,
                        help="Explicit filename for source table 1 (overrides role auto-detection)")
    parser.add_argument("--source2-file", default=None,
                        help="Explicit filename for source table 2 (overrides role auto-detection)")
    parser.add_argument("--gt-file", default=None,
                        help="Explicit filename for ground truth (overrides role auto-detection)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse only — do not write any files")
    parser.add_argument("--run", action="store_true",
                        help="Run the full pipeline after integration")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"[ERROR] Path not found: {source}")
        sys.exit(1)

    # Auto-infer source names from directory name if not provided
    s1_name = args.source1_name
    s2_name = args.source2_name
    if not s1_name or not s2_name:
        # Scan filenames for common names
        if source.is_dir():
            stems = [f.stem.lower() for f in _iter_files(source)
                     if _detect_format(f) in ("tabular", "parquet", "json")
                     and not any(kw in f.stem.lower() for kw in ["gt", "gold", "match", "map"])]
            if not s1_name:
                s1_name = stems[0] if stems else "source1"
                s1_name = re.sub(r"clean$", "", s1_name).strip("_-") or "source1"
            if not s2_name and len(stems) > 1:
                s2_name = stems[1]
                s2_name = re.sub(r"clean$", "", s2_name).strip("_-") or "source2"
            elif not s2_name:
                s2_name = "source2"

    ok = integrate(source, args.name, s1_name, s2_name, dry_run=args.dry_run,
                   source1_file=args.source1_file,
                   source2_file=args.source2_file,
                   gt_file=args.gt_file)

    if ok and args.run and not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  RUNNING PIPELINE for '{args.name}'")
        print(f"{'='*60}")
        import subprocess
        cmd = [
            sys.executable, str(REPO_ROOT / "cli" / "run_pipeline.py"),
            "--dataset", args.name,
            "--from-step", "1",
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
