"""
Member 5 — Baseline pyJedAI
============================
Exécute le pipeline pyJedAI complet sur n'importe quel dataset du projet
et sauvegarde les résultats dans output/<dataset>/.

Usage:
    python cli/execution_pyjedai.py --dataset abt_buy
    python cli/execution_pyjedai.py --dataset amazon_google
    python cli/execution_pyjedai.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Configuration de chaque dataset ──────────────────────────────────────────

DATASET_CONFIGS: dict[str, dict] = {
    "abt_buy": {
        "file1":      "data/raw/Abt-Buy/Abt.csv",
        "file2":      "data/raw/Abt-Buy/Buy.csv",
        "sep":        ",",
        "enc1":       "latin-1",
        "enc2":       "latin-1",
        "id1":        "id",
        "id2":        "id",
        "attrs1":     ["name"],
        "attrs2":     ["name"],
        "gt_file":    "data/raw/Abt-Buy/abt_buy_perfectMapping.csv",
        "gt_sep":     ",",
        "gt_enc":     "latin-1",
        "gt_col1":    "idAbt",
        "gt_col2":    "idBuy",
        "threshold":  0.17,
    },
    "amazon_google": {
        "file1":      "data/raw/Amazon-GoogleProducts/Amazon.csv",
        "file2":      "data/raw/Amazon-GoogleProducts/GoogleProducts.csv",
        "sep":        ",",
        "enc1":       "latin-1",
        "enc2":       "latin-1",
        "id1":        "id",
        "id2":        "id",
        "attrs1":     ["title"],
        "attrs2":     ["name"],
        "gt_file":    "data/raw/Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv",
        "gt_sep":     ",",
        "gt_enc":     "latin-1",
        "gt_col1":    "idAmazon",
        "gt_col2":    "idGoogleBase",
        "threshold":  0.20,
    },
    "dblp_acm": {
        "file1":      "data/raw/DBLP-ACM/tableA.csv",
        "file2":      "data/raw/DBLP-ACM/tableB.csv",
        "sep":        ",",
        "enc1":       "utf-8",
        "enc2":       "utf-8",
        "id1":        "id",
        "id2":        "id",
        "attrs1":     ["title"],
        "attrs2":     ["title"],
        "gt_file":    None,   # construit depuis train/valid/test
        "gt_sep":     ",",
        "gt_enc":     "utf-8",
        "gt_col1":    "ltable_id",
        "gt_col2":    "rtable_id",
        "threshold":  0.30,
    },
    "dblp_scholar": {
        "file1":      "data/raw/DBLP-Scholar/DBLP1.csv",
        "file2":      "data/raw/DBLP-Scholar/Scholar.csv",
        "sep":        ",",
        "enc1":       "latin-1",
        "enc2":       "utf-8-sig",
        "id1":        "id",
        "id2":        "id",
        "attrs1":     ["title"],
        "attrs2":     ["title"],
        "gt_file":    "data/raw/DBLP-Scholar/DBLP-Scholar_perfectMapping.csv",
        "gt_sep":     ",",
        "gt_enc":     "utf-8",
        "gt_col1":    "idDBLP",
        "gt_col2":    "idScholar",
        "threshold":  0.25,
    },
    "dbpedia_imdb": {
        "file1":      "data/raw/DBpedia-IMDb/dbpediaclean.csv",
        "file2":      "data/raw/DBpedia-IMDb/imdbclean.csv",
        "sep":        "|",
        "enc1":       "utf-8",
        "enc2":       "utf-8",
        "id1":        "id",
        "id2":        "id",
        "attrs1":     ["title"],
        "attrs2":     ["title"],
        "gt_file":    "data/raw/DBpedia-IMDb/gtclean.csv",
        "gt_sep":     "|",
        "gt_enc":     "utf-8",
        "gt_col1":    "D1",
        "gt_col2":    "D2",
        "threshold":  0.20,
    },
    "walmart_amazon": {
        "file1":      "data/raw/Walmart-Amazon-pyjedai/walmartclean.csv",
        "file2":      "data/raw/Walmart-Amazon-pyjedai/amazonclean.csv",
        "sep":        "|",
        "enc1":       "utf-8",
        "enc2":       "utf-8",
        "id1":        "id",
        "id2":        "id",
        "attrs1":     ["title"],
        "attrs2":     ["title"],
        "gt_file":    "data/raw/Walmart-Amazon-pyjedai/gtclean.csv",
        "gt_sep":     "|",
        "gt_enc":     "utf-8",
        "gt_col1":    "D1",
        "gt_col2":    "D2",
        "threshold":  0.20,
    },
}


# ── Chargement des données ────────────────────────────────────────────────────

def _load_dblp_acm_ground_truth() -> pd.DataFrame:
    """Construit le ground truth DBLP-ACM depuis train/valid/test."""
    frames = []
    for split in ["train.csv", "valid.csv", "test.csv"]:
        path = os.path.join("data/raw/DBLP-ACM", split)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, encoding="utf-8")
        if {"ltable_id", "rtable_id", "label"} <= set(df.columns):
            frames.append(df[df["label"] == 1][["ltable_id", "rtable_id"]])
    if not frames:
        return pd.DataFrame(columns=["ltable_id", "rtable_id"])
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = DATASET_CONFIGS[name]

    df1 = pd.read_csv(cfg["file1"], sep=cfg["sep"], encoding=cfg["enc1"],
                      engine="python", na_filter=False)
    df2 = pd.read_csv(cfg["file2"], sep=cfg["sep"], encoding=cfg["enc2"],
                      engine="python", na_filter=False)

    if name == "dblp_acm":
        gt = _load_dblp_acm_ground_truth()
    elif cfg["gt_file"] and os.path.exists(cfg["gt_file"]):
        gt = pd.read_csv(cfg["gt_file"], sep=cfg["gt_sep"], encoding=cfg["gt_enc"],
                         engine="python", na_filter=False)
    else:
        gt = pd.DataFrame()

    print(f"  source1 : {len(df1)} entités")
    print(f"  source2 : {len(df2)} entités")
    print(f"  GT      : {len(gt)} paires")
    return df1, df2, gt


# ── Affichage et sauvegarde des métriques ────────────────────────────────────

def _print_metrics(name, precision, recall, f1, reduction_ratio,
                   n_evaluated, total_possible, elapsed):
    mins = int(elapsed // 60)
    secs = elapsed % 60
    time_str = f"{mins}m {secs:.1f}s" if mins > 0 else f"{secs:.1f}s"

    print(f"\n{'='*55}")
    print(f"  MÉTRIQUES FINALES — {name.upper()}")
    print(f"{'='*55}")
    if precision is not None:
        print(f"  {'Précision':<30} {precision:.4f}")
        print(f"  {'Rappel':<30} {recall:.4f}")
        print(f"  {'F1-score':<30} {f1:.4f}")
    else:
        print(f"  Pas de ground truth — métriques de qualité indisponibles")
    print(f"  {'Ratio de Réduction':<30} {reduction_ratio:.4f}")
    print(f"  {'Paires évaluées':<30} {n_evaluated:,}")
    print(f"  {'Paires possibles':<30} {total_possible:,}")
    print(f"  {'Temps d\'exécution':<30} {time_str}")
    print(f"{'='*55}\n")


def _save_metrics(name, out_dir, precision, recall, f1, reduction_ratio,
                  n_evaluated, total_possible, elapsed):
    data = {
        "dataset":          name,
        "precision":        round(precision, 4) if precision is not None else None,
        "recall":           round(recall, 4)    if recall    is not None else None,
        "f1":               round(f1, 4)        if f1        is not None else None,
        "reduction_ratio":  round(reduction_ratio, 4),
        "n_evaluated":      n_evaluated,
        "total_possible":   total_possible,
        "execution_time_s": round(elapsed, 2),
    }
    metrics_path = os.path.join(out_dir, "pyjedai_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Métriques sauvegardées : {metrics_path}")


# ── Pipeline pyJedAI ──────────────────────────────────────────────────────────

def run_pyjedai(name: str) -> None:
    cfg = DATASET_CONFIGS[name]
    out_dir = os.path.join("output", name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  pyJedAI — {name.upper()}")
    print(f"{'='*60}")

    # ── Import pyJedAI ────────────────────────────────────────────────────────
    try:
        import pyjedai
        from pyjedai.datamodel import Data
        from pyjedai.block_building import StandardBlocking
        from pyjedai.block_cleaning import BlockPurging, BlockFiltering
        from pyjedai.comparison_cleaning import WeightedEdgePruning
        from pyjedai.matching import EntityMatching
        from pyjedai.clustering import UniqueMappingClustering
    except ImportError:
        print("[ERREUR] pyjedai n'est pas installé. Lance : pip install pyjedai")
        sys.exit(1)

    t_start = time.time()

    # ── Chargement ────────────────────────────────────────────────────────────
    print("\n[1/6] Chargement des données...")
    df1, df2, gt = load_dataset(name)
    n1, n2 = len(df1), len(df2)
    total_pairs_possible = n1 * n2

    has_gt = not gt.empty
    data = Data(
        dataset_1=df1,
        id_column_name_1=cfg["id1"],
        dataset_2=df2,
        id_column_name_2=cfg["id2"],
        ground_truth=gt if has_gt else None,
    )
    data.print_specs()
    data.clean_dataset(remove_stopwords=True, remove_punctuation=True,
                       remove_numbers=False, remove_unicodes=False)

    # ── Block Building ────────────────────────────────────────────────────────
    print("\n[2/6] Block Building (StandardBlocking)...")
    bb = StandardBlocking()
    blocks = bb.build_blocks(data,
                             attributes_1=cfg["attrs1"],
                             attributes_2=cfg["attrs2"])
    bb.report()
    if has_gt:
        bb.evaluate(blocks)

    # ── Block Purging ─────────────────────────────────────────────────────────
    print("\n[3/6] Block Purging...")
    bp = BlockPurging()
    cleaned = bp.process(blocks, data, tqdm_disable=False)
    bp.report()
    if has_gt:
        bp.evaluate(cleaned)

    # ── Block Filtering ───────────────────────────────────────────────────────
    print("\n[4/6] Block Filtering...")
    bf = BlockFiltering()
    filtered = bf.process(cleaned, data, tqdm_disable=False)
    if has_gt:
        bf.evaluate(filtered)

    # ── Comparison Cleaning ───────────────────────────────────────────────────
    print("\n[5/6] Comparison Cleaning (WeightedEdgePruning)...")
    wep = WeightedEdgePruning(weighting_scheme="EJS")
    candidate_pairs = wep.process(filtered, data, tqdm_disable=False)

    df_candidates = wep.export_to_df(candidate_pairs)
    candidates_path = os.path.join(out_dir, "pyjedai_candidate_pairs.csv")
    df_candidates.to_csv(candidates_path, index=False)
    n_evaluated = len(df_candidates)
    print(f"  Sauvegardé : {candidates_path}  ({n_evaluated} paires)")

    # ── Entity Matching ───────────────────────────────────────────────────────
    print("\n[6/6] Entity Matching (TF-IDF cosine)...")
    em = EntityMatching(
        metric="cosine",
        tokenizer="char_tokenizer",
        vectorizer="tfidf",
        qgram=3,
        similarity_threshold=0.0,
    )
    pairs_graph = em.predict(candidate_pairs, data, tqdm_disable=False)
    if has_gt:
        em.evaluate(pairs_graph)

    df_matches = em.export_to_df(pairs_graph)
    match_path = os.path.join(out_dir, "pyjedai_match_results.csv")
    df_matches.to_csv(match_path, index=False)
    print(f"  Sauvegardé : {match_path}  ({len(df_matches)} paires)")

    # ── Entity Clustering ─────────────────────────────────────────────────────
    threshold = cfg["threshold"]
    print(f"\n[CLUSTER] UniqueMappingClustering (threshold={threshold})...")
    ccc = UniqueMappingClustering()
    clusters = ccc.process(pairs_graph, data, similarity_threshold=threshold)
    ccc.report()

    df_clusters = ccc.export_to_df(clusters)
    clusters_path = os.path.join(out_dir, "pyjedai_clusters.csv")
    df_clusters.to_csv(clusters_path, index=False)
    print(f"  Sauvegardé : {clusters_path}  ({len(df_clusters)} entrées)")

    # ── Métriques finales ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    reduction_ratio = 1.0 - (n_evaluated / total_pairs_possible) if total_pairs_possible > 0 else 0.0

    precision, recall, f1 = None, None, None
    if has_gt:
        try:
            result = ccc.evaluate(clusters)
        except Exception:
            result = None
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            precision, recall, f1 = float(result[0]), float(result[1]), float(result[2])
        elif isinstance(result, dict):
            precision = result.get("precision") or result.get("Precision")
            recall    = result.get("recall")    or result.get("Recall")
            f1        = result.get("f1")        or result.get("F1")

    _print_metrics(name, precision, recall, f1, reduction_ratio, n_evaluated,
                   total_pairs_possible, elapsed)
    _save_metrics(name, out_dir, precision, recall, f1, reduction_ratio,
                  n_evaluated, total_pairs_possible, elapsed)

    print(f"\n[DONE] {name} — résultats dans {out_dir}/")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Member 5 — Baseline pyJedAI sur tous les datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        default="abt_buy",
        help="Dataset à traiter (défaut: abt_buy)",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    if args.dataset == "all":
        for name in DATASET_CONFIGS:
            run_pyjedai(name)
    else:
        run_pyjedai(args.dataset)


if __name__ == "__main__":
    main()
