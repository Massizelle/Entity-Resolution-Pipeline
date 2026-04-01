# -*- coding: utf-8 -*-
"""
Génère un rapport PDF (français) sur le travail du membre 2 (traitement des blocs).

  python scripts/generate_member2_report_pdf.py
  python scripts/generate_member2_report_pdf.py -o chemin/sortie.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("Installez fpdf2 : pip install fpdf2", file=sys.stderr)
    raise


class ReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _width(pdf: FPDF) -> float:
    return pdf.w - pdf.l_margin - pdf.r_margin


def add_heading(pdf: FPDF, text: str, size: int = 14) -> None:
    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", size)
    pdf.multi_cell(_width(pdf), 8, text)
    pdf.set_font("Helvetica", "", 11)


def add_para(pdf: FPDF, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(_width(pdf), 6, text)
    pdf.ln(2)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_out = (
        root.parent / "presentation" / "Member2_Rapport_Membre2_FR.pdf"
    )

    parser = argparse.ArgumentParser(
        description="Génère le rapport PDF (membre 2) en français"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_out,
        help="Chemin du PDF de sortie",
    )
    args = parser.parse_args()
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(_width(pdf), 10, "Membre 2 : traitement des blocs (block processing)")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        _width(pdf),
        7,
        "Chaîne de résolution d'entités (cours Big Data)",
    )
    pdf.ln(6)

    add_para(
        pdf,
        "Ce document résume le travail réalisé pour le membre 2 dans le projet de "
        "résolution d'entités de bout en bout. Le membre 2 se situe entre le blocage "
        "(membre 1) et l'appariement d'entités (membre 3) : nous réduisons l'ensemble "
        "des paires d'enregistrements que les étapes aval doivent comparer, tout en "
        "conservant autant de vrais appariements que le permettent les seuils choisis.",
    )

    add_heading(pdf, "1. Responsabilités (selon le plan de projet)", 13)
    add_para(
        pdf,
        "- Purge des blocs : supprimer entièrement les blocs qui dépassent un seuil "
        "de taille maximal. Les blocs très grands sont coûteux et apportent souvent "
        "peu d'information utile pour l'appariement.",
    )
    add_para(
        pdf,
        "- Méta-blocage (meta-blocking) : construire un graphe dont les noeuds sont "
        "les entités. Deux entités issues de sources différentes sont reliées par une "
        "arête si elles apparaissent ensemble dans au moins un bloc après la purge. Chaque arête "
        "est pondérée par la similarité de Jaccard entre les ensembles d'identifiants "
        "de blocs des deux entités. Les paires dont le poids est inférieur à un seuil "
        "minimal sont éliminées.",
    )
    add_para(
        pdf,
        "- Sortie : un fichier de paires candidates par jeu de données, au format CSV "
        "convenu, destiné au membre 3.",
    )

    add_heading(pdf, "2. Contrat de données", 13)
    add_para(
        pdf,
        "Entrée : blocks.csv (produit par le membre 1). Colonnes obligatoires : "
        "block_id, entity_id, source.",
    )
    add_para(
        pdf,
        "Sortie : candidate_pairs.csv. Colonnes : id_A, id_B. id_A désigne toujours "
        "une entité de la source 1 et id_B une entité de la source 2, selon les noms "
        "de source définis dans DATASET_REGISTRY (par ex. abt/buy, amazon/google, "
        "spimbench_a/spimbench_b).",
    )
    add_para(
        pdf,
        "Fichier de métriques optionnel : member2_stats.json (avec l'option "
        "--write-stats), contenant les compteurs de purge, la taille du produit "
        "cartésien n1*n2, le nombre final de paires candidates, et "
        "reduction_vs_cartesian = 1 - (candidats / (n1*n2)).",
    )

    add_heading(pdf, "3. Algorithmes", 13)
    add_para(
        pdf,
        "Purge : regrouper les lignes par block_id. Supprimer toutes les lignes "
        "appartenant à un bloc comptant strictement plus de max_block_size lignes. "
        "Valeur par défaut de max_block_size : 1000 (ligne de commande : "
        "--max-block-size).",
    )
    add_para(
        pdf,
        "Méta-blocage : pour chaque entité, collecter l'ensemble des block_id où elle "
        "apparaît après purge. Énumérer chaque paire inter-sources (source1 x source2) "
        "présente ensemble dans au moins un bloc. Pour chaque paire (a, b), calculer "
        "Jaccard = |intersection de B(a) et B(b)| / |réunion de B(a) et B(b)|. "
        "Conserver la paire si "
        "Jaccard >= min_jaccard (CLI : --min-jaccard, défaut 0.0). Un graphe NetworkX "
        "est construit sur les arêtes conservées pour la structure et un usage ultérieur "
        "éventuel.",
    )

    add_heading(pdf, "4. Fichiers implémentés", 13)
    add_para(
        pdf,
        "- block_processing.py : chargement/validation CSV, purge_oversized_blocks, "
        "meta_blocking_candidate_pairs, run_block_processing (orchestration E/S et "
        "statistiques).",
    )
    add_para(
        pdf,
        "- run_member2.py : interface en ligne de commande pour tous les jeux de données "
        "ou un seul ; --mock lit output/mock/<dataset>/blocks.csv ; --skip-missing "
        "ignore les fichiers blocks manquants ; --write-stats écrit member2_stats.json.",
    )
    add_para(
        pdf,
        "- tests/test_block_processing.py : tests pytest (colonnes, purge, filtrage "
        "Jaccard, bout en bout CSV + JSON). Exécution : pip install -r "
        "requirements-dev.txt puis python -m pytest tests/test_block_processing.py -v",
    )

    add_heading(pdf, "5. Utilisation", 13)
    add_para(
        pdf,
        "Depuis le répertoire Entity-Resolution-Pipeline, après "
        "pip install -r requirements.txt :",
    )
    add_para(
        pdf,
        "  python run_member2.py                    # tous les jeux, vrais blocks.csv",
    )
    add_para(
        pdf,
        "  python run_member2.py --dataset amazon_google",
    )
    add_para(
        pdf,
        "  python run_member2.py --mock             # petits blocs fictifs (output/mock/)",
    )
    add_para(
        pdf,
        "  python run_member2.py --write-stats      # produit aussi member2_stats.json",
    )
    add_para(
        pdf,
        "  python run_member2.py --min-jaccard 0.15 # méta-blocage plus strict",
    )

    add_heading(pdf, "6. Notes d'intégration", 13)
    add_para(
        pdf,
        "Le membre 1 peut déjà limiter la taille des blocs lors du blocage par tokens ; "
        "le membre 2 applique néanmoins une purge sur le blocks.csv livré afin que le "
        "rapport attribue clairement le traitement des blocs à cette étape. Régler "
        "min_jaccard arbitre entre le rappel des paires candidates et le coût pour le "
        "membre 3 ; il faut documenter les valeurs retenues dans le rapport final.",
    )
    add_para(
        pdf,
        "Un passage sur les mocks écrase output/mock/.../candidate_pairs.csv. Pour "
        "régénérer les mocks de référence : python -c \"from create_mocks import "
        "generate_all_mocks; generate_all_mocks()\"",
    )

    pdf.output(str(out_path))
    print(f"OK -> {out_path}")


if __name__ == "__main__":
    main()
