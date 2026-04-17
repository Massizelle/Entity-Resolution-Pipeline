"""
Interactive launcher for the ER pipeline.

The goal is pragmatic:
- ask a small set of explicit questions
- build the right command
- execute it immediately

No external dependencies, no magic configuration layer.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CLI_DIR = os.path.dirname(__file__)


# ── Dataset list (loaded dynamically from DATASET_REGISTRY) ──────────────────

_EXCLUDED_DATASETS = {"rexa_dblp"}


def _load_datasets() -> list[dict]:
    """Return dataset list from DATASET_REGISTRY (includes extra_datasets.json)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from pipeline.data_ingestion import DATASET_REGISTRY
        entries = []
        for name, cfg in DATASET_REGISTRY.items():
            if name in _EXCLUDED_DATASETS:
                continue
            entries.append({
                "key":    name,
                "type":   cfg.get("data_type", "?"),
                "s1":     cfg.get("source1", "?"),
                "s2":     cfg.get("source2", "?"),
                "desc":   cfg.get("description", ""),
                "extra":  cfg.get("pre_cleaned", False),
            })
        return entries
    except Exception:
        # Fallback si import échoue
        return [{"key": k, "type": "", "s1": "", "s2": "", "desc": "", "extra": False}
                for k in ["abt_buy", "amazon_google", "dblp_acm", "spimbench"]]


MEMBERS = {
    "pipeline": os.path.join(CLI_DIR, "run_pipeline.py"),
    "member1":  os.path.join(CLI_DIR, "run_member1.py"),
    "member2":  os.path.join(CLI_DIR, "run_member2.py"),
    "member3":  os.path.join(CLI_DIR, "run_member3.py"),
    "member4":  os.path.join(CLI_DIR, "run_member4.py"),
}

CLUSTERING_ALGOS     = ["connected_components"]

# Datasets connus pour être lourds (avertissement affiché)
HEAVY_DATASETS = {"dblp_scholar", "dbpedia_imdb"}

# ── Couleurs ANSI ─────────────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    """Wrap text in ANSI colour code (only if stdout is a tty)."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

BOLD  = lambda t: _c(t, "1")
DIM   = lambda t: _c(t, "2")
GREEN = lambda t: _c(t, "32")
CYAN  = lambda t: _c(t, "36")
YELLOW = lambda t: _c(t, "33")
RED   = lambda t: _c(t, "31")


# ── Prompt helpers ────────────────────────────────────────────────────────────

class PromptAborted(Exception):
    """Raised when the interactive prompt cannot continue cleanly."""


def _read_input(prompt: str) -> str:
    """Read one line from stdin and surface friendly interruption errors."""
    try:
        return input(prompt)
    except EOFError as exc:
        raise PromptAborted(
            "Entrée interactive indisponible ou incomplète. "
            "Relance le launcher dans un terminal interactif, ou exécute directement "
            "un script `cli/run_*.py` avec ses options."
        ) from exc
    except KeyboardInterrupt as exc:
        raise PromptAborted("Interaction annulée par l'utilisateur.") from exc


def _is_quit(raw: str) -> bool:
    return raw.lower() in {"q", "quit", "exit"}


def _print_cancel_hint() -> None:
    print(DIM("Tape 'q' pour annuler à tout moment."))


def _print_dataset_table(default: str = "abt_buy") -> None:
    datasets = _load_datasets()
    print(f"\n{BOLD('Dataset disponibles :')}")
    print(f"  {'#':<3} {'Nom':<22} {'Type':<26} {'Sources'}")
    print(f"  {'-'*3} {'-'*22} {'-'*26} {'-'*25}")
    for idx, d in enumerate(datasets, start=1):
        tag   = GREEN(" [extra]") if d["extra"] else ""
        warn  = YELLOW(" ⚠ lourd") if d["key"] in HEAVY_DATASETS else ""
        srcs  = f"{d['s1']} × {d['s2']}" if d["s1"] else ""
        dtype = d["type"][:25] if d["type"] else ""
        dflt  = DIM(" (défaut)") if d["key"] == default else ""
        print(f"  {idx:<3} {d['key']:<22} {dtype:<26} {srcs}{tag}{warn}{dflt}")
    print(f"  {len(datasets)+1:<3} {'all':<22} {'(tous les datasets)'}")

def _prompt_dataset(default: str = "abt_buy") -> str:
    """Show dataset menu with type/source info and return chosen key."""
    datasets = _load_datasets()
    keys = [d["key"] for d in datasets] + ["all"]

    _print_dataset_table(default=default)

    while True:
        raw = _read_input(f"\nChoix [{default}]: ").strip()
        if not raw:
            return default
        if _is_quit(raw):
            raise PromptAborted("Interaction annulée.")
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(datasets):
                return datasets[idx]["key"]
            if idx == len(datasets):
                return "all"
        if raw in keys:
            return raw
        print("  Choix invalide. Entre un numéro de la liste ou un nom de dataset.")


def _prompt_choice(prompt: str, options: list[str], default: str | None = None,
                   descriptions: dict[str, str] | None = None) -> str:
    print(f"\n{BOLD(prompt)}")
    for idx, option in enumerate(options, start=1):
        marker = DIM("  (défaut)") if option == default else ""
        desc   = DIM(f"  — {descriptions[option]}") if descriptions and option in descriptions else ""
        print(f"  {idx}. {option}{marker}{desc}")
    while True:
        raw = _read_input("> ").strip()
        if not raw and default is not None:
            return default
        if _is_quit(raw):
            raise PromptAborted("Interaction annulée.")
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(options):
                return options[index]
        if raw in options:
            return raw
        print(f"  Choix invalide. Options: {', '.join(options)}.")


def _prompt_int(prompt: str, default: int | None = None) -> int | None:
    suffix = DIM(f" [{default}]") if default is not None else ""
    while True:
        raw = _read_input(f"  {prompt}{suffix}: ").strip()
        if not raw:
            return default
        if _is_quit(raw):
            raise PromptAborted("Interaction annulée.")
        try:
            return int(raw)
        except ValueError:
            print("  Valeur invalide. Entier attendu.")


def _prompt_float(prompt: str, default: float | None = None) -> float | None:
    suffix = DIM(f" [{default}]") if default is not None else ""
    while True:
        raw = _read_input(f"  {prompt}{suffix}: ").strip()
        if not raw:
            return default
        if _is_quit(raw):
            raise PromptAborted("Interaction annulée.")
        try:
            return float(raw)
        except ValueError:
            print("  Valeur invalide. Nombre attendu.")


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = DIM(" [O/n]") if default else DIM(" [o/N]")
    while True:
        raw = _read_input(f"  {prompt}{suffix}: ").strip().lower()
        if not raw:
            return default
        if _is_quit(raw):
            raise PromptAborted("Interaction annulée.")
        if raw in {"y", "yes", "o", "oui"}:
            return True
        if raw in {"n", "no", "non"}:
            return False
        print("  Réponse invalide. Entre o/oui ou n/non.")


def _python_executable() -> str:
    venv_python = os.path.join(".", "venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


# ── Commande pipeline complète ────────────────────────────────────────────────

def _build_pipeline_command() -> list[str]:
    command = [_python_executable(), MEMBERS["pipeline"]]

    dataset = _prompt_dataset(default="abt_buy")
    if dataset != "all":
        command.extend(["--dataset", dataset])
        if dataset in HEAVY_DATASETS:
            print(YELLOW(f"\n  ⚠  '{dataset}' est un dataset lourd — les étapes 2 et 3 peuvent prendre plusieurs heures."))

    from_step = _prompt_int("Étape de début", default=1)
    to_step   = _prompt_int("Étape de fin",   default=4)
    command.extend(["--from-step", str(from_step), "--to-step", str(to_step)])

    print(f"\n{BOLD('Stratégie de candidats :')} {GREEN('cw_semantic_predictive')}")
    print(DIM("  witness-first + rescues sémantiques (stratégie finale unique)"))

    if from_step <= 3 <= to_step:
        print(f"\n{BOLD('Options Step 3 (matching) :')}")
        time_limit = _prompt_float("Budget temps en minutes (vide = illimité)", None)
        if time_limit is not None:
            command.extend(["--time-limit-minutes", str(time_limit)])

        chunk_size = _prompt_int("Taille de chunk", default=250)
        if chunk_size and chunk_size != 250:
            command.extend(["--chunk-size", str(chunk_size)])

        online = _prompt_yes_no("Clustering incrémental pendant le matching", default=False)
        if online:
            command.append("--online-clustering")
            
            every_n = _prompt_int("Rafraîchir les clusters tous les N chunks", default=5)
            if every_n:
                command.extend(["--online-cluster-every-n-chunks", str(every_n)])

        if _prompt_yes_no("Redémarrer de zéro (ignore le cache existant)", default=False):
            command.append("--no-resume")

    if from_step <= 4 <= to_step:
        print(f"\n{BOLD('Options Step 4 (clustering) :')}")
        print(f"  Algorithme utilisé : {GREEN('connected_components')}")

    print(f"\n{BOLD('Options avancées :')}")
    limit = _prompt_int("Limiter le nombre de paires Step 3 (vide = aucune limite)", None)
    if limit is not None:
        command.extend(["--limit", str(limit)])

    if _prompt_yes_no("Mode mock (fixtures pré-générées, pas de données brutes)", default=False):
        command.append("--mock")

    if _prompt_yes_no("Ignorer les fichiers manquants", default=False):
        command.append("--skip-missing")

    return command


# ── Commande membre individuel ────────────────────────────────────────────────

def _build_member_command(member_key: str) -> list[str]:
    command = [_python_executable(), MEMBERS[member_key]]

    dataset = _prompt_dataset(default="abt_buy")
    if dataset != "all":
        command.extend(["--dataset", dataset])
        if dataset in HEAVY_DATASETS:
            print(YELLOW(f"\n  ⚠  '{dataset}' est un dataset lourd."))

    if member_key in ("member2", "member3"):
        print(f"\n{BOLD('Stratégie de candidats :')} {GREEN('cw_semantic_predictive')}")
        print(DIM("  witness-first + rescues sémantiques (stratégie finale unique)"))

    if member_key == "member3":
        print(f"\n{BOLD('Options matching :')}")
        time_limit = _prompt_float("Budget temps en minutes (vide = illimité)", None)
        if time_limit is not None:
            command.extend(["--time-limit-minutes", str(time_limit)])

        chunk_size = _prompt_int("Taille de chunk", default=250)
        if chunk_size and chunk_size != 250:
            command.extend(["--chunk-size", str(chunk_size)])

        if _prompt_yes_no("Clustering incrémental", default=False):
            command.append("--online-clustering")
            command.extend(["--clustering-algorithm", "connected_components"])
            every_n = _prompt_int("Rafraîchir tous les N chunks", default=5)
            if every_n:
                command.extend(["--online-cluster-every-n-chunks", str(every_n)])

        if _prompt_yes_no("Redémarrer de zéro", default=False):
            command.append("--no-resume")

    if member_key == "member4":
        print(f"\n{BOLD('Clustering :')} {GREEN('connected_components')}")
        print(DIM("  fermeture transitive, algorithme conservé dans le projet"))

    if member_key == "member1":
        if _prompt_yes_no("Générer uniquement les fixtures mock (pas de données brutes)", default=False):
            command.append("--mocks-only")
    else:
        if _prompt_yes_no("Mode mock", default=False):
            command.append("--mock")

    if _prompt_yes_no("Ignorer les fichiers manquants", default=False):
        command.append("--skip-missing")

    return command


# ── Intégration d'un nouveau dataset ─────────────────────────────────────────

def _build_integrate_command() -> list[str]:
    title = "Intégration d'un nouveau dataset"
    print(f"\n{BOLD(title)}")
    print(DIM("  Le script analyse le répertoire/zip et détecte automatiquement le format."))

    source = _read_input("\n  Chemin vers le répertoire ou .zip du dataset: ").strip()
    if not source:
        print("  Chemin requis.")
        return []
    if _is_quit(source):
        raise PromptAborted("Interaction annulée.")

    name = _read_input("  Nom du dataset (ex: mon_dataset): ").strip()
    if not name:
        print("  Nom requis.")
        return []
    if _is_quit(name):
        raise PromptAborted("Interaction annulée.")

    s1 = _read_input("  Nom source 1 (ex: tablea, laisser vide = auto): ").strip() or None
    s2 = _read_input("  Nom source 2 (ex: tableb, laisser vide = auto): ").strip() or None
    sf1 = _read_input("  Fichier source 1 (ex: tablea.csv, laisser vide = auto): ").strip() or None
    sf2 = _read_input("  Fichier source 2 (ex: tableb.csv, laisser vide = auto): ").strip() or None
    gtf = _read_input("  Fichier ground truth (ex: gt.csv, laisser vide = auto): ").strip() or None

    dry = _prompt_yes_no("Mode dry-run (analyse sans écrire)", default=False)
    run = _prompt_yes_no("Lancer la pipeline après intégration", default=False)

    cmd = [_python_executable(), os.path.join(CLI_DIR, "integrate_dataset.py"),
           "--source", source, "--name", name]
    if s1:  cmd.extend(["--source1-name", s1])
    if s2:  cmd.extend(["--source2-name", s2])
    if sf1: cmd.extend(["--source1-file", sf1])
    if sf2: cmd.extend(["--source2-file", sf2])
    if gtf: cmd.extend(["--gt-file", gtf])
    if dry: cmd.append("--dry-run")
    if run: cmd.append("--run")

    return cmd


# ── Affichage du résumé des datasets ─────────────────────────────────────────

def _show_status() -> None:
    print(f"\n{BOLD('État des outputs :')}")
    output_root = REPO_ROOT / "output"
    datasets = _load_datasets()
    for d in datasets:
        name = d["key"]
        out  = output_root / name
        if not out.exists():
            status = RED("— pas encore exécuté")
        else:
            labels = []
            if (out / "blocks.csv").exists():        labels.append("blocks")
            if (out / "candidate_pairs.csv").exists(): labels.append("candidats")
            if (out / "match_results_collective.csv").exists(): labels.append("matching")
            if (out / "clusters.csv").exists():       labels.append("clusters")
            status = GREEN("✓ " + " → ".join(labels)) if labels else DIM("répertoire vide")
        print(f"  {name:<22} {status}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive launcher pour la pipeline d'entity resolution."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Afficher l'état des outputs puis quitter.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Afficher la liste des datasets disponibles puis quitter.",
    )
    return parser.parse_args()


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.list_datasets:
        _print_dataset_table()
        return

    if args.status:
        _show_status()
        return

    print(f"\n{BOLD('╔══════════════════════════════════════════╗')}")
    print(f"{BOLD('║   Entity Resolution Pipeline — Launcher  ║')}")
    print(f"{BOLD('╚══════════════════════════════════════════╝')}")
    _print_cancel_hint()

    try:
        action = _prompt_choice(
            "Que veux-tu faire ?",
            ["pipeline", "member1", "member2", "member3", "member4",
             "intégrer un dataset", "voir l'état des outputs"],
            default="pipeline",
            descriptions={
                "pipeline":              "chaîne complète (steps 1→4)",
                "member1":               "ingestion + blocking",
                "member2":               "block purging + candidats",
                "member3":               "matching SBERT/TF-IDF",
                "member4":               "clustering + fusion",
                "intégrer un dataset":   "ajouter un nouveau dataset au pipeline",
                "voir l'état des outputs": "résumé rapide des runs existants",
            },
        )

        if action == "voir l'état des outputs":
            _show_status()
            return

        if action == "intégrer un dataset":
            command = _build_integrate_command()
        elif action == "pipeline":
            command = _build_pipeline_command()
        else:
            command = _build_member_command(action)
        if not command:
            return

        pretty = " ".join(shlex.quote(part) for part in command)
        print(f"\n{BOLD('Commande construite :')}")
        print(f"  {CYAN(pretty)}")

        if not _prompt_yes_no("\nExécuter cette commande maintenant", default=True):
            print(DIM("  Commande non exécutée."))
            return

        print()
        raise SystemExit(subprocess.call(command, cwd=str(REPO_ROOT)))
    except PromptAborted as exc:
        print(f"\n{YELLOW('[INFO]')} {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
