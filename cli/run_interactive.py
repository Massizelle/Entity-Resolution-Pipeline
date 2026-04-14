"""
Interactive launcher for the ER pipeline.

The goal is pragmatic:
- ask a small set of explicit questions
- build the right command
- execute it immediately

No external dependencies, no magic configuration layer.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CLI_DIR = os.path.dirname(__file__)

DATASETS = [
    "abt_buy",
    "amazon_google",
    "dblp_acm",
    "spimbench",
    "rexa_dblp",
    "all",
]

MEMBERS = {
    "pipeline": os.path.join(CLI_DIR, "run_pipeline.py"),
    "member1": os.path.join(CLI_DIR, "run_member1.py"),
    "member2": os.path.join(CLI_DIR, "run_member2.py"),
    "member3": os.path.join(CLI_DIR, "run_member3.py"),
    "member4": os.path.join(CLI_DIR, "run_member4.py"),
}

CLUSTERING_ALGOS = ["connected_components", "center"]
CANDIDATE_STRATEGIES = ["v0", "cw_semantic_predictive"]


def _prompt_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    print(f"\n{prompt}")
    for idx, option in enumerate(options, start=1):
        marker = " (default)" if option == default else ""
        print(f"  {idx}. {option}{marker}")
    while True:
        raw = input("> ").strip()
        if not raw and default is not None:
            return default
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(options):
                return options[index]
        if raw in options:
            return raw
        print("Choix invalide. Réessaie.")


def _prompt_int(prompt: str, default: int | None = None) -> int | None:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Valeur invalide. Entier attendu.")


def _prompt_float(prompt: str, default: float | None = None) -> float | None:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Valeur invalide. Nombre attendu.")


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    raw = input(f"{prompt}{suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "o", "oui"}


def _python_executable() -> str:
    venv_python = os.path.join(".", "venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def _build_pipeline_command() -> list[str]:
    command = [_python_executable(), MEMBERS["pipeline"]]

    dataset = _prompt_choice("Choisis le dataset", DATASETS, default="abt_buy")
    command.extend(["--dataset", dataset])

    from_step = _prompt_int("Étape de début", 1)
    to_step = _prompt_int("Étape de fin", 4)
    command.extend(["--from-step", str(from_step), "--to-step", str(to_step)])

    candidate_strategy = _prompt_choice(
        "Choisis la stratégie de candidats",
        CANDIDATE_STRATEGIES,
        default="cw_semantic_predictive",
    )
    command.extend(["--candidate-strategy", candidate_strategy])

    if from_step <= 3 <= to_step:
        time_limit = _prompt_float("Budget temps Step 3 en minutes (vide = illimité)", None)
        if time_limit is not None:
            command.extend(["--time-limit-minutes", str(time_limit)])

        chunk_size = _prompt_int("Chunk size pour Step 3", 250)
        if chunk_size is not None:
            command.extend(["--chunk-size", str(chunk_size)])

        online = _prompt_yes_no("Activer le clustering incrémental pendant Step 3", default=False)
        if online:
            command.append("--online-clustering")
            every_n = _prompt_int("Rafraîchir les clusters tous les N chunks", 1)
            if every_n is not None:
                command.extend(["--online-cluster-every-n-chunks", str(every_n)])

        no_resume = _prompt_yes_no("Redémarrer de zéro (désactiver resume)", default=False)
        if no_resume:
            command.append("--no-resume")

    if from_step <= 4 <= to_step or "--online-clustering" in command:
        algorithm = _prompt_choice(
            "Choisis l'algorithme de clustering",
            CLUSTERING_ALGOS,
            default="connected_components",
        )
        command.extend(["--algorithm", algorithm])

    limit = _prompt_int("Limiter le nombre de paires Step 3 (vide = aucune limite)", None)
    if limit is not None:
        command.extend(["--limit", str(limit)])

    if _prompt_yes_no("Mode mock", default=False):
        command.append("--mock")

    if _prompt_yes_no("Ignorer les fichiers manquants", default=False):
        command.append("--skip-missing")

    return command


def _build_member_command(member_key: str) -> list[str]:
    command = [_python_executable(), MEMBERS[member_key]]
    dataset = _prompt_choice("Choisis le dataset", DATASETS, default="abt_buy")
    command.extend(["--dataset", dataset])

    if member_key == "member2":
        strategy = _prompt_choice(
            "Choisis la stratégie de candidats",
            CANDIDATE_STRATEGIES,
            default="cw_semantic_predictive",
        )
        command.extend(["--candidate-strategy", strategy])

    if member_key == "member3":
        strategy = _prompt_choice(
            "Choisis la stratégie de candidats",
            CANDIDATE_STRATEGIES,
            default="cw_semantic_predictive",
        )
        command.extend(["--candidate-strategy", strategy])

        time_limit = _prompt_float("Budget temps en minutes (vide = illimité)", None)
        if time_limit is not None:
            command.extend(["--time-limit-minutes", str(time_limit)])

        chunk_size = _prompt_int("Chunk size", 250)
        if chunk_size is not None:
            command.extend(["--chunk-size", str(chunk_size)])

        if _prompt_yes_no("Activer le clustering incrémental", default=False):
            command.append("--online-clustering")
            algo = _prompt_choice(
                "Choisis l'algorithme de clustering",
                CLUSTERING_ALGOS,
                default="connected_components",
            )
            command.extend(["--clustering-algorithm", algo])
            every_n = _prompt_int("Rafraîchir les clusters tous les N chunks", 1)
            if every_n is not None:
                command.extend(["--online-cluster-every-n-chunks", str(every_n)])

        if _prompt_yes_no("Redémarrer de zéro (désactiver resume)", default=False):
            command.append("--no-resume")

    if member_key == "member4":
        algo = _prompt_choice(
            "Choisis l'algorithme de clustering",
            CLUSTERING_ALGOS,
            default="connected_components",
        )
        command.extend(["--algorithm", algo])

    if _prompt_yes_no("Mode mock", default=False):
        command.append("--mock")

    if _prompt_yes_no("Ignorer les fichiers manquants", default=False):
        command.append("--skip-missing")

    return command


def main() -> None:
    target = _prompt_choice(
        "Que veux-tu lancer ?",
        ["pipeline", "member1", "member2", "member3", "member4"],
        default="pipeline",
    )
    if target == "pipeline":
        command = _build_pipeline_command()
    else:
        command = _build_member_command(target)

    pretty = " ".join(shlex.quote(part) for part in command)
    print("\nCommande construite:")
    print(pretty)

    if not _prompt_yes_no("Exécuter cette commande maintenant", default=True):
        return

    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
