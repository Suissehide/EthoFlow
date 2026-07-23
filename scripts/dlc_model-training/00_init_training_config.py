"""Wizard interactif pour créer un dossier de config d'entraînement DLC.

Génère un `_config.py` custom à un chemin de ton choix, en dehors du repo.
Évite d'éditer directement le template `_config.py` versionné dans le repo,
et permet d'avoir plusieurs configs en parallèle (une par projet DLC).

Le fichier généré contient tous les paramètres du template mais avec les
valeurs que tu saisis à l'invite. Tu peux relancer ce wizard plus tard
pour recréer un config (il t'avertira si le fichier existe déjà).

Après génération, tous les scripts numérotés (01 → 06) acceptent un flag
`--config-dir <chemin>` qui pointe vers le dossier contenant ton
`_config.py`. Sans ce flag, ils retombent sur le template du repo.

Usage :
    python scripts/dlc_model-training/00_init_training_config.py

Exemple de session interactive :
    Dossier de travail : [D:/EthoFlow/models]
    Nom du projet DLC : souris-bottomview
      → dossier de config : D:/EthoFlow/models/souris-bottomview
    Identifiant expérimentateur : labo
    Chemin de la vidéo pilote : D:/data/bottom_view/970.mp4
    SuperAnimal [quadruped/topviewmouse] : quadruped
    → Écrit : D:/EthoFlow/models/souris-bottomview/_config.py

    Étape suivante :
      python scripts/dlc_model-training/01_setup_project.py \\
          --config-dir D:/EthoFlow/models/souris-bottomview

Le _config.py et le projet DLC lui-même vivent côte à côte sous le
dossier de travail :
    D:/EthoFlow/models/souris-bottomview/         ← _config.py
    D:/EthoFlow/models/souris-bottomview-labo-2026-YY-MM/   ← projet DLC créé par 01
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


TEMPLATE_PATH = Path(__file__).resolve().parent / "_config.py"


def prompt(question: str, default: str | None = None,
            choices: list[str] | None = None) -> str:
    """Prompt utilisateur avec valeur par défaut + validation optionnelle."""
    while True:
        suffix = f" [{default}]" if default else ""
        if choices:
            suffix = f" [{'/'.join(choices)}]" + (f" (défaut {default})" if default else "")
        raw = input(f"{question}{suffix} : ").strip()
        val = raw or (default or "")
        if not val:
            print("  ⚠ valeur requise")
            continue
        if choices and val not in choices:
            print(f"  ⚠ choix invalide, attendus : {choices}")
            continue
        return val


def render_config(
    project_name: str,
    experimenter: str,
    workdir: str,
    pilot_video: str,
    superanimal: str,
    n_auto_frames: int,
) -> str:
    """Génère le contenu texte d'un _config.py rempli avec les valeurs
    saisies.

    On repart du template du repo et on remplace les valeurs des
    constantes clés. Le reste (paramètres MOG2, EPOCHS, etc.) garde les
    défauts, éditables à la main plus tard.
    """
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template introuvable : {TEMPLATE_PATH}")
    tpl = TEMPLATE_PATH.read_text(encoding="utf-8")

    # Remplacements ciblés — on modifie uniquement les lignes-valeurs
    # évidentes pour rester robuste aux évolutions du template.
    replacements = [
        (r'PROJECT_NAME = "souris-bottomview"',
         f'PROJECT_NAME = "{project_name}"'),
        (r'EXPERIMENTER = "labo"',
         f'EXPERIMENTER = "{experimenter}"'),
        (r'WORKDIR = Path(r"D:\EthoFlow\models")',
         f'WORKDIR = Path(r"{workdir}")'),
        (r'PILOT_VIDEO = Path(r"D:\path\to\pilot_video.mp4")',
         f'PILOT_VIDEO = Path(r"{pilot_video}")'),
        (r'SUPERANIMAL_NAME = "superanimal_quadruped"',
         f'SUPERANIMAL_NAME = "{superanimal}"'),
        (r'N_AUTO_FRAMES = 120',
         f'N_AUTO_FRAMES = {n_auto_frames}'),
    ]
    n_hits = 0
    for old, new in replacements:
        if old in tpl:
            tpl = tpl.replace(old, new)
            n_hits += 1
        else:
            print(f"  ⚠ Motif non trouvé dans le template : {old!r}",
                  file=sys.stderr)
    if n_hits < len(replacements):
        print("  ⚠ Certains motifs n'ont pas matché — le template a peut-être "
              "évolué. Ouvre le fichier généré pour vérifier.", file=sys.stderr)
    return tpl


def main() -> None:
    print("=" * 60)
    print("Wizard init config d'entraînement DLC")
    print("=" * 60)
    print()

    # ---- Dossier de travail (racine commune config + projet DLC) ----
    workdir_str = prompt(
        "Dossier de travail (où config + projet DLC seront créés)",
        default=r"D:\EthoFlow\models",
    )
    workdir = workdir_str

    # ---- Nom du projet ----
    project_name = prompt("Nom du projet DLC (arbitraire)",
                          default="souris-bottomview")

    # ---- Dossier de config = <workdir>/<project_name> (computed) ----
    config_dir = Path(workdir) / project_name
    print(f"  → dossier de config : {config_dir}")
    target = config_dir / "_config.py"
    if target.exists():
        overwrite = prompt(
            f"{target} existe déjà. Écraser ?",
            default="n", choices=["y", "n"],
        )
        if overwrite != "y":
            print("Annulé.")
            sys.exit(0)

    # ---- Identifiant expérimentateur ----
    experimenter = prompt("Identifiant expérimentateur (utilisé par DLC "
                          "dans les noms de fichiers)",
                          default="labo")

    # ---- Vidéo pilote ----
    while True:
        pilot_str = prompt("Chemin de la vidéo pilote (.mp4)")
        pilot = Path(pilot_str)
        if pilot.exists():
            break
        print(f"  ⚠ Fichier introuvable : {pilot}")
        retry = prompt("Continuer quand même (le chemin sera écrit tel quel) ?",
                       default="n", choices=["y", "n"])
        if retry == "y":
            break

    # ---- SuperAnimal ----
    print()
    print("Choix du modèle SuperAnimal pour le transfer learning :")
    print("  - quadruped   : vue latérale / bottom-view (voit les pattes)")
    print("  - topviewmouse : top-view rongeurs (pattes non visibles)")
    sa_choice = prompt("SuperAnimal", default="quadruped",
                        choices=["quadruped", "topviewmouse"])
    superanimal = ("superanimal_quadruped" if sa_choice == "quadruped"
                    else "superanimal_topviewmouse")

    # ---- N frames k-means (défaut sensé recommandé Tony/LIN) ----
    n_frames_str = prompt(
        "Nombre de frames k-means à extraire au setup",
        default="120",
    )
    try:
        n_auto_frames = int(n_frames_str)
    except ValueError:
        print("  ⚠ Valeur non entière, on utilise 120")
        n_auto_frames = 120

    # ---- Génération ----
    config_dir.mkdir(parents=True, exist_ok=True)
    content = render_config(
        project_name=project_name,
        experimenter=experimenter,
        workdir=workdir.replace("\\", "\\\\"),  # échappement Windows dans le raw string
        pilot_video=pilot_str.replace("\\", "\\\\"),
        superanimal=superanimal,
        n_auto_frames=n_auto_frames,
    )
    target.write_text(content, encoding="utf-8")

    print()
    print(f"✅ Écrit : {target}")
    print()
    print("Étape suivante :")
    print(f"  conda activate dlc")
    print(f"  python scripts\\dlc_model-training\\01_setup_project.py \\")
    print(f"      --config-dir {config_dir}")
    print()
    print("Le script 01 va :")
    print(f"  1. Créer le projet DLC dans {workdir}\\{project_name}-{experimenter}-<date>\\")
    print(f"     (DLC ajoute la date du jour au nom du dossier)")
    print(f"  2. Écrire automatiquement les 12 bodyparts + skeleton dans config.yaml")
    print(f"  3. Régler numframes2pick = {n_auto_frames}")
    print(f"  4. Extraire les {n_auto_frames} frames k-means")
    print(f"  5. Mettre à jour PROJECT_DIR dans ton _config.py pour pointer")
    print(f"     vers le dossier créé (aucune édition manuelle nécessaire)")


if __name__ == "__main__":
    main()
