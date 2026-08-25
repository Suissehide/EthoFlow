"""Ouvrir un dossier dans l'explorateur de fichiers du système.

Sans Streamlit : une vue n'a pas le droit d'appeler un sous-processus (elle
place des widgets, la logique vit dans `lib/`). Même principe que
`lib/envcheck.py`, qui sonde les environnements conda — ce sont des appels
courts et synchrones, hors du système de jobs, qui n'a de sens que pour les
traitements longs du pipeline.

**Le dossier s'ouvre sur la machine qui HÉBERGE le serveur**, pas sur celle
qui affiche le navigateur. En usage local — le cas normal — ce sont les
mêmes. Lancée en `--server.address=0.0.0.0` et consultée depuis un autre
poste, l'app ouvrirait le dossier sur l'hôte, ce que l'utilisateur distant
ne verrait pas ; l'appelant est censé le dire.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Assez pour laisser l'explorateur démarrer, assez court pour ne pas figer
# un rerun Streamlit si la commande part en vrille.
TIMEOUT_S = 10


def _plateforme() -> str:
    """« darwin » | « windows » | « autre ».

    Un seul endroit décide de la plateforme, et il est indirect exprès :
    les tests le remplacent pour exercer les trois branches depuis
    n'importe quelle machine. Monkeypatcher `sys.platform` ferait la même
    chose en apparence, mais casse l'outillage de test qui le lit aussi.
    """
    if sys.platform == "darwin":
        return "darwin"
    if os.name == "nt":
        return "windows"
    return "autre"


def nom_explorateur() -> str:
    """Nom lisible de l'explorateur, pour les libellés d'interface."""
    return {
        "darwin": "Finder",
        "windows": "Explorateur",
    }.get(_plateforme(), "gestionnaire de fichiers")


def _commande(chemin: Path) -> list[str]:
    outil = {
        "darwin": "open",
        "windows": "explorer",
    }.get(_plateforme(), "xdg-open")
    return [outil, str(chemin)]


# Un humain choisit dans une boîte de dialogue : il lui faut le temps de
# naviguer. Le run Streamlit attend pendant ce temps — acceptable puisque
# c'est lui qui vient de cliquer — mais la borne évite qu'une fenêtre
# oubliée fige l'app indéfiniment.
TIMEOUT_DIALOGUE_S = 180


def _commande_dialogue(titre: str, depart: Path | None) -> list[str] | None:
    """Commande ouvrant un sélecteur de dossier natif, ou None si aucune."""
    plateforme = _plateforme()
    if plateforme == "darwin":
        # `POSIX path of` évite l'alias « Macintosh HD:Users:… » que
        # `choose folder` renvoie par défaut.
        script = f'POSIX path of (choose folder with prompt "{titre}"'
        if depart and depart.is_dir():
            script += f' default location POSIX file "{depart}"'
        script += ")"
        return ["osascript", "-e", script]
    if plateforme == "windows":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
            f'$d.Description = "{titre}";'
        )
        if depart and depart.is_dir():
            ps += f'$d.SelectedPath = "{depart}";'
        ps += (
            "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
        )
        return ["powershell", "-NoProfile", "-Command", ps]
    # Linux : zenity d'abord, kdialog ensuite. Aucun des deux n'est garanti.
    for outil in ("zenity", "kdialog"):
        if shutil.which(outil):
            if outil == "zenity":
                cmd = ["zenity", "--file-selection", "--directory",
                       f"--title={titre}"]
                if depart and depart.is_dir():
                    cmd.append(f"--filename={depart}/")
                return cmd
            return ["kdialog", "--getexistingdirectory",
                    str(depart if depart and depart.is_dir() else Path.home())]
    return None


def choisir_dossier(
    titre: str = "Choisir le dossier des projets",
    depart: Path | None = None,
) -> tuple[Path | None, str]:
    """Ouvre un sélecteur de dossier natif. Retourne (chemin ou None, message).

    Trois issues distinctes, que l'appelant doit pouvoir différencier :
      - un dossier choisi   → (Path, "")
      - annulation          → (None, "") — normal, ne rien afficher
      - impossible/échec    → (None, message) — à montrer

    **La fenêtre s'ouvre sur la machine qui HÉBERGE le serveur.** En local
    c'est celle de l'utilisateur ; consultée à distance, il ne la verrait
    pas s'ouvrir et l'app semblerait figée jusqu'au délai.
    """
    commande = _commande_dialogue(titre, depart)
    if commande is None:
        return None, (
            "Aucun sélecteur de dossier disponible sur cette machine "
            "(ni zenity ni kdialog) — saisis le chemin à la main."
        )
    try:
        proc = subprocess.run(
            commande, capture_output=True, text=True,
            timeout=TIMEOUT_DIALOGUE_S,
        )
    except FileNotFoundError:
        return None, f"Commande « {commande[0]} » introuvable."
    except subprocess.TimeoutExpired:
        return None, (
            f"Sélecteur sans réponse après {TIMEOUT_DIALOGUE_S} s — "
            "fenêtre restée ouverte ?"
        )
    except Exception as e:  # noqa: BLE001 — jamais faire tomber l'appelant
        return None, f"Échec du sélecteur : {e}"

    sortie = (proc.stdout or "").strip()
    if not sortie:
        # Annulation : osascript sort en code 1 avec « User canceled »,
        # zenity en code 1 sans rien, PowerShell en code 0 sans rien. Aucun
        # n'est une erreur à afficher.
        return None, ""
    return Path(sortie), ""


def ouvrir_dans_explorateur(chemin: Path) -> tuple[bool, str]:
    """Ouvre `chemin` dans l'explorateur. Retourne (succès, message).

    Ne lève jamais : un échec d'ouverture est une gêne, pas une raison de
    faire tomber la page qui l'a demandé.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        return False, f"{chemin} n'existe pas."
    if not chemin.is_dir():
        chemin = chemin.parent

    commande = _commande(chemin)
    try:
        proc = subprocess.run(
            commande, capture_output=True, text=True, timeout=TIMEOUT_S
        )
    except FileNotFoundError:
        return False, (
            f"Commande « {commande[0]} » introuvable sur cette machine."
        )
    except subprocess.TimeoutExpired:
        return False, f"« {commande[0]} » n'a pas répondu en {TIMEOUT_S} s."
    except Exception as e:  # noqa: BLE001 — jamais faire tomber l'appelant
        return False, f"Échec de l'ouverture : {e}"

    # `explorer.exe` renvoie 1 même quand il a bien ouvert la fenêtre :
    # traiter ce code comme un échec afficherait une erreur à chaque clic
    # sous Windows. Sur cette plateforme, on considère que le lancement a
    # réussi dès lors que la commande a pu être exécutée.
    if _plateforme() == "windows":
        return True, f"Dossier ouvert dans l'{nom_explorateur()}."

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, (
            f"« {commande[0]} » a échoué (code {proc.returncode})"
            + (f" : {detail}" if detail else ".")
        )
    return True, f"Dossier ouvert dans le {nom_explorateur()}."
