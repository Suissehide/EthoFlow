"""Qualité du modèle DLC du projet — lecture seule, sans DeepLabCut.

Trois questions, trois scripts de `scripts/dlc_model-training/` dont on
réutilise les fonctions telles quelles (aucune ne dépend de DLC) :

- l'entraînement converge-t-il ?     `watch_training`  (learning_stats.csv)
- quels keypoints sont mal prédits ? `keypoint_errors` (évaluation vs labels)
- le modèle généralise-t-il à une vidéo du projet ? `inspect_predictions`

Les seuils et les diagnostics restent dans les scripts : `verdict()` est
capturé tel qu'il s'affiche au terminal plutôt que réécrit ici.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

from lib.project import SCRIPTS_DIR, dlc_config_path, dlc_config_status

_DOSSIER = SCRIPTS_DIR / "dlc_model-training"
if str(_DOSSIER) not in sys.path:
    sys.path.insert(0, str(_DOSSIER))
import inspect_predictions as IP  # noqa: E402
import keypoint_errors as KE  # noqa: E402
import watch_training as WT  # noqa: E402
from paths import dlc_output_dir  # noqa: E402

DEFAULT_PCUTOFF = KE.DEFAULT_PCUTOFF


def modele_dir(project: Path) -> Path | None:
    """Dossier du projet DLC configuré, s'il existe sur le disque."""
    statut, _ = dlc_config_status(project)
    return Path(dlc_config_path(project)).parent if statut == "ok" else None


def stats_entrainement(model_dir: Path) -> list[dict]:
    """Un dict par `learning_stats*.csv` (plus récent d'abord) :
    `{chemin, df, epochs_prevues, mtime}`, `df` ayant une colonne `step`."""
    sorties = []
    for csv in WT.trouver_stats(model_dir):
        stats = WT.lire_stats(csv)
        if stats is None:
            continue
        df = pd.DataFrame(stats["valeurs"]).astype(float)
        df.insert(0, "step", stats["steps"])
        sorties.append({"chemin": csv, "df": df, "mtime": csv.stat().st_mtime,
                        "epochs_prevues": WT.epochs_prevues(csv)})
    return sorties


def erreurs_keypoints(model_dir: Path, pcutoff: float = DEFAULT_PCUTOFF
                      ) -> tuple[pd.DataFrame, dict, Path]:
    """(erreurs par keypoint + TOTAL, confusions, fichier de prédictions).

    Lève `FileNotFoundError` avec le diagnostic du script (ce qui existe
    réellement sur le disque) si l'évaluation n'a jamais tourné.
    """
    predictions = KE.find_predictions(model_dir)
    if predictions is None:
        raise FileNotFoundError(KE.diagnostic_absence(model_dir))
    gt = KE.load_ground_truth(model_dir)
    pred = pd.read_hdf(predictions)
    return (KE.per_keypoint_errors(gt, pred, pcutoff),
            KE.diagnostic_confusions(gt, pred), predictions)


def h5_sessions(project: Path) -> dict[str, Path]:
    """Session -> .h5 d'inférence brut (ni filtré, ni nettoyé) le plus récent."""
    racine = dlc_output_dir(Path(project))
    sorties = {}
    for dossier in sorted(racine.iterdir()) if racine.is_dir() else []:
        candidats = [p for p in dossier.glob("*.h5")
                     if not any(m in p.stem for m in ("filtered", "cleaned", "results"))]
        if candidats:
            sorties[dossier.name] = max(candidats, key=lambda p: p.stat().st_mtime)
    return sorties


def confiances(h5: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(résumé par keypoint, test de superposition G/D, verdict du script)."""
    df = pd.read_hdf(h5)
    tab = IP.resume(IP.likelihoods(df))
    sup = IP.test_superposition(df, IP.paires_gauche_droite(tab.index))
    texte = io.StringIO()
    with contextlib.redirect_stdout(texte):
        IP.verdict(tab, IP.likelihoods(df), sup)
    return tab, sup, texte.getvalue().strip()
