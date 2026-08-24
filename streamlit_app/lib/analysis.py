"""Lecture des sorties d'analyze_vame destinées à alimenter l'interface."""
from __future__ import annotations

import re
from pathlib import Path

# "  captopril                2 groupes : Captopril (8 sessions), Control (8 sessions)"
_LIGNE = re.compile(
    r"^\s{2,}(?P<nom>[A-Za-z_][\w]*)\s{2,}(?P<n>\d+)\s+groupes?\s*:\s*(?P<groupes>.+)$"
)


def parse_list_columns(texte: str) -> list[dict]:
    """Transforme la sortie de `analyze_vame --list-columns` en données.

    Retourne une liste de `{nom, n_groupes, groupes}`. Les lignes que
    `conda run` ajoute au flux sont ignorées : seules celles qui matchent
    la forme attendue sont retenues.
    """
    resultats: list[dict] = []
    for ligne in (texte or "").splitlines():
        m = _LIGNE.match(ligne.rstrip())
        if m:
            resultats.append({
                "nom": m.group("nom"),
                "n_groupes": int(m.group("n")),
                "groupes": m.group("groupes").strip(),
            })
    return resultats


# ============================================================
# Regroupement des sorties de analyze_vame par axe de comparaison
# ============================================================

# Fichiers produits une seule fois, jamais liés à un axe de comparaison —
# constatés en lançant réellement le script (voir task-17-report.md), pas
# une liste de "types de figures" à maintenir : ce sont les seuls noms que
# `analyze_vame.py` écrit sans jamais varier selon l'axe.
_FICHIERS_GLOBAUX = {
    "motif_usage.csv", "motif_usage_long.csv",
    "heatmap_usage.png", "validity_per_session.csv",
    "usage_by_category.csv",
}


def _axe_depuis_nom(stem: str) -> str | None:
    """Extrait l'axe d'un nom de fichier `..._by_<axe>` (sans extension).

    `analyze_vame.py` place toujours l'axe juste après le DERNIER `_by_`
    du nom (`boxplots_by_category_by_condition` -> `condition`), sauf pour
    `stats_by_motif_<axe>` et `temporal_by_motif_<axe>`, où le mot `motif`
    s'intercale entre `_by_` et l'axe — seule exception documentée, retirée
    ici plutôt que par une liste de préfixes à maintenir.
    """
    idx = stem.rfind("_by_")
    if idx == -1:
        return None
    axe = stem[idx + len("_by_"):]
    if axe.startswith("motif_"):
        axe = axe[len("motif_"):]
    return axe or None


def group_analysis_files(paths: list[Path]) -> tuple[list[Path], dict[str, list[Path]]]:
    """Regroupe les fichiers de `analysis/` par axe de comparaison.

    Retourne `(globaux, par_axe)` : `globaux` les fichiers qui ne varient
    jamais selon l'axe, `par_axe` un dict axe -> fichiers (triés), trié par
    nom d'axe. Aucune liste de figures n'est codée en dur — un nouveau type
    de figure `xyz_by_<axe>.png` ajouté côté script est reconnu sans
    modification ici.
    """
    globaux: list[Path] = []
    par_axe: dict[str, list[Path]] = {}
    for p in paths:
        if p.name in _FICHIERS_GLOBAUX:
            globaux.append(p)
            continue
        axe = _axe_depuis_nom(p.stem)
        if axe is None:
            globaux.append(p)
            continue
        par_axe.setdefault(axe, []).append(p)
    globaux.sort(key=lambda p: p.name)
    return globaux, {axe: sorted(fichiers, key=lambda p: p.name)
                      for axe, fichiers in sorted(par_axe.items())}
