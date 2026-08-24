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


# Les deux SEULS préfixes où `analyze_vame.py` intercale littéralement le
# mot `motif` entre `_by_` et l'axe (constaté dans les f-strings du script,
# ruling R17.1) — un axe nommé par l'utilisateur qui commence par `motif_`
# (ex : colonne Excel `motif_stage`) n'a rien à voir avec ça et ne doit
# jamais se faire amputer de ce préfixe.
_PREFIXES_MOTIF = ("stats_by_motif_", "temporal_by_motif_")


def _axe_par_heuristique(stem: str) -> str | None:
    """Extrait l'axe d'un nom de fichier `..._by_<axe>` (sans extension),
    sans connaître le vocabulaire réel des axes — dernier recours.

    `analyze_vame.py` place l'axe juste après le DERNIER `_by_` du nom
    (`boxplots_by_category_by_condition` -> `condition`), sauf pour les
    deux préfixes de `_PREFIXES_MOTIF`, où l'axe suit immédiatement le
    préfixe entier plutôt que le dernier `_by_`.

    Purement structurel : un axe dont le nom contient lui-même `_by_`
    (ex : colonne Excel `grouped_by_cage`) sera mal coupé ici — c'est
    justement ce que `group_analysis_files(..., axes_connus=...)` corrige
    quand le vocabulaire réel est disponible.
    """
    for prefixe in _PREFIXES_MOTIF:
        if stem.startswith(prefixe):
            axe = stem[len(prefixe):]
            return axe or None
    idx = stem.rfind("_by_")
    if idx == -1:
        return None
    axe = stem[idx + len("_by_"):]
    return axe or None


def _axe_par_axes_connus(stem: str, axes_connus: list[str]) -> str | None:
    """Fait correspondre `stem` à l'un des axes RÉELLEMENT découverts
    (via `--list-columns`), par suffixe le plus long — pas par position
    du dernier `_by_`.

    Résout l'ambiguïté qu'aucune règle purement syntaxique ne peut lever :
    si `cage` ET `grouped_by_cage` sont deux axes réels distincts,
    `mean_by_grouped_by_cage.png` doit aller sous `grouped_by_cage`, pas
    fusionner silencieusement dans `cage`. On essaie les deux connecteurs
    possibles (`_by_` normal, `_by_motif_` pour les deux préfixes
    spéciaux) et on garde le nom d'axe le plus long qui matche : c'est
    forcément le plus spécifique.
    """
    meilleur: str | None = None
    for axe in axes_connus:
        for connecteur in ("_by_", "_by_motif_"):
            if stem.endswith(connecteur + axe):
                if meilleur is None or len(axe) > len(meilleur):
                    meilleur = axe
    return meilleur


def group_analysis_files(
    paths: list[Path], axes_connus: list[str] | None = None,
) -> tuple[list[Path], dict[str, list[Path]], list[Path]]:
    """Regroupe les fichiers de `analysis/` par axe de comparaison.

    Retourne `(globaux, par_axe, autres)` :
    - `globaux` : fichiers qui ne varient jamais selon l'axe ;
    - `par_axe` : dict axe -> fichiers (triés), trié par nom d'axe ;
    - `autres` : fichiers qu'aucune règle ni aucun axe connu n'explique —
      jamais perdus, juste pas rattachés (ruling R17.1 : une disparition
      silencieuse est le pire résultat, pire qu'un mauvais classement).

    Si `axes_connus` est fourni (la page l'obtient via `--list-columns`,
    lu par `_axes_disponibles`), chaque fichier est d'abord confronté au
    vrai vocabulaire d'axes — plus fiable que toute règle inférée du nom
    de fichier — et seuls les fichiers qui ne matchent aucun axe connu
    retombent sur l'heuristique structurelle. Aucune liste de figures
    n'est codée en dur : un nouveau `xyz_by_<axe>.png` côté script est
    reconnu sans modification ici, que `axes_connus` soit fourni ou non.
    """
    globaux: list[Path] = []
    par_axe: dict[str, list[Path]] = {}
    autres: list[Path] = []
    for p in paths:
        if p.name in _FICHIERS_GLOBAUX:
            globaux.append(p)
            continue
        axe = None
        if axes_connus:
            axe = _axe_par_axes_connus(p.stem, axes_connus)
        if axe is None:
            axe = _axe_par_heuristique(p.stem)
        if axe is None:
            autres.append(p)
            continue
        par_axe.setdefault(axe, []).append(p)
    globaux.sort(key=lambda p: p.name)
    autres.sort(key=lambda p: p.name)
    return (
        globaux,
        {axe: sorted(fichiers, key=lambda p: p.name)
         for axe, fichiers in sorted(par_axe.items())},
        autres,
    )
