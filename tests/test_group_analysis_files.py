"""Regroupement des sorties de analyze_vame par axe de comparaison.

`analyze_vame.py` nomme ses fichiers `<type>_by_<axe>.<ext>` — sauf deux
exceptions qui glissent le mot `motif` entre `_by_` et l'axe
(`stats_by_motif_<axe>.csv`, `temporal_by_motif_<axe>.png`), et une poignée
de fichiers globaux (jamais liés à un axe) : `motif_usage.csv`,
`motif_usage_long.csv`, `heatmap_usage.png`, `validity_per_session.csv`,
`usage_by_category.csv`.

Vrais noms observés en lançant `analyze_vame.py` (voir task-17-report.md) :
heatmap_usage_by_captopril.png, mean_by_captopril.png,
boxplots_top_by_captopril.png, boxplots_by_category_by_captopril.png,
stats_by_motif_captopril.csv, mean_by_condition_x_captopril.png (axe composite).

Ruling R17.1 (fix round 1) : les noms d'axe sont des colonnes Excel
arbitraires, pas une structure fixe. `_by_motif_` ne doit être retiré que
pour les deux préfixes qui le produisent réellement
(`stats_by_motif_`, `temporal_by_motif_`) — pas pour tout axe dont le nom
commence par `motif_` (ex : `motif_stage`). Et un axe dont le nom contient
lui-même `_by_` (ex : `grouped_by_cage`, sur le modèle de l'exemple `cage`
du brief) ne doit pas silencieusement fusionner avec un axe réel `cage` —
`axes_connus`, quand disponible (après « Découvrir les axes »), tranche
par correspondance de suffixe la plus longue plutôt que par la position
du dernier `_by_`.
"""
from __future__ import annotations

from pathlib import Path

from lib.analysis import group_analysis_files


def _paths(*noms: str) -> list[Path]:
    return [Path(n) for n in noms]


def test_fichiers_globaux_pas_rattaches_a_un_axe():
    globaux, par_axe, autres = group_analysis_files(_paths(
        "motif_usage.csv", "motif_usage_long.csv", "heatmap_usage.png",
        "validity_per_session.csv", "usage_by_category.csv",
    ))
    assert {p.name for p in globaux} == {
        "motif_usage.csv", "motif_usage_long.csv", "heatmap_usage.png",
        "validity_per_session.csv", "usage_by_category.csv",
    }
    assert par_axe == {}
    assert autres == []


def test_fichiers_simples_by_axe():
    _, par_axe, _ = group_analysis_files(_paths(
        "heatmap_usage_by_captopril.png",
        "mean_by_captopril.png",
        "boxplots_top_by_captopril.png",
    ))
    assert set(par_axe.keys()) == {"captopril"}
    assert len(par_axe["captopril"]) == 3


def test_boxplots_by_category_by_axe_extrait_le_dernier_by():
    _, par_axe, _ = group_analysis_files(_paths(
        "boxplots_by_category_by_condition.png",
    ))
    assert set(par_axe.keys()) == {"condition"}


def test_stats_by_motif_et_temporal_by_motif_retirent_linfixe_motif():
    _, par_axe, _ = group_analysis_files(_paths(
        "stats_by_motif_condition.csv",
        "temporal_by_motif_condition.png",
    ))
    assert set(par_axe.keys()) == {"condition"}
    assert len(par_axe["condition"]) == 2


def test_axe_composite_avec_underscore_dans_son_nom():
    _, par_axe, _ = group_analysis_files(_paths(
        "mean_by_condition_x_captopril.png",
        "stats_by_motif_condition_x_captopril.csv",
    ))
    assert set(par_axe.keys()) == {"condition_x_captopril"}
    assert len(par_axe["condition_x_captopril"]) == 2


def test_axes_multiples_restent_separes():
    _, par_axe, _ = group_analysis_files(_paths(
        "mean_by_captopril.png",
        "mean_by_condition.png",
        "stats_by_motif_captopril.csv",
    ))
    assert set(par_axe.keys()) == {"captopril", "condition"}
    assert len(par_axe["captopril"]) == 2
    assert len(par_axe["condition"]) == 1


# ============================================================
# Ruling R17.1 — régressions des trois cas de mauvais classement
# ============================================================

def test_r17_1_axe_utilisateur_nomme_motif_x_pas_ampute_du_prefixe():
    """`mean_by_motif_stage.png` : l'axe s'appelle `motif_stage`, entier.

    Avant le correctif, le strip de `motif_` était déclenché sur N'IMPORTE
    quel axe commençant par `motif_`, pas seulement sur les deux préfixes
    qui le produisent réellement — l'axe se retrouvait amputé en `stage`.
    """
    _, par_axe, _ = group_analysis_files(_paths("mean_by_motif_stage.png"))
    assert set(par_axe.keys()) == {"motif_stage"}
    assert "stage" not in par_axe


def test_r17_1_stats_by_motif_avec_axe_qui_sappelle_motif_stage():
    """`stats_by_motif_motif_stage.csv` : préfixe `stats_by_motif_` réel,
    puis l'axe entier `motif_stage` (qui contient lui-même `motif_`)."""
    _, par_axe, _ = group_analysis_files(
        _paths("stats_by_motif_motif_stage.csv"))
    assert set(par_axe.keys()) == {"motif_stage"}


def test_r17_1_axe_contenant_by_sans_axes_connus_reste_ambigu():
    """Sans vocabulaire d'axes connus, `mean_by_grouped_by_cage.png` se
    range par défaut (heuristique seule) sous l'axe `cage` — comportement
    documenté, pas une régression : c'est justement ce que `axes_connus`
    doit pouvoir corriger (test suivant)."""
    _, par_axe, _ = group_analysis_files(
        _paths("mean_by_grouped_by_cage.png"))
    assert set(par_axe.keys()) == {"cage"}


def test_r17_1_axes_connus_leve_lambiguite_grouped_by_cage():
    """Avec le vocabulaire réel (`cage` ET `grouped_by_cage` existent
    tous les deux comme axes), la correspondance de suffixe la plus
    longue empêche la fusion silencieuse dans le mauvais axe : chaque
    fichier doit rester sous SON propre axe, jamais les deux ensemble."""
    _, par_axe, _ = group_analysis_files(
        _paths("mean_by_cage.png", "mean_by_grouped_by_cage.png"),
        axes_connus=["cage", "grouped_by_cage"],
    )
    assert set(par_axe.keys()) == {"cage", "grouped_by_cage"}
    assert [p.name for p in par_axe["cage"]] == ["mean_by_cage.png"]
    assert [p.name for p in par_axe["grouped_by_cage"]] == [
        "mean_by_grouped_by_cage.png"]


def test_r17_1_fichier_non_reconnu_va_dans_autres_pas_perdu():
    """Un fichier qui ne matche ni un axe connu, ni l'heuristique `_by_`,
    ni la liste des globaux ne doit pas disparaître — il doit rester
    visible dans un panier « autres »."""
    _, par_axe, autres = group_analysis_files(
        _paths("readme_notes.txt", "mean_by_captopril.png"),
        axes_connus=["captopril"],
    )
    assert [p.name for p in autres] == ["readme_notes.txt"]
    assert "readme_notes" not in par_axe
    assert set(par_axe.keys()) == {"captopril"}
