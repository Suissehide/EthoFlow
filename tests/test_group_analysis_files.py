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
"""
from __future__ import annotations

from pathlib import Path

from lib.analysis import group_analysis_files


def _paths(*noms: str) -> list[Path]:
    return [Path(n) for n in noms]


def test_fichiers_globaux_pas_rattaches_a_un_axe():
    globaux, par_axe = group_analysis_files(_paths(
        "motif_usage.csv", "motif_usage_long.csv", "heatmap_usage.png",
        "validity_per_session.csv", "usage_by_category.csv",
    ))
    assert {p.name for p in globaux} == {
        "motif_usage.csv", "motif_usage_long.csv", "heatmap_usage.png",
        "validity_per_session.csv", "usage_by_category.csv",
    }
    assert par_axe == {}


def test_fichiers_simples_by_axe():
    _, par_axe = group_analysis_files(_paths(
        "heatmap_usage_by_captopril.png",
        "mean_by_captopril.png",
        "boxplots_top_by_captopril.png",
    ))
    assert set(par_axe.keys()) == {"captopril"}
    assert len(par_axe["captopril"]) == 3


def test_boxplots_by_category_by_axe_extrait_le_dernier_by():
    _, par_axe = group_analysis_files(_paths(
        "boxplots_by_category_by_condition.png",
    ))
    assert set(par_axe.keys()) == {"condition"}


def test_stats_by_motif_et_temporal_by_motif_retirent_linfixe_motif():
    _, par_axe = group_analysis_files(_paths(
        "stats_by_motif_condition.csv",
        "temporal_by_motif_condition.png",
    ))
    assert set(par_axe.keys()) == {"condition"}
    assert len(par_axe["condition"]) == 2


def test_axe_composite_avec_underscore_dans_son_nom():
    _, par_axe = group_analysis_files(_paths(
        "mean_by_condition_x_captopril.png",
        "stats_by_motif_condition_x_captopril.csv",
    ))
    assert set(par_axe.keys()) == {"condition_x_captopril"}
    assert len(par_axe["condition_x_captopril"]) == 2


def test_axes_multiples_restent_separes():
    _, par_axe = group_analysis_files(_paths(
        "mean_by_captopril.png",
        "mean_by_condition.png",
        "stats_by_motif_captopril.csv",
    ))
    assert set(par_axe.keys()) == {"captopril", "condition"}
    assert len(par_axe["captopril"]) == 2
    assert len(par_axe["condition"]) == 1
