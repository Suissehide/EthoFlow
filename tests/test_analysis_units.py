"""Unités des sorties de `analyze_vame.py`.

Trois garanties, que rien d'autre ne vérifie :

1. un en-tête de CSV porte l'unité de sa colonne, sans jamais la doubler
   (`n_valid_frames` ne devient pas `n_valid_frames_frames`) ;
2. `analysis_global_long.csv` empile toutes les mesures avec une colonne
   `unit` — c'est le fichier destiné à être relu ailleurs ;
3. les colonnes sans unité (p-values, identifiants, colonnes de l'Excel)
   sortent intactes : leur en coller une serait faux.

Rien ici ne dessine : `analyze_vame` rend son import de matplotlib
optionnel, donc ces tests tournent dans l'env `ethoflow` comme le reste
de la suite, sans l'env `vame`.
"""
from __future__ import annotations

import pandas as pd
import pytest

import analyze_vame as av


# ============================================================
# Nommage des colonnes
# ============================================================

def test_suffixe_ajoute_quand_il_manque():
    assert av.column_with_unit("frequency", av.UNIT_PROP) == "frequency_prop"
    assert av.column_with_unit("count", av.UNIT_FRAMES) == "count_frames"
    assert av.column_with_unit("mean_duration", av.UNIT_SEC) == "mean_duration_s"


def test_suffixe_pas_double_si_le_nom_le_dit_deja():
    """Sinon on écrirait `n_valid_frames_frames`."""
    assert av.column_with_unit("n_valid_frames", av.UNIT_FRAMES) == "n_valid_frames"
    assert av.column_with_unit("n_frames_total", av.UNIT_FRAMES) == "n_frames_total"
    assert av.column_with_unit("arena_radius_px", av.UNIT_PX) == "arena_radius_px"
    assert av.column_with_unit("n_bouts", av.UNIT_BOUTS) == "n_bouts"


def test_libelle_daxe_porte_lunite():
    assert av.axis_label("Usage", av.UNIT_PROP).startswith("Usage (")
    assert "0–1" in av.axis_label("Usage", av.UNIT_PROP)
    assert av.axis_label("Durée", av.UNIT_SEC) == "Durée (s)"


def test_unites_des_stats_suivent_les_noms_de_groupes():
    """Les colonnes d'un CSV de stats s'appellent d'après les groupes de
    l'utilisateur : elles ne peuvent pas être listées à l'avance."""
    units = av.units_for_stats(
        ["motif", "mean_Captopril", "n_Captopril", "diff", "p_value"]
    )
    assert units["mean_Captopril"] == av.UNIT_PROP
    assert units["n_Captopril"] == av.UNIT_SESSIONS
    assert units["diff"] == av.UNIT_PROP
    assert "p_value" not in units       # une p-value n'a pas d'unité


def test_ecriture_csv_renomme_puis_suffixe(tmp_path):
    df = pd.DataFrame({"condition": ["a"], "mean": [1.5], "count": [3],
                       "p_value": [0.04]})
    out = tmp_path / "bouts.csv"
    av.write_csv_with_units(
        df, out,
        {"mean_duration": av.UNIT_SEC, "n_bouts": av.UNIT_BOUTS},
        rename={"mean": "mean_duration", "count": "n_bouts"},
    )
    entetes = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert entetes == ["condition", "mean_duration_s", "n_bouts", "p_value"]


# ============================================================
# CSV global
# ============================================================

@pytest.fixture
def usage_df() -> pd.DataFrame:
    """Deux sessions × deux motifs, avec une colonne libre de l'Excel."""
    lignes = []
    for session, groupe in (("S1", "ctrl"), ("S2", "traite")):
        for motif, label, cat in ((0, "walking", "Locomotion"),
                                  (1, "grooming", "Grooming")):
            lignes.append({
                "session_full": session, "session_id": session, "arena": "",
                "condition": groupe, "regime_alimentaire": "gras",
                "motif": motif, "label": label, "category": cat,
                "frequency": 0.4 if motif == 0 else 0.6, "count": 100,
            })
    return pd.DataFrame(lignes)


def test_global_empile_usage_et_effectifs_avec_leurs_unites(usage_df):
    g = av.build_global_long(usage_df)
    mesures = dict(zip(g["metric"], g["unit"]))
    assert mesures["usage"] == av.UNIT_PROP
    assert mesures["n_frames"] == av.UNIT_FRAMES
    assert set(g["level"]) == {"motif"}
    assert len(g) == 8          # 2 sessions × 2 motifs × 2 mesures


def test_global_recopie_les_colonnes_de_lexcel(usage_df):
    """Le fichier doit être utilisable seul : sans les facteurs
    expérimentaux sur chaque ligne, il faudrait re-joindre les metadata."""
    g = av.build_global_long(usage_df)
    assert set(g.columns) >= {"condition", "regime_alimentaire", "level",
                              "motif", "label", "category", "metric",
                              "value", "unit"}
    assert set(g["regime_alimentaire"]) == {"gras"}


def test_global_distingue_les_niveaux_dobservation(usage_df):
    """Motif, catégorie et session cohabitent — un `group_by` qui les
    mélangerait sommerait des choses incomparables."""
    cat_df = pd.DataFrame({
        "session_full": ["S1", "S2"],
        "category": ["Grooming", "Grooming"],
        "frequency_total": [0.6, 0.55],
    })
    validity = pd.DataFrame({
        "session_full": ["S1", "S2"], "n_frames_total": [1000, 1000],
        "n_empty_start": [50, 0], "n_empty_end": [0, 10],
        "valid_fraction": [0.95, 0.99],
    })
    g = av.build_global_long(usage_df, cat_df=cat_df, validity_df=validity)
    assert set(g["level"]) == {"motif", "category", "session"}
    session = g[g["level"] == "session"]
    assert set(session["metric"]) == {"n_frames_total", "n_frames_empty_arena",
                                      "valid_fraction"}
    # Les lignes de session et de catégorie n'ont pas de motif
    assert g.loc[g["level"] != "motif", "motif"].isna().all()


def test_global_ecarte_les_motifs_artifact(usage_df):
    """Les analyses étendues relisent chaque frame et voient les motifs
    marqués `artifact`, que `build_dataframe` a écartés. Le fichier global
    ne doit pas les réintroduire, sans nom ni catégorie."""
    bouts = pd.DataFrame({
        "session_full": ["S1"] * 3 + ["S2"] * 3,
        "motif": [0, 1, 9, 0, 1, 9],       # 9 = motif artifact
        "duration_sec": [1.0, 2.0, 0.1] * 2,
    })
    g = av.build_global_long(usage_df, bouts_df=bouts)
    assert set(g.loc[g["metric"] == "bout_duration_mean", "motif"]) == {0, 1}
    # et les mesures étendues héritent du label du motif
    bd = g[g["metric"] == "bout_duration_mean"]
    assert set(bd["label"]) == {"walking", "grooming"}
    assert set(bd["unit"]) == {av.UNIT_SEC}


def test_global_donne_le_rayon_darene_en_cm_si_le_projet_est_calibre(usage_df):
    """En pixels le rayon ne se compare pas d'un setup à l'autre ; en cm si,
    et l'unité de la ligne dit laquelle des deux on lit."""
    spatial_px = pd.DataFrame({"session_full": ["S1", "S2"],
                               "arena_radius_px": [200.0, 200.0]})
    g = av.build_global_long(usage_df, spatial_df=spatial_px)
    rayon = g[g["metric"] == "arena_radius"]
    assert set(rayon["unit"]) == {av.UNIT_PX}

    spatial_cm = spatial_px.assign(arena_radius_cm=[20.0, 20.0])
    g = av.build_global_long(usage_df, spatial_df=spatial_cm)
    rayon = g[g["metric"] == "arena_radius"]
    assert set(rayon["unit"]) == {av.UNIT_CM}
    assert list(rayon["value"]) == [20.0, 20.0]


# ============================================================
# Séparateur du CSV global
# ============================================================
# `analysis_global_long.csv` sort en `;` pour qu'Excel en locale française
# l'ouvre en colonnes. L'app doit donc deviner le séparateur au lieu de
# supposer `,` — sinon l'aperçu affiche une seule colonne.

from lib.analysis import csv_separator  # noqa: E402


def test_separateur_detecte_le_point_virgule(tmp_path):
    f = tmp_path / "analysis_global_long.csv"
    f.write_text("session_full;metric;value;unit\nS1;usage;0.36;prop\n",
                 encoding="utf-8")
    assert csv_separator(f) == ";"


def test_separateur_reste_la_virgule_pour_les_autres_csv(tmp_path):
    f = tmp_path / "motif_usage_long.csv"
    f.write_text("session_full,motif,frequency_prop\nS1,0,0.36\n",
                 encoding="utf-8")
    assert csv_separator(f) == ","


def test_separateur_dun_fichier_illisible_ne_leve_pas(tmp_path):
    """L'aperçu ne doit pas planter sur un fichier disparu entre le
    listing et la lecture."""
    assert csv_separator(tmp_path / "absent.csv") == ","


def test_le_csv_global_se_relit_avec_le_separateur_detecte(tmp_path):
    """Le tour complet : écrit comme `analyze_vame.py` l'écrit, relu
    comme l'app le relit."""
    df = pd.DataFrame({"session_full": ["S1"], "condition": ["ctrl"],
                       "metric": ["usage"], "value": [0.36],
                       "unit": [av.UNIT_PROP]})
    f = tmp_path / "analysis_global_long.csv"
    df.to_csv(f, index=False, sep=";")
    relu = pd.read_csv(f, sep=csv_separator(f))
    assert list(relu.columns) == list(df.columns)
    assert relu.loc[0, "value"] == pytest.approx(0.36)
