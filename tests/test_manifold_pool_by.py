"""`--pool-by` : nuage de fond restreint à un groupe, axes inchangés.

`--pool-all-sessions` donne un manifold sur tout le projet. Pour une
figure « les MCCiECKO » à côté d'une figure « les MCCf/f », il faut
filtrer le nuage — mais SANS réajuster UMAP, sinon les deux figures ont
des axes qui ne veulent pas dire la même chose et ne se comparent plus.
L'ajustement reste donc commun (le cache poolé est réutilisé tel quel) ;
seul l'affichage est filtré.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

import behavior_structure_gif as B


@pytest.fixture
def projet_sessions(tmp_path):
    """Projet EthoFlow avec 4 sessions, 2 conditions, une sans valeur."""
    valeurs = {
        "S1": "MCCiECKO",
        "S2": "MCCf/f",
        "S3": "MCCiECKO",
        "S4": None,          # metadata sans la colonne
    }
    for sid, cond in valeurs.items():
        d = tmp_path / "data" / "raw" / sid
        d.mkdir(parents=True)
        meta = {"source_video": f"/videos/{sid}.mp4"}
        if cond is not None:
            meta["condition"] = cond
        (d / "metadata.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    return tmp_path


def test_lecture_de_la_colonne_par_session(projet_sessions):
    vals = B.session_group_values(
        projet_sessions, ["S1", "S2", "S3", "S4"], "condition")
    assert vals == {"S1": "MCCiECKO", "S2": "MCCf/f", "S3": "MCCiECKO",
                    "S4": None}


def test_colonne_absente_partout(projet_sessions):
    vals = B.session_group_values(
        projet_sessions, ["S1", "S2"], "regime_alimentaire")
    assert vals == {"S1": None, "S2": None}


def test_masque_du_groupe(projet_sessions):
    """Les tranches viennent de `session_slices` du pool : le masque doit
    couvrir exactement les frames des sessions du groupe."""
    slices = {"S1": (0, 10), "S2": (10, 25), "S3": (25, 30), "S4": (30, 40)}
    masque = B.group_background_mask(
        projet_sessions, slices, "condition", "MCCiECKO", total=40)
    attendu = np.zeros(40, dtype=bool)
    attendu[0:10] = True
    attendu[25:30] = True
    assert masque.tolist() == attendu.tolist()
    assert masque.sum() == 15


def test_le_groupe_de_la_session_animee(projet_sessions):
    slices = {"S1": (0, 10), "S2": (10, 25)}
    assert B.resolve_pool_group(projet_sessions, slices, "condition", "S2") == "MCCf/f"


def test_session_animee_sans_valeur_echoue(projet_sessions):
    """Filtrer sur « les sessions dont la condition est vide » ne veut rien
    dire — mieux vaut le dire que produire une figure au contenu arbitraire."""
    slices = {"S4": (0, 10)}
    with pytest.raises(SystemExit):
        B.resolve_pool_group(projet_sessions, slices, "condition", "S4")


def test_colonne_inconnue_echoue_en_listant_les_colonnes(projet_sessions, capsys):
    slices = {"S1": (0, 10), "S2": (10, 25)}
    with pytest.raises(SystemExit):
        B.resolve_pool_group(projet_sessions, slices, "genotype", "S1")
    err = capsys.readouterr().err
    assert "condition" in err          # les colonnes disponibles sont listées


def test_groupe_a_une_seule_session_passe_avec_avertissement(projet_sessions, capsys):
    slices = {"S2": (0, 10), "S1": (10, 20)}
    masque = B.group_background_mask(
        projet_sessions, slices, "condition", "MCCf/f", total=20)
    assert masque.sum() == 10
    assert "1 session" in capsys.readouterr().err


def test_le_nom_de_sortie_porte_le_groupe():
    """Sans le groupe dans le nom, la figure MCCiECKO et la figure MCCf/f
    de la même session animée s'écrasent l'une l'autre."""
    a = B.pool_name_fragment(True, "condition", "MCCiECKO")
    b = B.pool_name_fragment(True, "condition", "MCCf/f")
    assert a != b
    assert a == "_pooled-condition-MCCiECKO"
    # `/` est une valeur Excel legitime, pas un nom de fichier legal
    assert "/" not in b
    assert b == "_pooled-condition-MCCf-f"


def test_nom_de_sortie_sans_pool_by():
    assert B.pool_name_fragment(True, None, None) == "_pooled"
    assert B.pool_name_fragment(False, None, None) == ""


def test_bornes_calculees_sur_le_pool_entier():
    """Le filtre de groupe ne suffit pas à rendre deux figures
    superposables : matplotlib cadre sur ce qui est tracé, donc un nuage
    filtré donnerait un zoom différent par groupe. Les bornes doivent
    venir du pool COMPLET, identiques d'une figure à l'autre."""
    # Chaque groupe couvre une partie STRICTE de l'étendue du pool :
    # c'est justement le cas où l'autoscale les cadrerait différemment.
    pool = np.array([[0.0, 0.0], [1.0, 2.0], [9.0, 18.0], [10.0, 20.0]])
    groupe_a = pool[:2]
    groupe_b = pool[2:]

    bornes_pool = B.axis_limits(pool)
    assert B.axis_limits(pool) == bornes_pool          # déterministe
    assert B.axis_limits(groupe_a) != bornes_pool      # le piège évité
    assert B.axis_limits(groupe_b) != bornes_pool

    (x0, x1), (y0, y1) = bornes_pool
    assert x0 < 0.0 and x1 > 10.0                      # marge des deux côtés
    assert y0 < 0.0 and y1 > 20.0


def test_bornes_supportent_un_nuage_degenere():
    """Tous les points au même endroit : pas de borne nulle, sinon
    matplotlib lève sur un intervalle vide."""
    (x0, x1), (y0, y1) = B.axis_limits(np.array([[3.0, 4.0], [3.0, 4.0]]))
    assert x0 < x1 and y0 < y1
