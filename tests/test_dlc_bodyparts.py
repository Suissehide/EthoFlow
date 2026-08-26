"""Keypoints par vue caméra : bottom-view ≠ top-view.

Le template n'avait qu'un seul jeu de keypoints, avec `left_ear`,
`right_ear`, `center` et `left_flank` — un jeu top-view, servi tel quel
aux projets bottom-view où les oreilles ne sont pas visibles et où le
menton, le poitrail et le ventre le sont. Le wizard demande déjà la vue
(`SUPERANIMAL_NAME`) : c'est elle qui choisit le jeu.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import _config as C

DLC_TRAINING = Path(__file__).resolve().parent.parent / "scripts" / "dlc_model-training"


def _wizard():
    spec = importlib.util.spec_from_file_location(
        "init_training_config", DLC_TRAINING / "00_init_training_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("bodyparts,skeleton", [
    (C.BODYPARTS_BOTTOMVIEW, C.SKELETON_BOTTOMVIEW),
    (C.BODYPARTS_TOPVIEW, C.SKELETON_TOPVIEW),
])
def test_le_skeleton_ne_relie_que_des_keypoints_existants(bodyparts, skeleton):
    """Une liaison vers un keypoint absent est silencieusement ignorée par
    DLC — donc invisible jusqu'au moment où le skeleton ne sert à rien."""
    connus = set(bodyparts)
    for liaison in skeleton:
        assert len(liaison) == 2, liaison
        for kp in liaison:
            assert kp in connus, f"{kp} n'est pas dans les bodyparts"


@pytest.mark.parametrize("bodyparts", [C.BODYPARTS_BOTTOMVIEW, C.BODYPARTS_TOPVIEW])
def test_pas_de_doublon(bodyparts):
    assert len(bodyparts) == len(set(bodyparts))


def test_bottomview_sans_oreilles_avec_ventre():
    """Par en dessous : pas d'oreilles, mais menton, poitrail et ventre."""
    bp = set(C.BODYPARTS_BOTTOMVIEW)
    assert not bp & {"left_ear", "right_ear"}
    assert {"chin", "chest_center", "belly_center"} <= bp


def test_topview_garde_les_oreilles():
    assert {"left_ear", "right_ear"} <= set(C.BODYPARTS_TOPVIEW)


def test_les_deux_jeux_partagent_le_socle_commun():
    """`nose` et `tail_base` sont les points de référence de l'alignement
    égocentrique VAME (`run_vame.py cmd_align`) : ils doivent exister dans
    les deux jeux, sinon l'alignement retombe sur un keypoint arbitraire."""
    for bp in (C.BODYPARTS_BOTTOMVIEW, C.BODYPARTS_TOPVIEW):
        assert {"nose", "tail_base", "tail_mid", "tail_tip",
                "front_paw_left", "front_paw_right",
                "hind_paw_left", "hind_paw_right"} <= set(bp)


def test_la_vue_choisit_le_jeu():
    assert C.KEYPOINTS_PAR_VUE["superanimal_quadruped"] == (
        C.BODYPARTS_BOTTOMVIEW, C.SKELETON_BOTTOMVIEW)
    assert C.KEYPOINTS_PAR_VUE["superanimal_topviewmouse"] == (
        C.BODYPARTS_TOPVIEW, C.SKELETON_TOPVIEW)


def test_defaults_du_template_suivent_son_superanimal():
    attendu = C.KEYPOINTS_PAR_VUE[C.SUPERANIMAL_NAME]
    assert (C.DEFAULT_BODYPARTS, C.DEFAULT_SKELETON) == attendu


def test_superanimal_inconnu_echoue_a_l_import(tmp_path):
    """Une faute de frappe sur SUPERANIMAL_NAME doit se voir tout de suite,
    pas produire 300 frames labellisées sur le mauvais jeu de keypoints."""
    src = (DLC_TRAINING / "_config.py").read_text(encoding="utf-8").replace(
        'SUPERANIMAL_NAME = "superanimal_quadruped"',
        'SUPERANIMAL_NAME = "superanimal_typo"')
    with pytest.raises(ValueError, match="SUPERANIMAL_NAME"):
        exec(compile(src, "_config_typo.py", "exec"), {"__name__": "x"})


@pytest.mark.parametrize("superanimal,attendu_ears", [
    ("superanimal_quadruped", False),
    ("superanimal_topviewmouse", True),
])
def test_le_wizard_genere_le_bon_jeu(superanimal, attendu_ears):
    """Bout en bout : ce que le wizard écrit dans le `_config.py` de
    l'utilisateur résout vers le jeu de la vue choisie."""
    texte = _wizard().render_config(
        project_name="p", experimenter="labo", workdir="/tmp/w",
        pilot_video="/tmp/v.mp4", superanimal=superanimal, n_auto_frames=120,
    )
    espace: dict = {"__name__": "genere"}
    exec(compile(texte, "_config_genere.py", "exec"), espace)
    assert ("left_ear" in espace["DEFAULT_BODYPARTS"]) is attendu_ears
    assert espace["SUPERANIMAL_NAME"] == superanimal
