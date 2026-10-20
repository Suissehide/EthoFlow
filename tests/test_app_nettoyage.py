"""Vérification de bout en bout de la page Nettoyage via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme
`tests/test_app_pose.py`) pour ne jamais toucher `Path.home()` réel ni
`DEFAULT_PROJECTS_ROOT` (`D:\\EthoFlow\\projects`, un nom de dossier
littéral et relatif sur ce runner macOS).
"""
from __future__ import annotations

import base64
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")

# PNG 1x1 valide minimal — st.image() doit pouvoir l'ouvrir réellement,
# un fichier vide ferait planter le rendu plutôt que de tester la logique
# de la galerie.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _projet(tmp_path: Path, *, kind: str = "single",
           px_per_cm: float | None = None) -> Path:
    p = tmp_path / "projects" / "test-nettoyage"
    for sub in ("raw", "cropped", "dlc-output", "vame", "results"):
        (p / "data" / sub).mkdir(parents=True)
    (p / "configs").mkdir(parents=True)

    cfg: dict = {"kind": kind}
    if px_per_cm is not None:
        cfg["px_per_cm"] = px_per_cm

    (p / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
    )

    session_dir = p / "data" / "raw" / "S1"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"source_video": str(session_dir / "S1.mp4")}), encoding="utf-8",
    )

    return p


def _lancer_sur_projet(tmp_path: Path, monkeypatch, projet: Path) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": str(projet.parent), "models_root": str(tmp_path / "models")})
    (tmp_path / "models").mkdir(exist_ok=True)
    at = AppTest.from_file(APP_PY)
    at.session_state["current_project_path"] = str(projet)
    at.run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "nav_nettoyage" in boutons, list(boutons)
    boutons["nav_nettoyage"].click().run()
    assert not at.exception, at.exception
    return at


def test_avertissement_px_per_cm_absent(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=None)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert any("déplacements impossibles" in w.value and "désactivé" in w.value for w in at.warning), \
        [w.value for w in at.warning]


def test_pas_avertissement_si_px_per_cm_configure(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert not any("déplacements impossibles" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    assert any("12.5 pixels par cm" in c.value for c in at.caption), [c.value for c in at.caption]


def test_section_arenes_visible_pour_projet_multi(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="multi", px_per_cm=12.5)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert any("sépare d'abord" in c.value for c in at.caption), [c.value for c in at.caption]
    boutons = {b.key: b for b in at.button}
    assert "btn_nettoyage_arenes" in boutons


def test_section_arenes_remplacee_par_explication_pour_projet_single(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert any("rien à séparer" in c.value for c in at.caption), \
        [c.value for c in at.caption]
    boutons = {b.key: b for b in at.button}
    assert "btn_nettoyage_arenes" not in boutons


def test_expander_outils_avances_replie_par_defaut(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    expanders = [e for e in at.expander if "Outils avancés" in (e.label or "")]
    assert len(expanders) == 1
    assert expanders[0].proto.expanded is False


# ============================================================
# Task 21 — galerie QC des trajectoires
# ============================================================

def test_galerie_qc_avec_plots_offre_le_selecteur_de_keypoint_sans_grille(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    qc_dir = projet / "data" / "dlc-output" / "_qc_trajectories"
    qc_dir.mkdir(parents=True)
    (qc_dir / "S1_tail_base.png").write_bytes(_PNG_1X1)
    (qc_dir / "S1_paw_front_left.png").write_bytes(_PNG_1X1)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    selectbox = {s.key: s for s in at.selectbox}
    assert "nettoyage_qc_galerie_keypoint" in selectbox, list(selectbox)
    assert set(selectbox["nettoyage_qc_galerie_keypoint"].options) == {
        "tail_base", "paw_front_left",
    }
    # Pas de grille : le seul graphe affiché est celui d'« Avant / après »
    # (st.image n'est pas modélisé comme un type d'élément dédié par AppTest ;
    # il apparaît comme un UnknownElement de type "imgs", un par appel).
    assert len(at.get("imgs")) == 1, at.get("imgs")


def test_galerie_qc_sans_dossier_dit_qu_il_n_y_a_rien_plutot_que_planter(tmp_path, monkeypatch):
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    # Pas de data/dlc-output/_qc_trajectories/ du tout.
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert not at.exception, at.exception
    assert any("Aucun graphe de contrôle trouvé" in c.value for c in at.caption), \
        [c.value for c in at.caption]
    assert "nettoyage_qc_galerie_keypoint" not in {s.key for s in at.selectbox}


def test_galerie_qc_sans_labeled_mp4_est_presente_comme_normale(tmp_path, monkeypatch):
    """Aucun `_labeled.mp4` (typique du mode DLC `custom`, qui n'en produit
    jamais) : un simple message informatif, jamais un st.warning/st.error."""
    projet = _projet(tmp_path, kind="single", px_per_cm=12.5)
    qc_dir = projet / "data" / "dlc-output" / "_qc_trajectories"
    qc_dir.mkdir(parents=True)
    (qc_dir / "S1_tail_base.png").write_bytes(_PNG_1X1)
    # data/dlc-output/S1/ n'existe même pas — aucune inférence DLC ici.

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert not at.exception, at.exception
    assert any(
        "Pas de vidéo annotée" in c.value and "Mon modèle entraîné" in c.value
        for c in at.caption
    ), [c.value for c in at.caption]
    assert not any("vidéo annotée" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    assert not any("vidéo annotée" in e.value for e in at.error), \
        [e.value for e in at.error]


def test_toutes_les_sessions_ignore_la_selection_perimee(tmp_path, monkeypatch):
    """Régression : le multiselect désactivé garde son ancienne valeur, et
    prepare_vame_input (sans --all) la recevait malgré « Toutes » cochée."""
    import lib.pipeline as PL
    appels = []
    original = PL.prepare_vame_input
    monkeypatch.setattr(PL, "prepare_vame_input",
                        lambda *a, **kw: appels.append(kw) or original(*a, **kw))

    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    at.pills(key="nettoyage_clean_portee").set_value("choix").run()
    at.multiselect(key="nettoyage_clean_sel").select("S1").run()
    at.pills(key="nettoyage_clean_portee").set_value("tout").run()
    assert not at.exception, at.exception
    assert appels[-1]["sessions"] == []


def _job(projet: Path, job_id: str, script: str, log: str) -> None:
    import json
    d = projet / ".ethoflow" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "script": script, "env": "ethoflow", "label": script,
        "argv": ["python", script, "--project-dir", str(projet), "--no-prompt"],
        "started_at": "2026-01-01T00:00:00", "ended_at": "2026-01-01T00:01:00",
        "returncode": 0, "pid": None, "state": "succeeded",
        "cancel_requested": False, "owner_pid": None,
    }), encoding="utf-8")
    (d / f"{job_id}.log").write_text(log, encoding="utf-8")


def test_keypoint_choisi_dans_la_liste_du_modele(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    modele = tmp_path / "modele"
    modele.mkdir()
    (modele / "config.yaml").write_text(yaml.safe_dump({"bodyparts": ["nose", "tail_base", "paw"]}))
    P.set_dlc_config(projet, modele)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    choix = at.selectbox(key="nettoyage_qc_bodypart")
    assert choix.options == ["nose", "tail_base", "paw"] and choix.value == "tail_base"


def test_resume_survit_a_un_job_lance_apres_le_nettoyage(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    _job(projet, "20260101-000000-000000-nettoyage", "prepare_vame_input_custom.py",
         "[1/1] S1\n  ✓ 1000 frames | 12 kpts | médian | réparées 50 | 95.0% utilisables\n")
    _job(projet, "20260101-000100-000000-vame", "run_vame.py", "ok\n")
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    resume = [d.value for d in at.dataframe if "Points reconstitués" in d.value.columns]
    assert resume and resume[0].loc[0, "Session"] == "S1"


def test_bilan_qualite_en_tableau(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    projet = _projet(tmp_path)
    cols = pd.MultiIndex.from_product([["DLC"], ["nose", "tail_base"], ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    sortie = projet / "data" / "dlc-output" / "S1"
    sortie.mkdir(parents=True)
    valeurs = np.ones((20, 6))
    valeurs[:5, 0] = np.nan                     # nose absent sur 5 frames
    pd.DataFrame(valeurs, columns=cols).to_hdf(sortie / "S1DLC_resnet.h5", key="df")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    at.toggle(key="nettoyage_inspection").set_value(True).run()
    assert not at.exception, at.exception
    bilan = next(d.value for d in at.dataframe if "Verdict" in d.value.columns)
    assert bilan.loc[0, "Session"] == "S1" and bilan.loc[0, "Au moins un point (%)"] == 100.0
    # nose absent sur 5 images sur 20 : 75 % d'images complètes. Le verdict
    # suit cette part-là, pas la couverture à 100 %, et nomme le point en cause.
    assert bilan.loc[0, "Tous les points (%)"] == 75.0
    assert bilan.loc[0, "Verdict"].startswith("✓  bon") and "limité par nose" in bilan.loc[0, "Verdict"]
    par_kp = next(d.value for d in at.dataframe if "S1" in d.value.columns)
    assert par_kp.loc["nose", "S1"] == 75.0
