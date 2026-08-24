"""Vérification de bout en bout de la page Visualisations via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme
`tests/test_app_analyses.py` / `tests/test_app_vame.py`) pour ne jamais
toucher `Path.home()` réel ni `DEFAULT_PROJECTS_ROOT`
(`D:\\EthoFlow\\projects`, un nom de dossier littéral et relatif sur ce
runner macOS).

Quatre points, en écho au brief de la Task 22 :
1. Les trois onglets (Bande de motifs / Manifold / Dendrogramme) rendent.
2. `--labels` est pré-rempli depuis `motif_labels.csv` s'il existe, vide
   sinon (avec le rappel que sans lui les figures affichent `motif_0`...).
3. L'avertissement `--pool-all-sessions` (lenteur + blocage documenté au
   Troubleshooting du README) est présent quand l'option est cochée.
4. Un projet sans segmentation VAME dit qu'il ne peut rien rendre, plutôt
   que d'offrir des formulaires qui échoueraient à l'exécution.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def _projet(tmp_path: Path, *, segmente: bool = True) -> Path:
    p = tmp_path / "projects" / "test-visu"
    for sub in ("raw", "cropped", "dlc-output", "vame", "results"):
        (p / "data" / sub).mkdir(parents=True)
    (p / "configs").mkdir(parents=True)
    (p / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump({"kind": "single"}, sort_keys=False), encoding="utf-8",
    )
    (p / "data" / "vame" / "config.yaml").write_text(
        yaml.safe_dump(
            {"n_clusters": 15, "segmentation_algorithms": ["hmm"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if segmente:
        algo_dir = p / "data" / "vame" / "results" / "S1" / "VAME" / "hmm-15"
        algo_dir.mkdir(parents=True)
        (algo_dir / "15_hmm_label_S1.npy").write_bytes(b"\x00")
    return p


def _ecrire_motif_labels(projet: Path) -> Path:
    import lib.motif_labels as ML
    import pandas as pd
    df = pd.DataFrame([{
        "motif_id": "0", "label": "grooming", "category": "Grooming",
        "confidence": "", "qc_inspected_sessions": "", "notes": "",
        "usage_pct": "12.3", "video": "",
    }], columns=ML.COLUMNS)
    ML.save(projet, df)
    return ML.path(projet)


def _lancer_sur_projet(tmp_path: Path, monkeypatch, projet: Path) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": str(projet.parent), "models_root": str(tmp_path / "models")})
    (tmp_path / "models").mkdir(exist_ok=True)
    at = AppTest.from_file(APP_PY)
    at.session_state["current_project_path"] = str(projet)
    at.run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "nav_visualisations" in boutons, list(boutons)
    boutons["nav_visualisations"].click().run()
    assert not at.exception, at.exception
    return at


def _widget(collection, cle: str):
    trouve = {w.key: w for w in collection}
    assert cle in trouve, list(trouve)
    return trouve[cle]


# ============================================================
# 1. Les trois onglets rendent
# ============================================================

def test_trois_onglets_rendent_sans_exception(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    onglets = {t.proto.label: t for t in at.tabs}
    assert set(onglets) == {"Bande de motifs", "Manifold", "Dendrogramme"}, list(onglets)

    # Les trois formulaires sont bien construits (le bouton de lancement de
    # chaque script apparaît une fois AppTest a exécuté tout le script,
    # tabs y compris).
    boutons = {b.key: b for b in at.button}
    assert "btn_visu_motif_gif" in boutons
    assert "btn_visu_manifold" in boutons
    assert "btn_visu_dendro" in boutons


# ============================================================
# 2. --labels pré-rempli depuis motif_labels.csv
# ============================================================

def test_labels_pre_rempli_quand_le_csv_existe(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    _ecrire_motif_labels(projet)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    for cle in ("visu_motif_labels", "visu_manifold_labels", "visu_dendro_labels"):
        champ = _widget(at.text_input, cle)
        assert champ.value.endswith("motif_labels.csv"), (cle, champ.value)


def test_labels_vide_quand_le_csv_absent(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    for cle in ("visu_motif_labels", "visu_manifold_labels", "visu_dendro_labels"):
        champ = _widget(at.text_input, cle)
        assert champ.value == "", (cle, champ.value)

    # Et le rappel doit être visible quelque part sur la page.
    textes = " ".join(c.value for c in at.caption)
    assert "motif_0" in textes


# ============================================================
# 3. Avertissement --pool-all-sessions
# ============================================================

def test_avertissement_pool_all_sessions_apparait_quand_coche(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    case = _widget(at.checkbox, "visu_manifold_pool")
    assert case.value is False
    assert not at.warning   # pas encore coché : pas d'avertissement

    case.set_value(True).run()
    assert not at.exception, at.exception

    avertissements = " ".join(w.value for w in at.warning)
    assert "pool-all-sessions" in avertissements.lower() or "Troubleshooting" in avertissements
    assert "bloqu" in avertissements.lower() or "Troubleshooting" in avertissements


# ============================================================
# 4. Projet non segmenté : message plutôt que des formulaires cassés
# ============================================================

def test_projet_sans_segmentation_narrete_pas_doffrir_des_rendus(tmp_path, monkeypatch):
    projet = _projet(tmp_path, segmente=False)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    # Aucun des trois boutons de lancement ne doit apparaître : rien à
    # segmenter n'a de sens à rendre.
    boutons = {b.key for b in at.button}
    assert "btn_visu_motif_gif" not in boutons
    assert "btn_visu_manifold" not in boutons
    assert "btn_visu_dendro" not in boutons

    textes = " ".join(i.value for i in at.info)
    assert "segmentation" in textes.lower()
