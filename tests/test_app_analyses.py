"""Vérification de bout en bout de la page Analyses via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme
`tests/test_app_motifs.py` / `tests/test_app_vame.py`) pour ne jamais
toucher `Path.home()` réel ni `DEFAULT_PROJECTS_ROOT`
(`D:\\EthoFlow\\projects`, un nom de dossier littéral et relatif sur ce
runner macOS).

Trois points centraux, en écho au brief de la Task 17 :
1. Sans `analysis/`, l'onglet Résultats renvoie vers le lancement plutôt
   que de planter.
2. Un dossier `analysis/` peuplé de faux fichiers nommés selon les
   *vrais* patterns d'`analyze_vame.py` (pas les anciens noms de
   `views/results.py`, qui n'existent plus) doit apparaître groupé par
   axe de comparaison.
3. Les cases `--group-by` n'apparaissent qu'une fois les axes découverts
   — relus depuis le log du job `--list-columns`, jamais depuis
   `session_state` (elles doivent donc survivre à un `at.run()` qui
   simule un nouveau chargement de page).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")

SORTIE_LIST_COLUMNS = """
Colonnes exploitables comme axe de comparaison (2) :

  captopril                2 groupes : Captopril (8 sessions), Control (8 sessions)
  condition                2 groupes : MCCiECKO (8 sessions), MCCf/f (8 sessions)
"""


def _png_bytes() -> bytes:
    """PNG minimal mais valide — `st.image` refuse un simple faux-en-tête."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color="white").save(buf, format="PNG")
    return buf.getvalue()


def _projet(tmp_path: Path) -> Path:
    p = tmp_path / "projects" / "test-analyses"
    for sub in ("raw", "cropped", "dlc-output", "vame", "results"):
        (p / "data" / sub).mkdir(parents=True)
    (p / "configs").mkdir(parents=True)
    (p / "configs" / "pipeline_config.yaml").write_text(
        yaml.safe_dump({"kind": "single"}, sort_keys=False), encoding="utf-8",
    )
    (p / "data" / "vame" / "config.yaml").write_text(
        yaml.safe_dump(
            {"n_clusters": 3, "segmentation_algorithms": ["hmm"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return p


def _ecrire_job_list_columns_succeeded(projet: Path, sortie: str) -> None:
    """Fabrique le job/log qu'aurait laissé un vrai `--list-columns` réussi.

    Reproduit la forme lue par `_axes_disponibles` (voir
    `streamlit_app/views/analyses.py`) : un `.json` avec `script`,
    `argv`, `state="succeeded"`, et un `.log` avec la sortie du script.
    """
    jobs_dir = projet / ".ethoflow" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = "20260101-000000-000000-analyze_vame"
    job = {
        "job_id": job_id,
        "script": "analyze_vame.py",
        "env": "vame",
        "label": "Axes de comparaison disponibles",
        "argv": ["conda", "run", "-n", "vame", "python", "analyze_vame.py",
                 "--project-dir", str(projet), "--no-prompt", "--list-columns"],
        "started_at": "2026-01-01T00:00:00",
        "ended_at": "2026-01-01T00:00:05",
        "returncode": 0,
        "pid": None,
        "state": "succeeded",
        "cancel_requested": False,
        "owner_pid": None,
    }
    (jobs_dir / f"{job_id}.json").write_text(json.dumps(job), encoding="utf-8")
    (jobs_dir / f"{job_id}.log").write_text(sortie, encoding="utf-8")


def _lancer_sur_projet(tmp_path: Path, monkeypatch, projet: Path) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({"projects_root": str(projet.parent), "models_root": str(tmp_path / "models")})
    (tmp_path / "models").mkdir(exist_ok=True)
    at = AppTest.from_file(APP_PY)
    at.session_state["current_project_path"] = str(projet)
    at.run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "nav_analyses" in boutons, list(boutons)
    boutons["nav_analyses"].click().run()
    assert not at.exception, at.exception
    return at


# ============================================================
# Résultats : dossier analysis/ absent
# ============================================================

def test_sans_dossier_analysis_message_renvoie_vers_le_lancement(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    textes = [i.value for i in at.info]
    assert any("analysis" in t and "lancement" in t.lower() or "lance" in t.lower()
               for t in textes), textes
    # Aucune exception, aucun accès à un fichier inexistant.
    assert not at.exception, at.exception


# ============================================================
# Résultats : vrais noms de fichiers, groupés par axe
# ============================================================

def test_fichiers_reels_groupes_par_axe(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    analysis_dir = projet / "data" / "vame" / "analysis"
    analysis_dir.mkdir(parents=True)

    # Fichiers globaux (jamais liés à un axe).
    (analysis_dir / "motif_usage.csv").write_text("motif,frequency\n0,0.5\n", encoding="utf-8")
    (analysis_dir / "motif_usage_long.csv").write_text("session,motif,frequency\n", encoding="utf-8")
    (analysis_dir / "heatmap_usage.png").write_bytes(_png_bytes())
    (analysis_dir / "usage_by_category.csv").write_text("category,frequency\n", encoding="utf-8")

    # Fichiers par axe "captopril".
    (analysis_dir / "heatmap_usage_by_captopril.png").write_bytes(_png_bytes())
    (analysis_dir / "mean_by_captopril.png").write_bytes(_png_bytes())
    (analysis_dir / "boxplots_top_by_captopril.png").write_bytes(_png_bytes())
    (analysis_dir / "boxplots_by_category_by_captopril.png").write_bytes(_png_bytes())
    (analysis_dir / "stats_by_motif_captopril.csv").write_text("motif,p\n0,0.2\n", encoding="utf-8")

    # Fichiers par axe "condition" — un seul, pour vérifier la séparation.
    (analysis_dir / "mean_by_condition.png").write_bytes(_png_bytes())

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    assert not at.exception, at.exception

    textes_titres = [h.value for h in at.subheader] + [e.label for e in at.expander]
    assert any("Général" in t for t in textes_titres), textes_titres
    assert any("captopril" in t for t in textes_titres), textes_titres
    assert any("condition" in t for t in textes_titres), textes_titres

    # motif_usage.csv doit apparaître comme aperçu de dataframe (CSV global).
    assert any("motif_usage.csv" in m.value for m in at.markdown), \
        [m.value for m in at.markdown]


# ============================================================
# Lancer une analyse : cases --group-by après découverte des axes
# ============================================================

def test_pas_de_cases_group_by_sans_axes_decouverts(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    assert list(at.checkbox) == []
    boutons = {b.key: b for b in at.button}
    assert "btn_analyses_list_columns" in boutons


def test_cases_group_by_apparaissent_apres_decouverte(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    _ecrire_job_list_columns_succeeded(projet, SORTIE_LIST_COLUMNS)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    cases = {c.key: c for c in at.checkbox}
    assert "analyses_gb_captopril" in cases, list(cases)
    assert "analyses_gb_condition" in cases, list(cases)

    # Le libellé porte l'effectif par groupe (README §Étape 9).
    assert "8 sessions" in cases["analyses_gb_captopril"].label

    # Survit à un nouveau `.run()` (relu depuis le log, pas la session) :
    at.run()
    assert not at.exception, at.exception
    cases_apres = {c.key: c for c in at.checkbox}
    assert "analyses_gb_captopril" in cases_apres
