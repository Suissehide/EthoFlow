"""Vérification de bout en bout de la page Projet via AppTest.

Isolation : `lib.project.PREFS_PATH` est monkeypatché comme dans
`tests/test_project.py::test_prefs_round_trip` — jamais de dépendance à
`Path.home()` réel ni à `DEFAULT_PROJECTS_ROOT` (`D:\\EthoFlow\\projects`,
un chemin Windows qui n'est pas absolu du tout sur ce runner macOS/Linux :
un harnais qui ne l'isole pas crée un dossier littéral `D:\\EthoFlow\\
projects` dans le dépôt — piège déjà tombé sur une revue précédente).

`lib.runner._argv` est monkeypatché pour lancer `create_project.py`
directement via `sys.executable` plutôt que par un `conda run -n ethoflow`
imbriqué (on tourne déjà dans cet env) : le job réel devient quasi
instantané, donc ce test reste rapide malgré un vrai script + un vrai
sous-process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import pipeline as PL
from lib import project as P
from lib import runner as R

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def _argv_sans_conda(cmd: PL.Command) -> list[str]:
    """`conda run -n <env> python <script> <args>` -> `[sys.executable, script, *args]`."""
    argv = PL.to_argv(cmd)
    return [sys.executable] + argv[5:]


def _attendre_job_termine(project: Path, timeout: float = 20.0) -> None:
    import time
    fin = time.time() + timeout
    while time.time() < fin:
        job = R.current(project)
        if job and job.state != "running":
            return
        time.sleep(0.05)
    raise AssertionError("job de création toujours 'running' après timeout")


def test_creer_supprimer_retaper_ne_ressuscite_pas_le_projet(tmp_path, monkeypatch):
    """Séquence exacte du Critical remonté par la revue : créer X (bannière
    de succès), le supprimer via le flux guardé de la page, retaper le même
    nom X. Avant le ruling R10.6 : la bannière "créé avec succès" revenait,
    le bouton "Créer le projet" disparaissait pour de bon, et le dossier
    revenait comme un squelette vide sur disque."""
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    monkeypatch.setattr(R, "_argv", _argv_sans_conda)

    projects_root = tmp_path / "projects"
    models_root = tmp_path / "models"
    projects_root.mkdir()
    models_root.mkdir()
    P.save_prefs({"projects_root": str(projects_root), "models_root": str(models_root)})

    nom = "test-zombie"
    cible = projects_root / nom

    # ---- 1) Créer X, attendre le succès ----
    at = AppTest.from_file(APP_PY)
    at.run()
    assert not at.exception, at.exception
    list(at.text_input)[0].set_value(nom).run()
    boutons = {b.key: b for b in at.button}
    assert "btn_creer" in boutons, list(boutons)
    boutons["btn_creer"].click().run()
    assert not at.exception, at.exception

    _attendre_job_termine(cible.parent)
    # `_section_ouverture` auto-sélectionne le projet fraîchement listable
    # via son propre `st.rerun()` interne : quelques `.run()` "à froid"
    # pour la laisser se stabiliser d'abord. Sans ça, ré-affirmer la
    # valeur du champ "Nom du projet" dans la MÊME requête que ce rerun-là
    # ne "prend" pas côté AppTest (pas de frontend persistant comme dans
    # un vrai navigateur : la resoumission d'un widget qu'un rerun interne
    # traverse avant même de l'atteindre se perd) — artefact du harnais de
    # test, sans équivalent réel puisqu'un navigateur garde la valeur
    # tapée côté client quoi qu'il arrive côté serveur.
    at.run()
    at.run()
    nom_field = [t for t in at.text_input if t.label == "Nom du projet"][0]
    nom_field.set_value(nom).run()
    assert not at.exception, at.exception
    assert any("créé avec succès" in s.value for s in at.success), \
        [s.value for s in at.success]
    assert (cible / "configs" / "pipeline_config.yaml").is_file()

    # ---- 2) Supprimer X via le flux guardé (Supprimer -> confirmer) ----
    boutons = {b.key: b for b in at.button}
    assert "btn_supprimer_projet" in boutons, list(boutons)
    boutons["btn_supprimer_projet"].click().run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "btn_confirmer_suppression" in boutons, list(boutons)
    boutons["btn_confirmer_suppression"].click().run()
    assert not at.exception, at.exception

    assert not cible.exists(), "le rmtree n'a pas eu lieu"

    # ---- 3) Retaper le même nom ----
    text_inputs = list(at.text_input)
    nom_field = [t for t in text_inputs if t.label == "Nom du projet"][0]
    nom_field.set_value(nom).run()
    assert not at.exception, at.exception

    textes_success = [s.value for s in at.success]
    assert not any("créé avec succès" in t for t in textes_success), (
        f"bannière fantôme réapparue pour un projet supprimé : {textes_success}"
    )
    boutons = {b.key: b for b in at.button}
    assert "btn_creer" in boutons, (
        f"le bouton de création a disparu pour de bon après suppression : {list(boutons)}"
    )
    # Rien n'a dû être recréé sur disque par le simple fait de retaper le nom.
    assert not cible.exists(), "le dossier a été ressuscité rien qu'en retapant le nom"
