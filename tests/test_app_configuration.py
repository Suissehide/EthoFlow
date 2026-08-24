"""Vérification de bout en bout de la page Configuration via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme
`tests/test_app_projet.py`) — jamais de dépendance à `Path.home()` réel ni
à `DEFAULT_PROJECTS_ROOT`/`DEFAULT_MODELS_ROOT` (`D:\\EthoFlow\\...`, des
chemins Windows littéraux et relatifs sur ce runner macOS).

Les sondes d'environnement sont monkeypatchées via `lib.envcheck.probe_all`
(jamais de vrai `conda run` dans la suite pytest — voir `test_envcheck.py`
pour les tests unitaires de la sonde elle-même, et le rapport de la Task 23
pour la vérification manuelle sur les trois environnements réels).
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from lib import envcheck as EC
from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def _lancer(tmp_path: Path, monkeypatch) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    at = AppTest.from_file(APP_PY)
    at.run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "nav_config" in boutons, list(boutons)
    boutons["nav_config"].click().run()
    assert not at.exception, at.exception
    return at


def test_page_se_charge_sans_projet_ouvert(tmp_path, monkeypatch):
    at = _lancer(tmp_path, monkeypatch)
    assert not at.exception, at.exception
    assert any("Configuration" in t.value for t in at.title)


def test_racine_absente_est_signalee(tmp_path, monkeypatch):
    """Sur cette machine, les défauts Windows n'existent pas : la page doit
    le montrer plutôt que l'ignorer silencieusement."""
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({
        "projects_root": str(tmp_path / "introuvable-projets"),
        "models_root": str(tmp_path / "introuvable-modeles"),
    })
    at = _lancer(tmp_path, monkeypatch)
    textes = "\n".join(m.value for m in at.markdown)
    assert "introuvable" in textes


def test_racine_existante_est_confirmee(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    projets = tmp_path / "projets-ok"
    modeles = tmp_path / "modeles-ok"
    projets.mkdir()
    modeles.mkdir()
    P.save_prefs({"projects_root": str(projets), "models_root": str(modeles)})
    at = _lancer(tmp_path, monkeypatch)
    textes = "\n".join(m.value for m in at.markdown)
    assert str(projets) in textes
    assert str(modeles) in textes
    assert "introuvable" not in textes


def test_enregistrer_les_racines_persiste_les_prefs(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    at = _lancer(tmp_path, monkeypatch)

    nouvelle_racine_projets = str(tmp_path / "nouvelle-racine-projets")
    nouvelle_racine_modeles = str(tmp_path / "nouvelle-racine-modeles")

    inputs = {t.key: t for t in at.text_input}
    assert "config_projects_root" in inputs, list(inputs)
    assert "config_models_root" in inputs, list(inputs)
    inputs["config_projects_root"].set_value(nouvelle_racine_projets)
    inputs["config_models_root"].set_value(nouvelle_racine_modeles)

    boutons_form = [b for b in at.button if b.label == "Enregistrer"]
    assert boutons_form, [b.label for b in at.button]
    boutons_form[0].click().run()
    assert not at.exception, at.exception

    prefs = P.load_prefs()
    assert prefs["projects_root"] == nouvelle_racine_projets
    assert prefs["models_root"] == nouvelle_racine_modeles


def test_verification_environnements_affiche_ok_et_echec(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")

    resultats = {
        "ethoflow": EC.ProbeResult(env="ethoflow", ok=True, output=""),
        "dlc": EC.ProbeResult(env="dlc", ok=True, output="CUDA_DISPONIBLE=False", cuda=False),
        "vame": EC.ProbeResult(
            env="vame", ok=False,
            output="ModuleNotFoundError: No module named 'umap'",
        ),
    }
    monkeypatch.setattr(EC, "probe_all", lambda **kw: resultats)

    at = _lancer(tmp_path, monkeypatch)
    boutons = {b.key: b for b in at.button}
    assert "config_btn_verif_env" in boutons, list(boutons)
    boutons["config_btn_verif_env"].click().run()
    assert not at.exception, at.exception

    textes = "\n".join(m.value for m in at.markdown)
    assert "`ethoflow`" in textes
    assert "`dlc`" in textes
    assert "`vame`" in textes

    # CUDA indisponible pour dlc -> avertissement explicite (heures vs minutes).
    avertissements = "\n".join(w.value for w in at.warning)
    assert "CPU" in avertissements

    # L'échec de vame doit exposer le détail (ModuleNotFoundError) quelque part.
    corps = "\n".join(c.value for c in at.code)
    assert "umap" in corps


def test_projet_courant_affiche_si_ouvert(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    projet = tmp_path / "projects" / "proj-config-test"
    (projet / "data").mkdir(parents=True)
    P.save_prefs({
        "projects_root": str(projet.parent),
        "models_root": str(tmp_path / "models"),
        "last_project": str(projet),
    })
    at = _lancer(tmp_path, monkeypatch)
    textes = "\n".join(m.value for m in at.markdown)
    assert "Projet courant" in textes
    assert str(projet) in textes
