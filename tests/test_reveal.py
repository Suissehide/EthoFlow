"""Ouverture du dossier d'un projet dans l'explorateur du système.

Le point délicat n'est pas d'appeler la bonne commande, c'est de ne jamais
faire tomber la page qui le demande, et de ne pas afficher une erreur quand
il n'y en a pas — `explorer.exe` renvoie un code non nul même quand il a
correctement ouvert sa fenêtre.

La branche Windows est testée par simulation : cette machine est un Mac, et
un test qui se contenterait de se sauter sur les autres plateformes ne
protégerait rien.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lib import reveal

REPO = Path(__file__).resolve().parent.parent


def test_pas_de_streamlit_meme_transitivement():
    """`lib/reveal.py` doit s'importer sans Streamlit — c'est ce qui permet
    de le tester, et la contrainte vaut pour tout `lib/` sauf `config.py`.

    Vérifié dans un process neuf avec un bloqueur d'import : un simple
    `hasattr(module, "st")` ne dirait rien d'une dépendance transitive.
    """
    code = (
        "import sys\n"
        "class Bloqueur:\n"
        "    def find_module(self, nom, chemin=None):\n"
        "        if nom == 'streamlit' or nom.startswith('streamlit.'):\n"
        "            raise ImportError('streamlit interdit ici')\n"
        "sys.meta_path.insert(0, Bloqueur())\n"
        "sys.path.insert(0, 'streamlit_app')\n"
        "import lib.reveal\n"
        "assert not any(m == 'streamlit' or m.startswith('streamlit.')\n"
        "               for m in sys.modules), 'streamlit chargé'\n"
        "print('ok')\n"
    )
    resultat = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO,
        capture_output=True, text=True, timeout=60,
    )
    assert resultat.returncode == 0, resultat.stderr
    assert "ok" in resultat.stdout


@pytest.mark.parametrize(
    "plateforme, attendu",
    [("darwin", "open"), ("windows", "explorer"), ("autre", "xdg-open")],
)
def test_commande_par_plateforme(monkeypatch, plateforme, attendu):
    # On remplace `_plateforme`, jamais `sys.platform` : pytest lit lui
    # aussi cette variable, et la patcher globalement le fait planter.
    monkeypatch.setattr(reveal, "_plateforme", lambda: plateforme)
    assert reveal._commande(Path("/un/dossier"))[0] == attendu


def test_chemin_inexistant_refuse_sans_lever(tmp_path):
    ok, message = reveal.ouvrir_dans_explorateur(tmp_path / "nulle_part")
    assert ok is False
    assert "existe pas" in message


def test_un_fichier_ouvre_son_dossier_parent(tmp_path, monkeypatch):
    """Pointer un fichier doit révéler son dossier, pas tenter de l'ouvrir."""
    fichier = tmp_path / "notes.txt"
    fichier.write_text("x", encoding="utf-8")
    vus: list[list[str]] = []

    def _faux_run(commande, **kwargs):
        vus.append(commande)
        return subprocess.CompletedProcess(commande, 0, "", "")

    monkeypatch.setattr(reveal.subprocess, "run", _faux_run)
    ok, _ = reveal.ouvrir_dans_explorateur(fichier)
    assert ok is True
    assert vus[0][-1] == str(tmp_path), "c'est le parent qui doit être ouvert"


def test_windows_code_retour_non_nul_reste_un_succes(tmp_path, monkeypatch):
    """`explorer.exe` renvoie 1 même après avoir ouvert la fenêtre. Traiter
    ce code comme un échec afficherait une erreur à chaque clic sous
    Windows."""
    monkeypatch.setattr(reveal, "_plateforme", lambda: "windows")
    monkeypatch.setattr(
        reveal.subprocess, "run",
        lambda c, **k: subprocess.CompletedProcess(c, 1, "", ""),
    )
    ok, message = reveal.ouvrir_dans_explorateur(tmp_path)
    assert ok is True, "un code 1 sous Windows n'est pas un échec"
    assert "Explorateur" in message


def test_posix_code_retour_non_nul_est_un_echec(tmp_path, monkeypatch):
    """Sur les autres plateformes, en revanche, un code non nul est réel."""
    monkeypatch.setattr(reveal, "_plateforme", lambda: "autre")
    monkeypatch.setattr(
        reveal.subprocess, "run",
        lambda c, **k: subprocess.CompletedProcess(c, 3, "", "pas d'affichage"),
    )
    ok, message = reveal.ouvrir_dans_explorateur(tmp_path)
    assert ok is False
    assert "pas d'affichage" in message


def test_commande_absente_ne_leve_pas(tmp_path, monkeypatch):
    """Un Linux sans xdg-open ne doit pas faire tomber la page."""
    def _absente(commande, **kwargs):
        raise FileNotFoundError(commande[0])

    monkeypatch.setattr(reveal.subprocess, "run", _absente)
    ok, message = reveal.ouvrir_dans_explorateur(tmp_path)
    assert ok is False
    assert "introuvable" in message


def test_timeout_ne_leve_pas(tmp_path, monkeypatch):
    def _lent(commande, **kwargs):
        raise subprocess.TimeoutExpired(commande, reveal.TIMEOUT_S)

    monkeypatch.setattr(reveal.subprocess, "run", _lent)
    ok, message = reveal.ouvrir_dans_explorateur(tmp_path)
    assert ok is False
    assert "répondu" in message


def test_exception_inattendue_ne_leve_pas(tmp_path, monkeypatch):
    """Un échec d'ouverture est une gêne, pas une raison de faire tomber la
    page — le garde-fou attrape même l'imprévu."""
    def _boum(commande, **kwargs):
        raise RuntimeError("imprévu")

    monkeypatch.setattr(reveal.subprocess, "run", _boum)
    ok, message = reveal.ouvrir_dans_explorateur(tmp_path)
    assert ok is False
    assert "imprévu" in message


# ------------------------------------------------------------------ intégration

def test_bouton_present_dans_la_barre_laterale_avec_un_projet(tmp_path, monkeypatch):
    """Le bouton n'a de sens qu'avec un projet ouvert, et doit transmettre
    le chemin de CE projet."""
    from streamlit.testing.v1 import AppTest
    from lib import project as P

    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "prefs.yaml")
    monkeypatch.setattr(P, "DEFAULT_PROJECTS_ROOT", tmp_path / "defaut")
    racine = tmp_path / "projets"
    projet = racine / "mon-projet"
    (projet / "data").mkdir(parents=True)
    P.save_prefs({"projects_root": str(racine)})

    appels: list[Path] = []
    monkeypatch.setattr(
        reveal, "ouvrir_dans_explorateur",
        lambda chemin: (appels.append(Path(chemin)), (True, "ouvert"))[1],
    )

    at = AppTest.from_file(str(REPO / "streamlit_app" / "app.py"))
    at.run()
    assert not at.exception, at.exception

    boutons = {b.key: b for b in at.button}
    assert "btn_ouvrir_dossier" in boutons, list(boutons)
    aide = boutons["btn_ouvrir_dossier"].help or ""
    # Bouton icône : le libellé est vide (le glyphe est posé en CSS), donc
    # tout ce qui informe l'utilisateur passe par l'infobulle — le nom de
    # l'explorateur, le chemin visé, et le fait que le dossier s'ouvre côté
    # serveur (en usage LAN, l'utilisateur distant ne verrait rien).
    assert not boutons["btn_ouvrir_dossier"].label.strip()
    assert reveal.nom_explorateur() in aide
    assert projet.name in aide
    assert "machine" in aide

    boutons["btn_ouvrir_dossier"].click().run()
    assert not at.exception, at.exception
    assert appels, "le clic doit appeler l'ouverture"
    assert appels[-1].resolve() == projet.resolve()


def test_pas_de_bouton_sans_projet_ouvert(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from lib import project as P

    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "prefs.yaml")
    # Racine vide : aucun projet ne peut être ouvert automatiquement.
    vide = tmp_path / "vide"
    vide.mkdir()
    monkeypatch.setattr(P, "DEFAULT_PROJECTS_ROOT", vide)
    P.save_prefs({"projects_root": str(vide)})

    at = AppTest.from_file(str(REPO / "streamlit_app" / "app.py"))
    at.run()
    assert not at.exception, at.exception
    assert "btn_ouvrir_dossier" not in {b.key for b in at.button}


# --------------------------------------------------- sélecteur de dossier

def test_dialogue_annule_reste_silencieux(monkeypatch):
    """`osascript` sort en code 1 avec « User canceled » quand on ferme la
    fenêtre. Une annulation est un choix, pas une erreur : rien à afficher."""
    monkeypatch.setattr(reveal, "_plateforme", lambda: "darwin")
    monkeypatch.setattr(
        reveal.subprocess, "run",
        lambda c, **k: subprocess.CompletedProcess(c, 1, "", "User canceled."),
    )
    chemin, message = reveal.choisir_dossier()
    assert chemin is None
    assert message == "", "une annulation ne doit produire aucun message"


def test_dialogue_retourne_le_dossier_choisi(monkeypatch):
    monkeypatch.setattr(reveal, "_plateforme", lambda: "darwin")
    monkeypatch.setattr(
        reveal.subprocess, "run",
        lambda c, **k: subprocess.CompletedProcess(c, 0, "/tmp/choisi\n", ""),
    )
    chemin, message = reveal.choisir_dossier()
    assert chemin == Path("/tmp/choisi"), "le retour est nettoyé du saut de ligne"
    assert message == ""


def test_dialogue_macos_demande_un_chemin_posix(monkeypatch, tmp_path):
    """Sans `POSIX path of`, `choose folder` renvoie un alias
    « Macintosh HD:Users:… » inutilisable comme chemin."""
    monkeypatch.setattr(reveal, "_plateforme", lambda: "darwin")
    commande = reveal._commande_dialogue("Titre", tmp_path)
    assert commande[0] == "osascript"
    assert "POSIX path of" in commande[2]
    assert str(tmp_path) in commande[2], "le dossier de départ est transmis"


def test_dialogue_indisponible_sur_linux_nu(monkeypatch):
    """Ni zenity ni kdialog : on le dit et on renvoie vers la saisie."""
    monkeypatch.setattr(reveal, "_plateforme", lambda: "autre")
    monkeypatch.setattr(reveal.shutil, "which", lambda outil: None)
    chemin, message = reveal.choisir_dossier()
    assert chemin is None
    assert "à la main" in message


def test_dialogue_prefere_zenity_puis_kdialog(monkeypatch, tmp_path):
    monkeypatch.setattr(reveal, "_plateforme", lambda: "autre")
    monkeypatch.setattr(reveal.shutil, "which", lambda o: "/usr/bin/" + o)
    assert reveal._commande_dialogue("T", tmp_path)[0] == "zenity"

    monkeypatch.setattr(
        reveal.shutil, "which",
        lambda o: "/usr/bin/kdialog" if o == "kdialog" else None,
    )
    assert reveal._commande_dialogue("T", tmp_path)[0] == "kdialog"


def test_dialogue_timeout_ne_leve_pas(monkeypatch):
    """Une fenêtre laissée ouverte ne doit pas figer l'app indéfiniment."""
    monkeypatch.setattr(reveal, "_plateforme", lambda: "darwin")

    def _sans_reponse(commande, **kwargs):
        raise subprocess.TimeoutExpired(commande, reveal.TIMEOUT_DIALOGUE_S)

    monkeypatch.setattr(reveal.subprocess, "run", _sans_reponse)
    chemin, message = reveal.choisir_dossier()
    assert chemin is None
    assert "sans réponse" in message
