"""Choix de l'emplacement des projets depuis la page Projet.

Le champ « Emplacement » accepte une racine (dossier contenant des projets)
ou un projet directement, mémorise les emplacements récents, et refuse un
chemin relatif.

Ce dernier point est un garde-fou concret, pas théorique : `D:\\EthoFlow\\
projects` n'est un chemin absolu que sous Windows. Sous macOS et Linux,
pathlib y voit un unique composant portant littéralement ce nom, donc un
dossier RELATIF créé là où l'app a été lancée. Un projet a réellement
atterri dans un dossier au nom absurde avant ce garde-fou.

Isolation : `lib.project.PREFS_PATH` ET `lib.project.DEFAULT_PROJECTS_ROOT`
sont monkeypatchés. Le second doit l'être sur `lib.project`, pas sur
`interactive` : le nom y est lié par valeur à l'import.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")


def _isoler(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Deux emplacements distincts, chacun avec un projet. Retourne (a, b)."""
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "prefs.yaml")
    monkeypatch.setattr(P, "DEFAULT_PROJECTS_ROOT", tmp_path / "defaut")
    a = tmp_path / "disque_interne"
    b = tmp_path / "disque_externe"
    (a / "projet-a" / "data").mkdir(parents=True)
    (b / "projet-b" / "data").mkdir(parents=True)
    P.save_prefs({"projects_root": str(a)})
    return a, b


def test_champ_prerempli_sur_la_racine_courante(tmp_path, monkeypatch):
    a, _ = _isoler(tmp_path, monkeypatch)
    at = AppTest.from_file(APP_PY)
    at.run()
    assert not at.exception, at.exception
    assert at.text_input(key="emplacement_saisi").value == str(a)
    assert Path(at.session_state["current_project_path"]).name == "projet-a"


def test_changer_d_emplacement_change_les_projets_listes(tmp_path, monkeypatch):
    _, b = _isoler(tmp_path, monkeypatch)
    at = AppTest.from_file(APP_PY)
    at.run()
    at.text_input(key="emplacement_saisi").set_value(str(b)).run()
    assert not at.exception, at.exception
    assert Path(at.session_state["current_project_path"]).name == "projet-b"
    # L'emplacement devient le défaut, donc survit au redémarrage.
    assert P.projects_root() == b


def test_chemin_de_projet_colle_ouvre_le_projet(tmp_path, monkeypatch):
    """Un projet sur disque externe n'a aucune raison de vivre sous la racine
    configurée : coller son chemin doit l'ouvrir et lister ses voisins."""
    a, _ = _isoler(tmp_path, monkeypatch)
    # Un voisin, pour vérifier que la racine effective est bien le parent.
    (a / "projet-voisin" / "data").mkdir(parents=True)

    at = AppTest.from_file(APP_PY)
    at.run()
    at.text_input(key="emplacement_saisi").set_value(str(a / "projet-a")).run()

    assert not at.exception, at.exception
    assert Path(at.session_state["current_project_path"]).name == "projet-a"
    # La racine retenue est le parent : le sélecteur propose les voisins.
    assert "projet-voisin" in at.selectbox(key="projet_selector").options
    assert P.projects_root() == a


def test_chemin_relatif_refuse_et_aucun_dossier_cree(tmp_path, monkeypatch):
    """Le cas qui a réellement produit un dossier fantôme : un chemin Windows
    saisi sous macOS. Il doit être refusé, et surtout ne rien créer."""
    _isoler(tmp_path, monkeypatch)
    fantome = Path.cwd() / "D:\\EthoFlow\\projects"
    assert not fantome.exists(), "pré-condition : pas de dossier fantôme"

    at = AppTest.from_file(APP_PY)
    at.run()
    at.text_input(key="emplacement_saisi").set_value("D:\\EthoFlow\\projects").run()

    assert not at.exception, at.exception
    assert any("absolu" in e.value for e in at.error), [e.value for e in at.error]
    assert not fantome.exists(), "un chemin relatif ne doit créer aucun dossier"


def test_chemin_inexistant_signale(tmp_path, monkeypatch):
    _isoler(tmp_path, monkeypatch)
    at = AppTest.from_file(APP_PY)
    at.run()
    at.text_input(key="emplacement_saisi").set_value(str(tmp_path / "nulle_part")).run()
    assert not at.exception, at.exception
    assert any("existe pas" in e.value for e in at.error)


def test_emplacements_recents_memorises(tmp_path, monkeypatch):
    a, b = _isoler(tmp_path, monkeypatch)
    at = AppTest.from_file(APP_PY)
    at.run()
    at.text_input(key="emplacement_saisi").set_value(str(b)).run()
    recents = P.recent_roots()
    assert b in recents and a in recents
    assert recents[0] == b, "le plus récent en tête"


def test_bascule_par_bouton_recent_ne_leve_pas(tmp_path, monkeypatch):
    """Écrire dans la clé d'un widget déjà instancié lève
    `StreamlitAPIException` ; les bascules passent donc par une clé tampon
    appliquée au run suivant. Ce test garde ce détour."""
    a, b = _isoler(tmp_path, monkeypatch)
    at = AppTest.from_file(APP_PY)
    at.run()
    at.text_input(key="emplacement_saisi").set_value(str(b)).run()

    boutons = {btn.key: btn for btn in at.button if btn.key.startswith("recent_")}
    assert boutons, "l'emplacement précédent doit apparaître dans les récents"
    next(iter(boutons.values())).click().run()

    assert not at.exception, at.exception
    assert at.text_input(key="emplacement_saisi").value == str(a)


def test_bouton_parcourir_applique_le_dossier_choisi(tmp_path, monkeypatch):
    """Le sélecteur natif renvoie un chemin : il doit atterrir dans le champ
    et devenir la racine effective."""
    from lib import reveal
    a, b = _isoler(tmp_path, monkeypatch)
    monkeypatch.setattr(reveal, "choisir_dossier",
                        lambda **kwargs: (b, ""))

    at = AppTest.from_file(APP_PY)
    at.run()
    boutons = {btn.key: btn for btn in at.button}
    assert "btn_parcourir" in boutons, list(boutons)
    boutons["btn_parcourir"].click().run()

    assert not at.exception, at.exception
    assert at.text_input(key="emplacement_saisi").value == str(b)
    assert P.projects_root() == b


def test_bouton_parcourir_annule_ne_change_rien(tmp_path, monkeypatch):
    """Annuler la fenêtre est un choix, pas une erreur : ni changement, ni
    message affiché."""
    from lib import reveal
    a, _ = _isoler(tmp_path, monkeypatch)
    monkeypatch.setattr(reveal, "choisir_dossier", lambda **kwargs: (None, ""))

    at = AppTest.from_file(APP_PY)
    at.run()
    {btn.key: btn for btn in at.button}["btn_parcourir"].click().run()

    assert not at.exception, at.exception
    assert at.text_input(key="emplacement_saisi").value == str(a)
    # Assertion ciblée : la page affiche par ailleurs des avertissements
    # légitimes (modèle DLC non configuré, par exemple). On vérifie qu'aucun
    # ne vient du sélecteur, pas qu'il n'y en a aucun.
    du_selecteur = [
        w.value for w in at.warning
        if "sélecteur" in w.value or "à la main" in w.value
    ]
    assert not du_selecteur, du_selecteur


def test_bouton_parcourir_indisponible_le_dit(tmp_path, monkeypatch):
    """Un Linux sans zenity ni kdialog doit renvoyer vers la saisie."""
    from lib import reveal
    a, _ = _isoler(tmp_path, monkeypatch)
    monkeypatch.setattr(
        reveal, "choisir_dossier",
        lambda **kwargs: (None, "Aucun sélecteur — saisis le chemin à la main."),
    )

    at = AppTest.from_file(APP_PY)
    at.run()
    {btn.key: btn for btn in at.button}["btn_parcourir"].click().run()

    assert not at.exception, at.exception
    assert any("à la main" in w.value for w in at.warning)
    assert at.text_input(key="emplacement_saisi").value == str(a)
