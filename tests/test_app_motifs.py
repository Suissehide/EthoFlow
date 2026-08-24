"""Vérification de bout en bout de la page Motifs via AppTest.

Isolation : `lib.project.PREFS_PATH` monkeypatché (comme
`tests/test_app_nettoyage.py` / `tests/test_app_vame.py`) pour ne jamais
toucher `Path.home()` réel ni `DEFAULT_PROJECTS_ROOT`
(`D:\\EthoFlow\\projects`, un nom de dossier littéral et relatif sur ce
runner macOS).

Le point central de ces tests, en écho à Task 6 (`lib/motif_labels.py`) :
la page ne doit jamais perdre une colonne ajoutée à la main dans le CSV,
ni le séparateur `;`/l'encodage `utf-8-sig`, en sauvegardant un label
depuis l'onglet « Par motif ».
"""
from __future__ import annotations

from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from lib import project as P

REPO = Path(__file__).resolve().parent.parent
APP_PY = str(REPO / "streamlit_app" / "app.py")

CSV_HEADER = (
    "motif_id;label;category;confidence;qc_inspected_sessions;notes;"
    "usage_pct;video"
)


def _projet(tmp_path: Path) -> Path:
    p = tmp_path / "projects" / "test-motifs"
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


def _ecrire_csv(projet: Path, lignes: list[str], extra_header: str = "") -> Path:
    vame = projet / "data" / "vame"
    csv_path = vame / "motif_labels.csv"
    header = CSV_HEADER + extra_header
    csv_path.write_text(
        header + "\n" + "\n".join(lignes) + "\n", encoding="utf-8-sig",
    )
    return csv_path


def _lancer_sur_projet(tmp_path: Path, monkeypatch, projet: Path) -> AppTest:
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "app_prefs.yaml")
    P.save_prefs({
        "projects_root": str(projet.parent),
        "models_root": str(tmp_path / "models"),
    })
    (tmp_path / "models").mkdir(exist_ok=True)
    at = AppTest.from_file(APP_PY)
    at.session_state["current_project_path"] = str(projet)
    at.run()
    assert not at.exception, at.exception
    boutons = {b.key: b for b in at.button}
    assert "nav_motifs" in boutons, list(boutons)
    boutons["nav_motifs"].click().run()
    assert not at.exception, at.exception
    return at


# ============================================================
# CSV absent : deux boutons, rien d'autre
# ============================================================

def test_sans_csv_seulement_les_deux_boutons_de_generation(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    boutons = {b.key: b for b in at.button}
    assert "btn_motifs_videos" in boutons
    assert "btn_motifs_labels" in boutons

    # Rien de la partie édition ne doit apparaître : ni sélecteur de motif,
    # ni champ label/catégorie, ni tableau.
    assert list(at.select_slider) == []
    assert list(at.selectbox) == []
    assert not any(
        (t.label or "").startswith("Label") for t in at.text_input
    )


# ============================================================
# CSV présent : tri par usage décroissant
# ============================================================

def test_motifs_tries_par_usage_decroissant(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    _ecrire_csv(projet, [
        "0;;;;;;5.00;",
        "1;;;;;;42.50;",
        "2;;;;;;10.00;",
    ])
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    sliders = list(at.select_slider)
    assert len(sliders) == 1
    slider = sliders[0]
    # `.options` renvoie déjà le texte formaté (format_func appliqué) : la
    # première option (motif d'usage max, 42.50 %) doit être motif_id=1,
    # la dernière (5.00 %) motif_id=0.
    assert "motif 1" in slider.options[0]
    assert "motif 0" in slider.options[-1]


def test_categorie_offre_exactement_les_8_valeurs_fermees(tmp_path, monkeypatch):
    from lib.config import categories

    projet = _projet(tmp_path)
    _ecrire_csv(projet, ["0;;;;;;5.00;"])
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    cats_widget = [s for s in at.selectbox if s.label == "Catégorie"]
    assert len(cats_widget) == 1
    assert cats_widget[0].options == categories()
    assert len(cats_widget[0].options) == 8


# ============================================================
# CRITIQUE : une `category` hors liste fermée ne doit jamais être perdue.
#
# README.md:517 dit d'écrire `artifact` dans `category` — qui n'est pas une
# des 8 valeurs ETHOGRAM. Avant le correctif, le selectbox affichait cette
# valeur comme "non catégorisé" (options=cats uniquement, valeur hors
# options -> None), et un simple clic sur Enregistrer réécrivait
# `category=""`, effaçant l'annotation sans le moindre avertissement.
# ============================================================

def test_categorie_hors_liste_visible_et_pas_perdue_a_lecture(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    _ecrire_csv(projet, ["0;grooming_face;artifact;;;;5.00;"])
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    cats_widget = [s for s in at.selectbox if s.label == "Catégorie"][0]
    # La valeur du CSV doit rester affichée — pas "non catégorisé" (None).
    assert cats_widget.value == "artifact"
    assert "artifact" in cats_widget.options

    # Un avertissement explicite doit signaler que c'est hors liste.
    avertissements = " ".join(w.value for w in at.warning)
    assert "artifact" in avertissements
    assert "hors" in avertissements.lower()


def test_categorie_hors_liste_survit_a_une_edition_du_label_seul(tmp_path, monkeypatch):
    """Le cas exact du bug critique : le chercheur ne touche QUE le label
    (via le vocabulaire suggéré ou en tapant), clique Enregistrer, et la
    `category` hors liste déjà présente dans le fichier doit survivre
    intacte — aucun clic ne doit l'effacer sans que l'utilisateur ne l'ait
    demandé."""
    projet = _projet(tmp_path)
    csv_path = _ecrire_csv(projet, ["0;;artifact;;;;5.00;"])
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    champ_label = [t for t in at.text_input if t.label == "Label (texte libre)"][0]
    champ_label.set_value("bruit_de_tracking").run()
    assert not at.exception, at.exception

    bouton_save = [b for b in at.button if b.key.startswith("motifs_save_")][0]
    bouton_save.click().run()
    assert not at.exception, at.exception

    import pandas as pd
    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False,
                      encoding="utf-8-sig")
    assert df.loc[df["motif_id"] == "0", "label"].iloc[0] == "bruit_de_tracking"
    assert df.loc[df["motif_id"] == "0", "category"].iloc[0] == "artifact"


def test_categorie_hors_liste_typo_ou_vocabulaire_labo_aussi_protegee(tmp_path, monkeypatch):
    """Pas seulement `artifact` : n'importe quelle valeur hors liste (typo,
    vocabulaire propre à un labo) doit être traitée pareil."""
    projet = _projet(tmp_path)
    csv_path = _ecrire_csv(projet, ["0;;Locomotton;;;;5.00;"])  # typo volontaire
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    cats_widget = [s for s in at.selectbox if s.label == "Catégorie"][0]
    assert cats_widget.value == "Locomotton"

    bouton_save = [b for b in at.button if b.key.startswith("motifs_save_")][0]
    bouton_save.click().run()
    assert not at.exception, at.exception

    import pandas as pd
    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False,
                      encoding="utf-8-sig")
    assert df.loc[df["motif_id"] == "0", "category"].iloc[0] == "Locomotton"


# ============================================================
# Édition d'un label : écrite sur disque
# ============================================================

def test_editer_un_label_ecrit_sur_disque(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    csv_path = _ecrire_csv(projet, ["0;;;;;;5.00;"])
    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    champ_label = [t for t in at.text_input if t.label == "Label (texte libre)"][0]
    champ_label.set_value("grooming_face").run()
    assert not at.exception, at.exception

    bouton_save = [b for b in at.button if b.key.startswith("motifs_save_")][0]
    bouton_save.click().run()
    assert not at.exception, at.exception

    contenu = csv_path.read_text(encoding="utf-8-sig")
    assert "grooming_face" in contenu


# ============================================================
# Intégrité des données : colonne ajoutée à la main préservée
# ============================================================

def test_colonne_ajoutee_a_la_main_survit_a_une_edition(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    vame = projet / "data" / "vame"
    csv_path = vame / "motif_labels.csv"
    contenu_initial = (
        CSV_HEADER + ";observateur\n"
        "0;;;;;;18.42;results/community_videos/motif_0.mp4;Léo\n"
        "1;walking;Locomotion;high;;;12.07;results/community_videos/motif_1.mp4;Léo\n"
    )
    csv_path.write_text(contenu_initial, encoding="utf-8-sig")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    # Le tri par usage_pct décroissant place motif_0 (18.42%) en premier —
    # c'est donc lui qui est affiché par défaut par le select_slider.
    champ_label = [t for t in at.text_input if t.label == "Label (texte libre)"][0]
    champ_label.set_value("grooming_face").run()
    assert not at.exception, at.exception
    bouton_save = [b for b in at.button if b.key.startswith("motifs_save_")][0]
    bouton_save.click().run()
    assert not at.exception, at.exception

    brut = csv_path.read_bytes()
    assert brut.startswith(b"\xef\xbb\xbf")  # BOM utf-8-sig préservé
    texte = brut.decode("utf-8-sig")
    lignes = [l for l in texte.splitlines() if l]
    assert lignes[0].split(";")[-1] == "observateur"
    assert ";" in lignes[0] and "," not in lignes[0]

    # La colonne ajoutée à la main et son contenu survivent, pour les deux
    # lignes — y compris celle qui n'a pas été éditée.
    import pandas as pd
    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False,
                      encoding="utf-8-sig")
    assert list(df.columns)[-1] == "observateur"
    assert (df["observateur"] == "Léo").all()
    assert df.loc[df["motif_id"] == "0", "label"].iloc[0] == "grooming_face"
    # La ligne du motif 1, non éditée, reste strictement inchangée.
    assert df.loc[df["motif_id"] == "1", "label"].iloc[0] == "walking"
    assert df.loc[df["motif_id"] == "1", "category"].iloc[0] == "Locomotion"


# ============================================================
# Onglet "Tableau" : sauvegarde en ordre `motif_id`, pas en ordre d'usage
#
# `_tab_table` affiche `df` déjà trié par `usage_pct` décroissant (cohérent
# avec l'onglet « Par motif »). Si on sauvegardait cet ordre d'affichage
# tel quel, on réordonnerait pour de bon le fichier de `run_vame` à chaque
# édition depuis le Tableau. `st.data_editor` n'est pas pilotable depuis
# AppTest (pas d'accesseur dédié, contrairement à `st.dataframe`) : on
# monkeypatche donc `st.data_editor` pour simuler le retour d'une édition,
# et on vérifie ce que `_tab_table` écrit sur disque.
# ============================================================

def test_sauvegarde_tableau_conserve_ordre_motif_id(tmp_path, monkeypatch):
    import views.motifs as motifs_module

    projet = _projet(tmp_path)
    csv_path = _ecrire_csv(projet, [
        "0;;;;;;5.00;",
        "1;;;;;;42.50;",
        "2;label2;;;;;10.00;",
    ])

    def faux_data_editor(data, *args, **kwargs):
        # Simule l'utilisateur éditant `label` pour le motif 1, sans
        # changer l'ordre des lignes que `st.data_editor` lui présente
        # (ordre d'affichage = usage décroissant : 1, 2, 0).
        edited = data.copy()
        edited.loc[edited["motif_id"] == "1", "label"] = "edite_depuis_tableau"
        return edited

    monkeypatch.setattr(motifs_module.st, "data_editor", faux_data_editor)

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)
    assert not at.exception, at.exception

    import pandas as pd
    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False,
                      encoding="utf-8-sig")
    # Le fichier reste dans l'ordre `motif_id` écrit par `run_vame`
    # (0, 1, 2), pas dans l'ordre d'affichage par usage décroissant
    # (1, 2, 0) qu'aurait produit une sauvegarde de `edited` telle quelle.
    assert list(df["motif_id"]) == ["0", "1", "2"]
    assert df.loc[df["motif_id"] == "1", "label"].iloc[0] == "edite_depuis_tableau"
    # Les autres lignes, non éditées, restent intactes.
    assert df.loc[df["motif_id"] == "2", "label"].iloc[0] == "label2"


# ============================================================
# Reprise de l'ancien format YAML
# ============================================================

def test_reprise_ancien_yaml_proposee_et_ecrit_apres_confirmation(tmp_path, monkeypatch):
    projet = _projet(tmp_path)
    csv_path = _ecrire_csv(projet, ["0;;;;;;5.00;", "1;;;;;;3.00;"])
    ancien = projet / "data" / "vame" / "analysis" / "motif_labels_hmm-3.yaml"
    ancien.parent.mkdir(parents=True, exist_ok=True)
    ancien.write_text(yaml.safe_dump({0: "grooming"}), encoding="utf-8")

    at = _lancer_sur_projet(tmp_path, monkeypatch, projet)

    # Rien n'est écrit tant que le bouton d'import n'a pas été cliqué.
    avant = csv_path.read_text(encoding="utf-8-sig")
    assert "grooming" not in avant

    boutons_import = [b for b in at.button if b.key.startswith("btn_import_")]
    assert len(boutons_import) == 1
    boutons_import[0].click().run()
    assert not at.exception, at.exception

    apres = csv_path.read_text(encoding="utf-8-sig")
    assert "grooming" in apres
