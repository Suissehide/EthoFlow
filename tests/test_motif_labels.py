import pandas as pd
import yaml

from lib import motif_labels as ML


def _csv_factice(vame_project, lignes=3):
    contenu = "motif_id;label;category;confidence;qc_inspected_sessions;notes;usage_pct;video\n"
    for i in range(lignes):
        contenu += f"{i};;;;;;{10.0 - i:.2f};results/community_videos/motif_{i}.mp4\n"
    (vame_project / "motif_labels.csv").write_text(contenu, encoding="utf-8-sig")


def test_colonnes_viennent_des_scripts():
    """Une copie divergerait au premier changement de run_vame.py."""
    import run_vame
    assert ML.COLUMNS == run_vame.MOTIF_LABELS_COLUMNS
    assert ML.categories() == run_vame.ETHOGRAM_CATEGORIES
    assert "Transitions" in ML.categories()
    assert len(ML.categories()) == 8


def test_absent_retourne_none(project, vame_project):
    assert not ML.exists(project)
    assert ML.load(project) is None


def test_chargement(project, vame_project):
    _csv_factice(vame_project)
    df = ML.load(project)
    assert list(df.columns) == ML.COLUMNS
    assert len(df) == 3
    assert df.loc[0, "usage_pct"] == "10.00"


def test_ecriture_conserve_separateur_et_encodage(project, vame_project):
    _csv_factice(vame_project)
    ML.set_fields(project, 0, label="grooming_face", category="Grooming")
    brut = (vame_project / "motif_labels.csv").read_bytes()
    assert brut.startswith(b"\xef\xbb\xbf")            # BOM utf-8-sig
    assert b";" in brut.split(b"\n")[0]
    assert b"," not in brut.split(b"\n")[0]
    assert ML.load(project).loc[0, "label"] == "grooming_face"


def test_colonne_utilisateur_preservee(project, vame_project):
    """Quelqu'un ajoute une colonne dans Excel : elle ne doit pas disparaître."""
    (vame_project / "motif_labels.csv").write_text(
        "motif_id;label;category;confidence;qc_inspected_sessions;notes;"
        "usage_pct;video;observateur\n"
        "0;;;;;;10.00;v0.mp4;Leo\n",
        encoding="utf-8-sig",
    )
    ML.set_fields(project, 0, label="walking")
    df = ML.load(project)
    assert df.loc[0, "observateur"] == "Leo"
    assert df.loc[0, "label"] == "walking"
    assert list(df.columns)[-1] == "observateur"


def test_set_fields_ne_touche_pas_les_autres_lignes(project, vame_project):
    _csv_factice(vame_project)
    ML.set_fields(project, 1, category="Locomotion")
    df = ML.load(project)
    assert df.loc[1, "category"] == "Locomotion"
    assert df.loc[0, "category"] == ""
    assert df.loc[2, "usage_pct"] == "8.00"


def test_reprise_depuis_ancien_yaml(project, vame_project):
    _csv_factice(vame_project)
    ancien = vame_project / "analysis" / "motif_labels_hmm-15.yaml"
    ancien.parent.mkdir(parents=True, exist_ok=True)
    ancien.write_text(yaml.safe_dump({0: "grooming", 2: "walking"}))
    assert ML.migrate_from_yaml(project, ancien) == 2
    df = ML.load(project)
    assert df.loc[0, "label"] == "grooming"
    assert df.loc[2, "label"] == "walking"
    assert df.loc[1, "label"] == ""


def test_detection_des_anciens_yaml(project, vame_project):
    (vame_project / "analysis").mkdir(parents=True, exist_ok=True)
    (vame_project / "analysis" / "motif_labels_hmm-15.yaml").write_text("{}")
    assert [p.name for p in ML.legacy_yaml_files(project)] == \
        ["motif_labels_hmm-15.yaml"]


def test_video_path_resolu_depuis_la_colonne(project, vame_project):
    _csv_factice(vame_project)
    clip = vame_project / "results" / "community_videos" / "motif_0.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00")
    df = ML.load(project)
    assert ML.video_path(project, df.loc[0]) == clip
    assert ML.video_path(project, df.loc[1]) is None    # fichier absent
