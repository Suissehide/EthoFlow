"""`watch_training.py` : lire `learning_stats.csv` pendant qu'il s'écrit.

Le point dur n'est pas le tracé, c'est la lecture. DLC réécrit le CSV en
entier (`open("w")`) à chaque epoch : une lecture peut tomber sur un
fichier vide, tronqué au milieu d'une ligne, ou sans en-tête. Le viewer
doit alors garder le dernier état valide au lieu d'afficher une page
cassée — c'est ce que ces tests verrouillent.
"""
from __future__ import annotations

from pathlib import Path

import watch_training as W

ENTETE = "step,losses/train.total_loss,losses/eval.total_loss,metrics/test.rmse\n"


def _train_dir(racine: Path, shuffle: str = "sourisJun3-trainset95shuffle1") -> Path:
    d = racine / "dlc-models-pytorch" / "iteration-0" / shuffle / "train"
    d.mkdir(parents=True)
    return d


def _ecrire(train: Path, lignes: str, nom: str = "learning_stats.csv") -> Path:
    chemin = train / nom
    chemin.write_text(ENTETE + lignes, encoding="utf-8")
    return chemin


# ----------------------------------------------------------------------
# Localisation
# ----------------------------------------------------------------------
def test_trouver_stats_ramene_pose_et_detecteur(tmp_path):
    train = _train_dir(tmp_path)
    _ecrire(train, "0,0.1,0.12,40\n")
    _ecrire(train, "0,0.3,0.31,50\n", nom="learning_stats_detector.csv")

    noms = {p.name for p in W.trouver_stats(tmp_path)}
    assert noms == {"learning_stats.csv", "learning_stats_detector.csv"}


def test_trouver_stats_vide_si_rien_entraine(tmp_path):
    (tmp_path / "dlc-models-pytorch").mkdir()
    assert W.trouver_stats(tmp_path) == []


def test_epochs_prevues_lit_le_pytorch_config(tmp_path):
    train = _train_dir(tmp_path)
    csv_path = _ecrire(train, "0,0.1,0.12,40\n")
    (train / "pytorch_config.yaml").write_text(
        "train_settings:\n  batch_size: 8\n  epochs: 50\n", encoding="utf-8")

    assert W.epochs_prevues(csv_path) == 50


def test_epochs_prevues_none_si_pas_de_config(tmp_path):
    train = _train_dir(tmp_path)
    assert W.epochs_prevues(_ecrire(train, "0,0.1,0.12,40\n")) is None


# ----------------------------------------------------------------------
# Lecture du CSV
# ----------------------------------------------------------------------
def test_lire_stats_parse_steps_et_colonnes(tmp_path):
    train = _train_dir(tmp_path)
    chemin = _ecrire(train, "0,0.10,0.12,40.5\n1,0.05,0.07,22.1\n")

    lu = W.lire_stats(chemin)
    assert lu["steps"] == [0, 1]
    assert lu["valeurs"]["losses/train.total_loss"] == [0.10, 0.05]
    assert lu["valeurs"]["metrics/test.rmse"] == [40.5, 22.1]


def test_lire_stats_trous_et_nan_deviennent_none(tmp_path):
    """DLC n'évalue pas à tous les epochs : les cellules vides sont des trous."""
    train = _train_dir(tmp_path)
    chemin = _ecrire(train, "0,0.10,0.12,40.5\n1,0.05,,nan\n")

    lu = W.lire_stats(chemin)
    assert lu["valeurs"]["losses/eval.total_loss"] == [0.12, None]
    assert lu["valeurs"]["metrics/test.rmse"] == [40.5, None]


def test_lire_stats_jette_les_colonnes_entierement_vides(tmp_path):
    train = _train_dir(tmp_path)
    chemin = _ecrire(train, "0,0.10,,\n1,0.05,,\n")

    assert set(W.lire_stats(chemin)["valeurs"]) == {"losses/train.total_loss"}


def test_lire_stats_ignore_la_ligne_coupee_en_cours_decriture(tmp_path):
    """Une dernière ligne sans `step` exploitable ne doit pas tuer la lecture."""
    train = _train_dir(tmp_path)
    chemin = _ecrire(train, "0,0.10,0.12,40.5\n1,0.05,0.07,22.1\n,0.0")

    lu = W.lire_stats(chemin)
    assert lu["steps"] == [0, 1]


def test_lire_stats_none_sur_fichier_entete_seule(tmp_path):
    train = _train_dir(tmp_path)
    assert W.lire_stats(_ecrire(train, "")) is None


def test_lire_stats_none_sans_colonne_step(tmp_path):
    train = _train_dir(tmp_path)
    chemin = train / "learning_stats.csv"
    chemin.write_text("epoch,loss\n0,0.1\n", encoding="utf-8")
    assert W.lire_stats(chemin) is None


def test_lire_stats_none_sur_fichier_vide(tmp_path):
    """L'instant exact où DLC a tronqué le fichier avant de le réécrire."""
    train = _train_dir(tmp_path)
    chemin = train / "learning_stats.csv"
    chemin.write_text("", encoding="utf-8")
    assert W.lire_stats(chemin) is None


# ----------------------------------------------------------------------
# Payload servi à la page
# ----------------------------------------------------------------------
def test_payload_decrit_le_shuffle_et_les_epochs(tmp_path):
    train = _train_dir(tmp_path)
    _ecrire(train, "0,0.10,0.12,40.5\n1,0.05,0.07,22.1\n")
    (train / "pytorch_config.yaml").write_text(
        "train_settings:\n  epochs: 50\n", encoding="utf-8")

    source = W.Moniteur(tmp_path, None).payload()["sources"][0]
    assert source["shuffle"] == "sourisJun3-trainset95shuffle1"
    assert source["epochs_prevues"] == 50
    assert source["detecteur"] is False
    assert source["actif"] is True
    assert source["partiel"] is False
    assert source["steps"] == [0, 1]


def test_payload_garde_le_dernier_etat_valide_si_le_csv_est_tronque(tmp_path):
    train = _train_dir(tmp_path)
    chemin = _ecrire(train, "0,0.10,0.12,40.5\n1,0.05,0.07,22.1\n")
    moniteur = W.Moniteur(tmp_path, None)
    assert moniteur.payload()["sources"][0]["steps"] == [0, 1]

    chemin.write_text("", encoding="utf-8")  # DLC vient de tronquer
    source = moniteur.payload()["sources"][0]
    assert source["partiel"] is True
    assert source["steps"] == [0, 1]  # on continue d'afficher les courbes


def test_payload_vide_tant_que_rien_na_ete_ecrit(tmp_path):
    """Le viewer se lance avant `02_train.py` : la page attend, sans planter."""
    (tmp_path / "dlc-models-pytorch").mkdir()
    assert W.Moniteur(tmp_path, None).payload()["sources"] == []


def test_payload_met_le_detecteur_apres_le_modele_de_pose(tmp_path):
    train = _train_dir(tmp_path)
    _ecrire(train, "0,0.3,0.31,50\n", nom="learning_stats_detector.csv")
    _ecrire(train, "0,0.1,0.12,40\n")

    sources = W.Moniteur(tmp_path, None).payload()["sources"]
    assert [s["detecteur"] for s in sources] == [False, True]


def test_payload_suit_un_csv_donne_explicitement(tmp_path):
    """`--stats <chemin>` : hors de toute arborescence DLC."""
    chemin = tmp_path / "learning_stats.csv"
    chemin.write_text(ENTETE + "0,0.10,0.12,40.5\n", encoding="utf-8")

    sources = W.Moniteur(None, [chemin]).payload()["sources"]
    assert len(sources) == 1
    assert sources[0]["steps"] == [0]
