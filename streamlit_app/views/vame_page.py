"""Page VAME — étape 7 du pipeline, présentée en stepper.

Six sous-étapes tournent dans cet ordre, dans l'env conda `vame` :
`setup`, `align`, `trainset`, `train`, `evaluate`, `segment`. Contrairement
au modèle DLC (réutilisable d'un projet à l'autre), VAME s'entraîne une
fois **par projet** : le VAE apprend la structure posturale de CES
animaux, pas d'un référentiel général.

**Honnêteté du stepper — le point sensible de cette page.**

`lib.vame.stage_status(projet)` détecte l'avancement en lisant le
disque, pas en interrogeant un job. Deux pièges à ne pas transformer en
faux positifs affichés à l'écran :

- `train: True` signifie qu'un `.pkl` existe sous `model/best_model/` —
  c'est-à-dire qu'**un modèle utilisable existe**, pas que l'entraînement
  est *terminé*. vame-py 0.13.0 écrit ce fichier dès la première
  amélioration de loss acceptée après la fin de l'annealing KL, qui peut
  survenir bien avant la fin du run. Afficher « Entraînement terminé »
  ici inviterait à segmenter un modèle à moitié entraîné et à perdre une
  nuit de GPU à le découvrir le lendemain — cette page ne le dit jamais.
- `align: True` signifie qu'**au moins une** session a produit un
  `*_processed.nc` (`any()`, pas `all()`) — pas que toutes les sessions
  du projet sont alignées. Le texte le dit explicitement.

`stage_status` n'a pas de clé `evaluate` : rien sur disque ne signale de
façon stable que cette sous-étape a tourné (elle ne produit que des
figures, à des chemins non prévisibles). Plutôt que d'inventer une
détection, cette page l'affiche comme **non vérifiable** en permanence,
et se contente de garder son bouton actif dès qu'un modèle existe.

Aucune commande n'est exécutée ici : `lib.pipeline.vame_stage` construit,
`views._job.bouton_lancer` lance via `lib.runner`.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import lib.pipeline as PL
import lib.vame as V
from lib.config import require_project
from lib.icons import lucide_title
from views import _job

# ============================================================
# Constantes du stepper
# ============================================================

_ORDRE = ["setup", "align", "trainset", "train", "evaluate", "segment"]

_TITRES = {
    "setup": "1. Setup",
    "align": "2. Align",
    "trainset": "3. Trainset",
    "train": "4. Train",
    "evaluate": "5. Evaluate",
    "segment": "6. Segment",
}

_DUREES = {
    "setup": "quelques secondes",
    "align": "quelques minutes",
    "trainset": "quelques minutes",
    "train": "3 à 8 h sur GPU",
    "evaluate": "quelques minutes",
    "segment": "quelques minutes",
}

_ICONES = {
    "setup": "settings",
    "align": "scan-line",
    "trainset": "database",
    "train": "brain",
    "evaluate": "chart-column",
    "segment": "tags",
}


def _puce(fait: bool | None) -> str:
    if fait is True:
        return "✅"
    if fait is False:
        return "⬜"
    return "❔"  # état non détectable (evaluate)


def _entete_etape(cle: str, fait: bool | None) -> None:
    st.markdown(
        lucide_title(_ICONES[cle], f"{_puce(fait)} {_TITRES[cle]}"),
        unsafe_allow_html=True,
    )
    st.caption(f"Durée annoncée : {_DUREES[cle]}.")


# ============================================================
# Sous-étape 1 : setup
# ============================================================

def _section_setup(projet: Path, etat: dict) -> None:
    _entete_etape("setup", etat["setup"])
    if etat["setup"]:
        st.caption(
            "Un projet VAME existe déjà (`data/vame/config.yaml` présent). "
            "Relancer sans `--force` échoue plutôt que d'écraser."
        )
    else:
        st.caption(
            "Crée `data/vame/` à partir des paires (vidéo croppée, `.h5`) "
            "trouvées dans le projet."
        )

    pose_confidence = st.number_input(
        "`--pose-confidence`", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        key="vame_setup_pose_confidence",
        help="Seuil de confiance VAME — 0.6 aligné sur le pré-filtrage "
             "EthoFlow (le défaut VAME, 0.99, masque la majorité des "
             "points SuperAnimal).",
    )
    copie = st.radio(
        "Vidéos dans le projet VAME",
        ["Auto", "Copier (--copy-videos)", "Symlink (--no-copy-videos)"],
        index=0, horizontal=True, key="vame_setup_copie",
        help="Auto : copie sur Windows, symlink ailleurs (les symlinks "
             "demandent le mode développeur sur Windows).",
    )
    force = st.checkbox(
        "Écraser un projet VAME existant (`--force`)",
        value=False, key="vame_setup_force",
    )

    with st.expander("Chemins avancés (rarement utiles)", expanded=False):
        st.caption(
            "Par défaut dérivés du projet (`data/dlc-output/` et "
            "`data/cropped/`) — à ne changer que pour pointer vers une "
            "autre sortie DLC."
        )
        input_dir = st.text_input("`--input-dir`", value="", key="vame_setup_input_dir")
        cropped_dir = st.text_input("`--cropped-dir`", value="", key="vame_setup_cropped_dir")

    extra: list[str] = ["--pose-confidence", str(pose_confidence)]
    if copie.startswith("Copier"):
        extra.append("--copy-videos")
    elif copie.startswith("Symlink"):
        extra.append("--no-copy-videos")
    if force:
        extra.append("--force")
    if input_dir.strip():
        extra += ["--input-dir", input_dir.strip()]
    if cropped_dir.strip():
        extra += ["--cropped-dir", cropped_dir.strip()]

    cmd = PL.vame_stage(projet, "setup", extra=extra)
    _job.bouton_lancer(
        projet, "Lancer setup", cmd, cle="btn_vame_setup",
        type="secondary" if etat["setup"] else "primary",
    )


# ============================================================
# Sous-étape 2 : align
# ============================================================

def _section_align(projet: Path, etat: dict) -> None:
    _entete_etape("align", etat["align"])
    if etat["align"]:
        st.caption(
            "Au moins une session alignée (`data/processed/*_processed.nc` "
            "trouvé) — pas nécessairement toutes. Si des sessions ont été "
            "ajoutées au projet VAME depuis, relance `align` pour les couvrir."
        )
    else:
        st.caption("Alignement égocentrique des poses (+ nettoyage optionnel).")

    manque = not etat["setup"]
    cmd = PL.vame_stage(projet, "align")
    _job.bouton_lancer(
        projet, "Lancer align", cmd, cle="btn_vame_align",
        disabled=manque,
        help="Lance d'abord `setup` — aucun projet VAME détecté." if manque else None,
    )


# ============================================================
# Sous-étape 3 : trainset
# ============================================================

def _section_trainset(projet: Path, etat: dict) -> None:
    _entete_etape("trainset", etat["trainset"])
    if etat["trainset"]:
        st.caption("Jeu d'entraînement déjà construit (`data/train/train_seq.npy` présent).")
    else:
        st.caption("Assemble le jeu d'entraînement à partir des sessions alignées.")

    manque = not etat["align"]
    cmd = PL.vame_stage(projet, "trainset")
    _job.bouton_lancer(
        projet, "Lancer trainset", cmd, cle="btn_vame_trainset",
        disabled=manque,
        help="Lance d'abord `align` — aucune session alignée détectée." if manque else None,
    )


# ============================================================
# Sous-étape 4 : train
# ============================================================

def _section_train(projet: Path, etat: dict) -> None:
    _entete_etape("train", etat["train"])
    if etat["train"]:
        st.info(
            "**Un modèle existe** (`model/best_model/*.pkl` présent) — cela "
            "ne veut *pas* dire que l'entraînement est terminé. vame-py "
            "écrit ce fichier dès la première amélioration de loss acceptée "
            "après la fin de l'annealing KL, pas à la fin du run ni à "
            "convergence. Si un job tourne encore, il continue d'affiner ce "
            "modèle."
        )
    else:
        st.caption("Aucun modèle sauvegardé pour l'instant.")

    st.warning(
        "**Dure 3 à 8 h sur GPU.** Le job tourne en arrière-plan et son "
        "état vit sur disque (`.ethoflow/jobs/`), pas dans l'onglet du "
        "navigateur : tu peux fermer cet onglet, éteindre l'écran ou "
        "naviguer ailleurs dans l'app, l'entraînement continue et tu "
        "retrouveras son état en revenant sur cette page. Pas besoin de "
        "garder le navigateur ouvert toute la nuit par précaution."
    )

    with st.expander("Options avancées (dépannage)", expanded=False):
        no_cluster_loss = st.checkbox(
            "Désactiver `cluster_loss` (`--no-cluster-loss`)",
            value=False, key="vame_train_no_cluster_loss",
            help="À cocher seulement si l'entraînement plante au tout "
                 "premier epoch avec une erreur SVD (latents mal "
                 "conditionnés) — contourne cluster_loss() par un no-op.",
        )

    manque = not etat["trainset"]
    extra = ["--no-cluster-loss"] if no_cluster_loss else None
    cmd = PL.vame_stage(projet, "train", extra=extra)
    _job.bouton_lancer(
        projet, "Lancer train", cmd, cle="btn_vame_train",
        disabled=manque,
        help="Lance d'abord `trainset` — aucun jeu d'entraînement détecté." if manque else None,
    )


# ============================================================
# Sous-étape 5 : evaluate (état indétectable)
# ============================================================

def _section_evaluate(projet: Path, etat: dict) -> None:
    _entete_etape("evaluate", None)
    st.caption(
        "EthoFlow ne peut pas détecter si `evaluate` a déjà tourné : "
        "contrairement aux autres sous-étapes, elle ne laisse aucun "
        "fichier stable et prévisible à vérifier (seulement des figures "
        "de diagnostic, à des chemins qui varient). Relance-la quand tu "
        "veux revoir l'état du modèle avant de segmenter."
    )

    manque = not etat["train"]
    cmd = PL.vame_stage(projet, "evaluate")
    _job.bouton_lancer(
        projet, "Lancer evaluate", cmd, cle="btn_vame_evaluate",
        disabled=manque,
        help="Aucun modèle entraîné pour l'instant — lance `train` d'abord." if manque else None,
    )


# ============================================================
# Sous-étape 6 : segment (+ n_clusters)
# ============================================================

def _section_segment(projet: Path, etat: dict) -> None:
    _entete_etape("segment", etat["segment"])
    if etat["segment"]:
        st.caption(
            "Au moins une session segmentée (fichier de labels de motifs "
            "trouvé) — pas nécessairement toutes."
        )
    else:
        st.caption("Segmente les poses alignées en motifs comportementaux.")

    n_actuel = V.n_clusters(projet)
    st.caption(
        f"`n_clusters` actuel dans le config VAME : "
        f"**{n_actuel if n_actuel else 'non défini (VAME utilisera son défaut, 15)'}**."
    )

    csv_path = V.vame_project(projet) / "motif_labels.csv"
    if csv_path.exists():
        st.warning(
            "`motif_labels.csv` existe déjà pour ce projet — **il est "
            "unique par projet**, pas par valeur de `n_clusters`. "
            "Changer `n_clusters` puis régénérer ce CSV écrase le travail "
            "d'annotation déjà saisi. Sauvegarde-le d'abord si tu changes "
            "de granularité."
        )
        try:
            contenu = csv_path.read_bytes()
        except OSError:
            contenu = None
        if contenu is not None:
            st.download_button(
                "Sauvegarder motif_labels.csv avant de continuer",
                data=contenu, file_name="motif_labels_backup.csv",
                mime="text/csv", key="vame_segment_backup_csv",
            )

    nouveau_n = st.number_input(
        "`--n-clusters` pour cette segmentation",
        min_value=2, value=int(n_actuel) if n_actuel else 15, step=1,
        key="vame_segment_n_clusters",
    )
    st.caption(
        "Chaque valeur crée son propre dossier de résultats "
        "(`hmm-15` à côté de `hmm-25`, par exemple) — rien n'est écrasé "
        "pour la segmentation elle-même. Évaluer le modèle avant de "
        "segmenter est recommandé mais pas imposé par cette page."
    )

    manque = not etat["train"]
    cmd = PL.vame_stage(projet, "segment", n_clusters=int(nouveau_n))
    _job.bouton_lancer(
        projet, "Lancer segment", cmd, cle="btn_vame_segment",
        disabled=manque,
        help="Aucun modèle entraîné pour l'instant — lance `train` d'abord." if manque else None,
    )


# ============================================================
# Aperçu global (stepper)
# ============================================================

def _section_apercu(etat: dict) -> None:
    st.markdown(lucide_title("waypoints", "Avancement"), unsafe_allow_html=True)
    cols = st.columns(len(_ORDRE))
    for col, cle in zip(cols, _ORDRE):
        with col:
            fait = etat.get(cle)  # None pour "evaluate", absent de stage_status
            st.markdown(f"**{_puce(fait)} {_TITRES[cle]}**")
            st.caption(_DUREES[cle])


def render() -> None:
    projet = require_project()

    st.title("VAME")
    st.caption(
        "Étape 7 du pipeline : segmentation non supervisée des séquences "
        "de pose en motifs comportementaux. Contrairement au modèle DLC, "
        "VAME s'entraîne une fois **par projet** — le VAE apprend la "
        "structure posturale de ces animaux précis, il n'est pas "
        "réutilisable ailleurs."
    )

    etat = V.stage_status(projet)

    _section_apercu(etat)
    _job.panneau(projet)

    st.divider()
    _section_setup(projet, etat)
    st.divider()
    _section_align(projet, etat)
    st.divider()
    _section_trainset(projet, etat)
    st.divider()
    _section_train(projet, etat)
    st.divider()
    _section_evaluate(projet, etat)
    st.divider()
    _section_segment(projet, etat)
    st.divider()
    _job.historique(projet)
