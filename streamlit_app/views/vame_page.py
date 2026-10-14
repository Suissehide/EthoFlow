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

import pandas as pd
import streamlit as st

import lib.pipeline as PL
import lib.vame as V
from lib.config import require_project
from views import _graphiques as G
from views import _job
from views import _widgets
from views._widgets import champ_chemin

# ============================================================
# Constantes du stepper
# ============================================================

_ORDRE = ["setup", "align", "trainset", "train", "evaluate", "segment"]

_TITRES = {
    "setup": "Créer le projet VAME",
    "align": "Aligner les poses",
    "trainset": "Préparer le jeu d'entraînement",
    "train": "Entraîner le modèle",
    "evaluate": "Évaluer le modèle",
    "segment": "Découper en motifs",
}

_DUREES = {
    "setup": "quelques secondes",
    "align": "quelques minutes",
    "trainset": "quelques minutes",
    "train": "3 à 8 h sur GPU",
    "evaluate": "quelques minutes",
    "segment": "quelques minutes",
}

# Les trois phases de la maquette, et les sous-étapes de chacune.
_PHASES = [
    ("Préparer", ("setup", "align", "trainset")),
    ("Entraîner", ("train", "evaluate")),
    ("Segmenter", ("segment",)),
]
_COULEURS = {"done": "#2f9567", "run": "#c27a12", "fail": "#c9473f",
             "todo": "#1c1b19", "wait": "#e4e2db"}


def _entete_etape(cle: str, fait: bool | None) -> None:
    """Titre d'une sous-étape + pastille d'état (fait / à faire / non vérifiable)."""
    etat = {True: ("fait", "#2f9567"), False: ("à faire", "#8a867e"),
            None: ("non vérifiable", "#8a867e")}[fait]
    st.markdown(
        f'<p class="ef-sous-etape">{_TITRES[cle]}'
        f'<span class="ef-puce" style="color:{etat[1]};border-color:{etat[1]}">{etat[0]}</span>'
        f'<span class="ef-muted">{_DUREES[cle]}</span></p>',
        unsafe_allow_html=True,
    )


def _etape_du_job(job) -> str | None:
    """Sous-étape VAME d'un job `run_vame.py` (lue dans sa ligne de commande)."""
    if job is None or job.script != "run_vame.py":
        return None
    return next((e for e in _ORDRE if e in job.argv), None)


def _etat_phases(projet: Path, etat: dict) -> list[tuple[str, str, str]]:
    """(nom, état, libellé) par phase. État : done/run/fail/todo/wait."""
    from lib import runner

    job = runner.current(projet)
    etape_job = _etape_du_job(job)
    faite = [etat["trainset"], etat["train"], etat["segment"]]
    sorties = []
    for k, (nom, etapes) in enumerate(_PHASES):
        if etape_job in etapes and job.state == "running":
            sorties.append((nom, "run", "En cours"))
        elif etape_job in etapes and job.state == "failed":
            sorties.append((nom, "fail", "Échec"))
        elif faite[k]:
            # Honnêteté : un .pkl prouve qu'un modèle existe, pas que
            # l'entraînement est terminé (voir le docstring du module).
            sorties.append((nom, "done", "Modèle disponible" if k == 1 else "Terminé"))
        elif k == 0 or faite[k - 1]:
            sorties.append((nom, "todo", "À faire"))
        else:
            sorties.append((nom, "wait", "En attente"))
    return sorties


def _cartes_phases(projet: Path, etat: dict) -> None:
    from html import escape

    from lib.sessions import list_sessions

    n = len(list_sessions(projet))
    descriptions = [
        "Crée le projet VAME, aligne les poses et prépare le jeu d'entraînement. "
        "Quelques minutes.",
        f"Apprend le modèle sur les {n} sessions. 3 à 8 h sur GPU.",
        "Attribue un motif à chaque image. Quelques minutes.",
    ]
    cartes = "".join(
        f'<div class="ef-card ef-phase-carte"><div class="ef-phase-etat">'
        f'<span class="ef-dot" style="background:{_COULEURS[e]}"></span>'
        f'<span class="ef-label">{escape(libelle)}</span></div>'
        f"<b>{escape(nom)}</b><span>{escape(d)}</span></div>"
        for (nom, e, libelle), d in zip(_etat_phases(projet, etat), descriptions))
    st.markdown(f'<div class="ef-phases">{cartes}</div>', unsafe_allow_html=True)


def _titre_phase(n: int, nom: str) -> None:
    st.markdown(f'<p class="ef-phase-titre">{n} · {nom}</p>', unsafe_allow_html=True)


# ============================================================
# Sous-étape 1 : setup
# ============================================================

def _section_setup(projet: Path, etat: dict) -> None:
    _entete_etape("setup", etat["setup"])
    if etat["setup"]:
        st.caption(
            "Le projet VAME existe déjà. Le recréer n'est possible qu'en "
            "cochant « Recréer le projet VAME »."
        )
    else:
        st.caption("Rassemble les poses nettoyées et leurs vidéos dans un projet VAME.")

    # Les quatre hyperparamètres que `run_vame.py setup` demande à l'invite.
    # Un job de l'app tourne sans terminal : ils sont réglés ici et la
    # commande part avec `--no-prompt`, sinon le premier `input()` du
    # script lèverait EOFError.
    col_n, col_w = st.columns(2)
    with col_n:
        n_clusters = st.number_input(
            "Nombre de motifs", min_value=2, max_value=200, value=15, step=1,
            key="vame_setup_n_clusters",
            help="Nombre de motifs à produire. C'est autant de clips à "
                 "nommer à l'étape Motifs — 15 se labellisent en ~1 h, 30 "
                 "en une demi-journée. Modifiable ensuite : chaque valeur "
                 "a son propre dossier de résultats.",
        )
    with col_w:
        time_window = st.number_input(
            "Fenêtre d'analyse (images)", min_value=5, max_value=300, value=30, step=5,
            key="vame_setup_time_window",
            help="Durée que le modèle regarde d'un coup. 30 images = 1 s à "
                 "30 images par seconde. Plus court : motifs brefs ; plus "
                 "long : les gestes courts se noient.",
        )
    col_e, col_c = st.columns(2)
    with col_e:
        max_epochs = st.number_input(
            "Durée maximale d'entraînement (cycles)", min_value=10, max_value=5000, value=500, step=50,
            key="vame_setup_max_epochs",
            help="500 (défaut) : 3 à 8 h sur GPU ; 100 pour un premier essai "
                 "de bout en bout. L'entraînement s'arrête aussi plus tôt "
                 "s'il ne progresse plus.",
        )
    with col_c:
        pose_confidence = st.number_input(
            "Confiance minimale des points", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            key="vame_setup_pose_confidence",
            help="0,6 comme au nettoyage. La valeur par défaut de VAME (0,99) "
                 "écarterait la plupart des points.",
        )
    copie = st.radio(
        "Vidéos dans le projet VAME",
        ["Auto", "Copier", "Lien, sans copie"],
        index=0, horizontal=True, key="vame_setup_copie",
        help="Auto : copie sous Windows, lien ailleurs (les liens demandent "
             "le mode développeur sous Windows).",
    )
    force = st.checkbox(
        "Recréer le projet VAME s'il existe déjà",
        value=False, key="vame_setup_force",
    )

    with st.expander("Réglages avancés", expanded=False):
        st.caption("Par défaut, les dossiers du projet. À changer seulement "
                   "pour utiliser d'autres poses.")
        input_dir = champ_chemin(
            "Dossier des poses", cle="vame_setup_input_dir",
            titre_dialogue="Dossier des .h5 pour VAME",
        )
        cropped_dir = champ_chemin(
            "Dossier des vidéos découpées", cle="vame_setup_cropped_dir",
            titre_dialogue="Dossier des vidéos croppées",
        )

    extra: list[str] = [
        "--n-clusters", str(int(n_clusters)),
        "--time-window", str(int(time_window)),
        "--max-epochs", str(int(max_epochs)),
        "--pose-confidence", str(pose_confidence),
        "--no-prompt",
    ]
    if copie.startswith("Copier"):
        extra.append("--copy-videos")
    elif copie.startswith("Lien"):
        extra.append("--no-copy-videos")
    if force:
        extra.append("--force")
    if input_dir.strip():
        extra += ["--input-dir", input_dir.strip()]
    if cropped_dir.strip():
        extra += ["--cropped-dir", cropped_dir.strip()]

    cmd = PL.vame_stage(projet, "setup", extra=extra)
    _job.bouton_lancer(
        projet, "Créer le projet VAME", cmd, cle="btn_vame_setup",
        type="secondary" if etat["setup"] else "primary",
    )
    _widgets.details_techniques(cmd)


# ============================================================
# Sous-étape 2 : align
# ============================================================

def _section_align(projet: Path, etat: dict) -> None:
    _entete_etape("align", etat["align"])
    if etat["align"]:
        st.caption(
            "Au moins une session alignée, pas forcément toutes. Si des "
            "sessions ont été ajoutées depuis, relance l'alignement."
        )
    else:
        st.caption("Ramène chaque pose dans le repère de l'animal (même "
                   "orientation, même centre).")

    manque = not etat["setup"]
    cmd = PL.vame_stage(projet, "align")
    _job.bouton_lancer(
        projet, "Aligner les poses", cmd, cle="btn_vame_align",
        disabled=manque,
        help="Crée d'abord le projet VAME." if manque else None,
    )


# ============================================================
# Sous-étape 3 : trainset
# ============================================================

def _section_trainset(projet: Path, etat: dict) -> None:
    _entete_etape("trainset", etat["trainset"])
    if etat["trainset"]:
        st.caption("Jeu d'entraînement déjà préparé.")
    else:
        st.caption("Assemble les sessions alignées en un jeu d'entraînement.")

    manque = not etat["align"]
    cmd = PL.vame_stage(projet, "trainset")
    _job.bouton_lancer(
        projet, "Préparer le jeu", cmd, cle="btn_vame_trainset",
        disabled=manque,
        help="Aligne d'abord les poses." if manque else None,
    )


# ============================================================
# Sous-étape 4 : train
# ============================================================

def _courbes(projet: Path) -> None:
    """Courbes d'entraînement lues dans `model/model_losses/`, réécrit par
    vame-py à chaque epoch : on suit un run de plusieurs heures en direct."""
    import altair as alt

    df = V.training_losses(projet)
    if df is None:
        return
    cfg = V.read_config(projet)
    n = int(df["epoch"].iloc[-1])
    meilleur = V.best_epoch(df)
    max_ep, conv = cfg.get("max_epochs"), cfg.get("model_convergence")

    c1, c2, c3 = st.columns(3)
    c1.metric("Epoch", f"{n} / {max_ep}" if max_ep else n)
    if meilleur:
        c2.metric("Meilleur modèle", f"epoch {meilleur}",
                  help=f"Test loss {df.loc[meilleur - 1, 'test']:.4g}. C'est le .pkl "
                       "de model/best_model/.")
        c3.metric("Sans amélioration", f"{n - meilleur} / {conv}" if conv else n - meilleur,
                  help="vame-py s'arrête quand ce compteur dépasse "
                       "`model_convergence` (config.yaml).")
    else:
        c2.metric("Meilleur modèle", "—",
                  help="Pas encore : vame-py ne sauvegarde qu'après l'annealing KL.")

    couches_fond = []
    if "kl_weight" in df:
        fin = df.loc[df["kl_weight"] > 0.99, "epoch"].min()
        fin = n if pd.isna(fin) else int(fin)
        bande = pd.DataFrame({"debut": [1], "fin": [fin], "texte": ["annealing KL"]})
        couches_fond = [
            alt.Chart(bande).mark_rect(color=G.NEUTRE, opacity=0.12)
            .encode(x="debut:Q", x2="fin:Q"),
            alt.Chart(bande).mark_text(align="left", baseline="top", dx=4, dy=4,
                                       color=G.NEUTRE, fontSize=11)
            .encode(x="debut:Q", y=alt.value(0), text="texte:N"),
        ]
    st.altair_chart(
        G.lignes(df, "epoch", {c: lib for c, lib in (("train", "Train"), ("test", "Test"))
                               if c in df},
                 titre_x="Epoch", titre_y="Loss", log=True, couches_fond=couches_fond,
                 couches_dessus=G.repere_vertical(meilleur, "meilleur modèle", "epoch")
                 if meilleur else None),
        width="stretch",
    )

    composantes = {"reconstruction": "Reconstruction (MSE)", "kl": "KL",
                   "kmeans": "K-means", "prediction": "Prédiction"}
    presentes = [c for c in composantes if c in df and df[c].abs().sum() > 0]
    if presentes:
        with st.expander("Composantes de la loss train", expanded=False):
            cols = st.columns(2)
            for i, c in enumerate(presentes):
                cols[i % 2].altair_chart(
                    G.petit_multiple(df, "epoch", c, composantes[c], titre_x="Epoch"),
                    width="stretch")
    with st.expander("Tableau des losses", expanded=False):
        st.dataframe(df, hide_index=True, width="stretch")


# Rafraîchie toutes les 30 s pendant un `train` : vame-py réécrit les
# losses à chaque epoch, il suffit de les relire.
_courbes_en_direct = st.fragment(run_every=30)(_courbes)


def _section_courbes(projet: Path) -> None:
    from lib import runner

    job = runner.current(projet)
    en_cours = (runner.is_running(projet) and job is not None
                and job.script == "run_vame.py" and "train" in job.argv)
    if en_cours:
        st.caption("Entraînement en cours : courbes rafraîchies toutes les 30 s.")
        _courbes_en_direct(projet)
    else:
        _courbes(projet)


def _section_train(projet: Path, etat: dict) -> None:
    _entete_etape("train", etat["train"])
    if etat["train"]:
        st.info(
            "**Un modèle existe**, ce qui ne veut *pas* dire que "
            "l'entraînement est fini : VAME l'enregistre dès sa première "
            "amélioration, pas à la fin du run ni à la convergence. Si "
            "l'entraînement tourne encore, il continue de l'améliorer."
        )
    else:
        st.caption("Aucun modèle sauvegardé pour l'instant.")

    _section_courbes(projet)

    st.warning(
        "**Dure 3 à 8 h sur GPU.** Tu peux fermer l'onglet ou changer de "
        "page : l'entraînement continue, et tu retrouves son avancement en "
        "revenant ici."
    )

    with st.expander("Réglages avancés", expanded=False):
        no_cluster_loss = st.checkbox(
            "Désactiver le regroupement pendant l'entraînement",
            value=False, key="vame_train_no_cluster_loss",
            help="Seulement si l'entraînement plante dès le premier cycle "
                 "avec une erreur « SVD ».",
        )

    manque = not etat["trainset"]
    extra = ["--no-cluster-loss"] if no_cluster_loss else None
    cmd = PL.vame_stage(projet, "train", extra=extra)
    _job.bouton_lancer(
        projet, "Lancer l'entraînement", cmd, cle="btn_vame_train",
        disabled=manque,
        help="Prépare d'abord le jeu d'entraînement." if manque else None,
    )
    _widgets.details_techniques(cmd)


# ============================================================
# Sous-étape 5 : evaluate (état indétectable)
# ============================================================

def _section_evaluate(projet: Path, etat: dict) -> None:
    _entete_etape("evaluate", None)
    st.caption(
        "EthoFlow ne peut pas détecter si l'évaluation a déjà tourné : "
        "elle ne produit que des figures, sans fichier repérable. Relance-la "
        "quand tu veux revoir le modèle avant de découper en motifs."
    )

    manque = not etat["train"]
    cmd = PL.vame_stage(projet, "evaluate")
    _job.bouton_lancer(
        projet, "Évaluer", cmd, cle="btn_vame_evaluate",
        disabled=manque,
        help="Aucun modèle entraîné pour l'instant : lance d'abord l'entraînement." if manque else None,
    )


# ============================================================
# Sous-étape 6 : segment (+ n_clusters)
# ============================================================

def _section_segment(projet: Path, etat: dict) -> None:
    _entete_etape("segment", etat["segment"])
    if etat["segment"]:
        st.caption(
            "Au moins une session découpée en motifs, pas forcément toutes."
        )
    else:
        st.caption("Attribue un motif comportemental à chaque image.")

    n_actuel = V.n_clusters(projet)
    st.caption(
        f"Nombre de motifs actuel : "
        f"**{n_actuel if n_actuel else 'non défini (VAME prendra 15)'}**."
    )

    csv_path = V.vame_project(projet) / "motif_labels.csv"
    if csv_path.exists():
        st.warning(
            "Des noms de motifs sont déjà saisis. Ce fichier est **unique "
            "par projet**, pas par nombre de motifs : changer ce nombre puis "
            "régénérer les noms efface ce qui a été saisi. Sauvegarde-le "
            "d'abord si tu changes le nombre de motifs."
        )
        try:
            contenu = csv_path.read_bytes()
        except OSError:
            contenu = None
        if contenu is not None:
            st.download_button(
                "Sauvegarder les noms de motifs",
                data=contenu, file_name="motif_labels_backup.csv",
                mime="text/csv", key="vame_segment_backup_csv",
            )

    nouveau_n = st.number_input(
        "Nombre de motifs pour ce découpage",
        min_value=2, value=int(n_actuel) if n_actuel else 15, step=1,
        key="vame_segment_n_clusters",
    )
    st.caption(
        "Chaque nombre de motifs a ses propres résultats : rien n'est "
        "écrasé. Évaluer le modèle avant est conseillé, pas imposé."
    )

    manque = not etat["train"]
    cmd = PL.vame_stage(projet, "segment", n_clusters=int(nouveau_n))
    _job.bouton_lancer(
        projet, "Découper en motifs", cmd, cle="btn_vame_segment",
        disabled=manque,
        help="Aucun modèle entraîné pour l'instant : lance d'abord l'entraînement." if manque else None,
    )
    _widgets.details_techniques(cmd)


# ============================================================
# Entrée : bannière d'échec, trois phases, puis leur détail
# ============================================================

def render() -> None:
    projet = require_project()
    _widgets.entete_etape(projet)
    etat = V.stage_status(projet)

    _job.panneau(projet)
    _cartes_phases(projet, etat)

    _titre_phase(1, "Préparer")
    _section_setup(projet, etat)
    _section_align(projet, etat)
    _section_trainset(projet, etat)
    st.divider()
    _titre_phase(2, "Entraîner")
    _section_train(projet, etat)
    _section_evaluate(projet, etat)
    st.divider()
    _titre_phase(3, "Segmenter")
    _section_segment(projet, etat)
    st.divider()
    _job.historique(projet)
