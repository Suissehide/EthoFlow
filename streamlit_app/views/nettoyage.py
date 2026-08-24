"""Page Nettoyage — étape 6b du pipeline.

DLC rend une position pour chaque keypoint, sur chaque frame, quoi qu'il
arrive — y compris quand la patte est cachée sous le corps ou qu'un reflet
IR ressemble à une truffe. VAME segmente des **trajectoires** : un keypoint
qui téléporte à travers l'arène pendant trois frames devient un faux motif
comportemental qui ressort dans les stats finales. Cette étape repère ces
frames et les reconstruit depuis leurs voisines — de l'assurance qualité,
pas une conversion de format : le pipeline tourne sans (README §6b).

`prepare_vame_input_custom.py` tourne dans l'env conda `dlc` (il importe
`deeplabcut` pour le filtre médian) — voir `SCRIPT_ENVS` dans
`lib/pipeline.py`. Aucune commande n'est exécutée ici : `lib.pipeline`
construit, `views._job.bouton_lancer` lance via `lib.runner`.
"""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

import lib.pipeline as PL
from lib.config import (
    dlc_output_dir,
    project_kind,
    px_per_cm,
    require_project,
)
from lib.icons import lucide_title
from lib import sessions as S
from views import _job, _widgets

# Ligne "[i/n] session_id" imprimée avant chaque session traitée.
_RE_SESSION = re.compile(r"^\[\d+/\d+\]\s+(\S+)\s*$")
# Ligne de stats imprimée après chaque session réussie — voir
# prepare_vame_input_custom.py, juste avant `n_ok += 1`.
_RE_STATS = re.compile(
    r"✓\s*(?P<n_frames>\d+)\s*frames\s*\|\s*"
    r"(?P<n_kpts>\d+)\s*kpts\s*\|.*?\|\s*"
    r"réparées\s*(?P<n_repaired>\d+)\s*\|\s*"
    r"(?P<pct_useful>[\d.]+)%\s*utilisables"
)


# ============================================================
# Section 1 : split par arène (multi uniquement)
# ============================================================

def _section_arenes(projet: Path) -> None:
    st.markdown(lucide_title("waypoints", "Split par arène"), unsafe_allow_html=True)

    if project_kind(projet) != "multi":
        st.caption(
            "Projet détecté comme `single` — un seul animal par vidéo. Le "
            "split par arène (`assign_arenas.py`) n'a de sens que pour "
            "séparer plusieurs animaux filmés dans le même cadre (voie A) ; "
            "cette section ne s'applique donc pas ici, les `.h5` produits "
            "par l'étape Pose sont déjà single-animal."
        )
        return

    st.caption(
        "Projet `multi` : `assign_arenas.py` doit tourner **avant** le "
        "nettoyage sur la voie A (DLC multi-animal, puis split par arène) — "
        "sinon `prepare_vame_input_custom.py` n'a pas de `.h5` "
        "single-animal par session à nettoyer."
    )

    choisies, tout = _widgets.selecteur_sessions(projet, cle="nettoyage_arenes")
    col_seuil, col_interp = st.columns(2)
    with col_seuil:
        seuil = st.number_input(
            "Seuil de likelihood (`--likelihood-threshold`)",
            min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            key="nettoyage_arenes_likelihood",
        )
    with col_interp:
        interp = st.number_input(
            "`--interp-limit` (frames)",
            min_value=1, value=25, step=1,
            key="nettoyage_arenes_interp",
        )
    clean = st.checkbox(
        "Interpoler pendant le split (sinon `--no-clean`)",
        value=True, key="nettoyage_arenes_clean",
    )

    pas_de_sessions = not tout and not choisies
    cmd = PL.assign_arenas(
        projet, sessions=choisies, all_sessions=tout,
        likelihood_threshold=float(seuil), interp_limit=int(interp),
        clean=clean,
    )
    _job.bouton_lancer(
        projet, "Lancer le split par arène", cmd,
        cle="btn_nettoyage_arenes",
        disabled=pas_de_sessions,
        help=None if not pas_de_sessions else
             "Sélectionne au moins une session, ou coche « Toutes les sessions ».",
    )


# ============================================================
# Section 2 : nettoyage des poses (les 4 passes)
# ============================================================

def _section_nettoyage(projet: Path) -> None:
    st.markdown(lucide_title("brush-cleaning", "Nettoyage des poses"), unsafe_allow_html=True)
    st.caption(
        "`prepare_vame_input_custom.py` — quatre passes successives par "
        "session (README §6b), dans cet ordre :"
    )

    st.markdown(
        "1. **Filtre médian temporel** (`dlc.filterpredictions`) — tue les "
        "jitters d'une ou deux frames.\n"
        "2. **Cutoff de likelihood** — le filet grossier.\n"
        "3. **Détection de vitesse aberrante** — la méthode que l'équipe "
        "VAME/LIN privilégie : indépendante de la likelihood, elle attrape "
        "aussi les labels *confiants mais faux*.\n"
        "4. **Détection de points collants** — repère les coordonnées où un "
        "keypoint atterrit anormalement souvent (reflet IR fixe, coin "
        "d'arène), en distinguant un artefact (frames dispersées dans le "
        "temps) d'une immobilité réelle (frames contiguës)."
    )
    st.caption(
        "Les frames marquées par les passes 2/3/4 sont **interpolées** "
        "depuis leurs voisines valides, pas jetées. Les trous plus longs "
        "que `--interp-limit` restent NaN."
    )

    choisies, tout = _widgets.selecteur_sessions(projet, cle="nettoyage_clean")

    col1, col2, col3 = st.columns(3)
    with col1:
        window_length = st.number_input(
            "Passe 1 — `--window-length` (frames)",
            min_value=1, value=5, step=2, key="nettoyage_window_length",
            help="Fenêtre du filtre médian temporel.",
        )
    with col2:
        likelihood_threshold = st.number_input(
            "Passe 2 — `--likelihood-threshold`",
            min_value=0.0, max_value=1.0, value=0.70, step=0.05,
            key="nettoyage_likelihood",
            help="Recommandation de l'équipe VAME/LIN : 0.70.",
        )
    with col3:
        max_speed = st.number_input(
            "Passe 3 — `--max-speed` (m/s)",
            min_value=0.1, value=5.0, step=0.5, key="nettoyage_max_speed",
        )

    # --- Passe 3 : garde-fou px_per_cm ---------------------------------
    echelle = px_per_cm(projet)
    if echelle is None:
        st.warning(
            "**Passe 3 (vitesse aberrante) désactivée** — `px_per_cm` n'est pas "
            "calibré pour ce projet. C'est la passe la plus efficace selon "
            "l'équipe VAME/LIN : elle attrape les points *confiants mais faux*, "
            "que le seuil de likelihood laisse passer. Calibre l'échelle dans "
            "**Vidéos & calibration → Échelle px/cm**, ou saisis la valeur "
            "ci-dessous pour cette exécution."
        )
        saisie = st.number_input("px/cm (pour cette exécution seulement)",
                                 min_value=0.0, value=0.0, step=0.5,
                                 key="nettoyage_px_per_cm_fallback")
        echelle = saisie if saisie > 0 else None
    else:
        st.caption(f"Échelle : {echelle} px/cm — passe 3 active.")

    col4, col5 = st.columns(2)
    with col4:
        sticky_detection = st.checkbox(
            "Passe 4 active (`--no-sticky-detection` si décoché)",
            value=True, key="nettoyage_sticky",
        )
    with col5:
        interp_limit = st.number_input(
            "`--interp-limit` (frames ≈ 1 s à 25)",
            min_value=1, value=25, step=1, key="nettoyage_interp_limit",
        )

    qc_bodypart = st.text_input(
        "`--qc-bodypart`", value="tail_base", key="nettoyage_qc_bodypart",
    )
    st.caption(
        "`tail_base` par défaut : c'est le point le plus stable du corps — "
        "il ne disparaît jamais sous l'animal et bouge peu par rapport au "
        "centre de masse, donc un saut sur sa trajectoire est forcément une "
        "erreur de tracking, jamais un vrai mouvement."
    )

    col6, col7 = st.columns(2)
    with col6:
        qc_plot = st.checkbox("Graphes de contrôle (`--qc-plot`)", value=True,
                              key="nettoyage_qc_plot")
    with col7:
        skip_existing = st.checkbox("Sauter les sessions déjà traitées",
                                    value=True, key="nettoyage_skip_existing")

    pas_de_sessions = not tout and not choisies

    cmd = PL.prepare_vame_input(
        projet,
        likelihood_threshold=float(likelihood_threshold),
        max_speed=float(max_speed),
        px_per_cm=echelle,
        sessions=choisies,
        sticky_detection=sticky_detection,
        qc_plot=qc_plot,
        qc_bodypart=qc_bodypart or "tail_base",
        interp_limit=int(interp_limit),
        window_length=int(window_length),
        skip_existing=skip_existing,
    )
    _job.bouton_lancer(
        projet, "Lancer le nettoyage", cmd,
        cle="btn_nettoyage_clean",
        disabled=pas_de_sessions,
        help=None if not pas_de_sessions else
             "Sélectionne au moins une session, ou coche « Toutes les sessions ».",
    )


# ============================================================
# Section 3 : résumé d'exécution + QC chiffré
# ============================================================

def _parse_resume(log: str) -> list[dict]:
    """Extrait (session, n_frames, n_repaired, pct_useful) du log.

    Parsing texte, pas de logique pipeline : seulement pour afficher ce que
    le script a déjà imprimé, sans relancer de calcul côté app.
    """
    lignes = log.splitlines()
    session_courante = None
    resultats = []
    for ligne in lignes:
        m_session = _RE_SESSION.match(ligne)
        if m_session:
            session_courante = m_session.group(1)
            continue
        m_stats = _RE_STATS.search(ligne)
        if m_stats and session_courante:
            n_frames = int(m_stats["n_frames"])
            n_kpts = int(m_stats["n_kpts"])
            n_repaired = int(m_stats["n_repaired"])
            pct_useful = float(m_stats["pct_useful"])
            slots = n_frames * max(n_kpts, 1)
            pct_repaired = 100 * n_repaired / slots if slots else 0.0
            resultats.append({
                "session": session_courante,
                "n_frames": n_frames,
                "n_repaired": n_repaired,
                "pct_useful": pct_useful,
                "pct_repaired": pct_repaired,
            })
            session_courante = None
    return resultats


def _section_resume(projet: Path) -> None:
    from lib import runner

    st.markdown(lucide_title("circle-check", "Résumé et QC"), unsafe_allow_html=True)

    job = runner.current(projet)
    if job is None or job.script != "prepare_vame_input_custom.py" or job.state != "succeeded":
        st.caption(
            "Le résumé (% de frames utilisables, frames réparées par "
            "session) apparaît ici après un nettoyage réussi."
        )
    else:
        log = runner.read_log(projet, job.job_id)
        resultats = _parse_resume(log or "")
        if not resultats:
            st.caption(
                "Job terminé, mais le résumé par session n'a pas pu être "
                "extrait du log — voir le log complet dans le panneau de "
                "job ci-dessus."
            )
        else:
            st.dataframe(
                [
                    {
                        "Session": r["session"],
                        "Frames": r["n_frames"],
                        "Réparées": r["n_repaired"],
                        "% utilisables": round(r["pct_useful"], 1),
                        "% réparées (≈)": round(r["pct_repaired"], 1),
                    }
                    for r in resultats
                ],
                width="stretch", hide_index=True,
            )
            pire = max(r["pct_repaired"] for r in resultats)
            if pire > 15:
                st.warning(
                    f"Jusqu'à {pire:.1f} % de points réparés sur une "
                    "session — au-delà de 10-15 %, le README est clair : le "
                    "problème est le modèle DLC, pas le post-traitement. "
                    "Compenser ici ne fera que masquer un modèle faible — "
                    "retourne plutôt au **Parcours B** (ré-entraîner)."
                )
            elif pire > 10:
                st.info(
                    f"Jusqu'à {pire:.1f} % de points réparés — proche du "
                    "seuil de 10-15 % au-delà duquel le README recommande "
                    "de retravailler le modèle DLC plutôt que de compenser "
                    "en post-traitement."
                )
            else:
                st.caption(
                    f"Au plus {pire:.1f} % de points réparés — sous le seuil "
                    "de 10-15 % du README, le modèle DLC est en bon état."
                )

    st.divider()
    _section_galerie_qc(projet)

    st.divider()
    st.caption(
        "Contrôle qualité chiffré indépendant du nettoyage — `inspect_session.py` "
        "resynthétise les statistiques de validité sans rien modifier."
    )
    choisies, tout = _widgets.selecteur_sessions(projet, cle="nettoyage_inspect")
    cmd = PL.inspect_session(projet, sessions=choisies, all_sessions=tout)
    _job.bouton_lancer(
        projet, "Inspecter la qualité", cmd,
        cle="btn_nettoyage_inspect", type="secondary",
    )


# ============================================================
# Section 3bis : galerie QC des trajectoires (Task 21)
# ============================================================

def _dernier_job_nettoyage(projet: Path):
    """Dernier job `prepare_vame_input_custom.py` réussi, courant ou passé.

    `runner.current` seul ne suffit pas : si l'utilisateur a lancé
    `inspect_session.py` (juste en dessous sur cette page) après le
    nettoyage, ce dernier n'est plus le job « courant » mais reste bien
    celui dont les seuils doivent servir à la régénération.
    """
    from lib import runner

    for job in [runner.current(projet), *runner.history(projet, limit=20)]:
        if job is not None and job.script == "prepare_vame_input_custom.py" and job.state == "succeeded":
            return job
    return None


def _section_regenerer_keypoint(projet: Path, keypoints_existants: list[str]) -> None:
    st.markdown(
        lucide_title("brush-cleaning", "Régénérer sur un autre keypoint"),
        unsafe_allow_html=True,
    )

    job = _dernier_job_nettoyage(projet)
    if job is None:
        st.caption(
            "Aucun nettoyage réussi dans l'historique de ce projet — lance "
            "d'abord un nettoyage dans la section ci-dessus avant de "
            "pouvoir régénérer sur un autre keypoint."
        )
        return

    kwargs = PL.parse_prepare_vame_input_args(job.argv)
    if kwargs is None:
        st.warning(
            "Les seuils du dernier nettoyage réussi n'ont pas pu être "
            "relus depuis sa commande enregistrée. Régénérer ici avec des "
            "seuils devinés produirait des graphes qui ne seraient plus "
            "comparables aux précédents — relance plutôt un nettoyage "
            "complet dans la section ci-dessus, avec les seuils voulus."
        )
        return

    st.caption(
        "Rejoue le dernier nettoyage réussi (« "
        f"{job.label} », {job.started_at}) avec **exactement les mêmes "
        "seuils et les mêmes sessions** — seul le keypoint change. Le nom "
        "de fichier du graphe porte le keypoint, donc les graphes "
        "**coexistent** : régénérer sur un nouveau keypoint n'efface "
        "jamais ceux déjà produits pour un autre."
    )
    st.caption(
        "`--skip-existing` n'est **pas** repris tel quel : il ferait "
        "sauter toutes les sessions déjà nettoyées — précisément celles "
        "pour lesquelles on veut un nouveau graphe — et ne produirait "
        "donc aucun graphe du tout."
    )

    nouveau_kp = st.text_input(
        "`--qc-bodypart`", value="tail_base", key="nettoyage_qc_regen_bodypart",
    )
    if nouveau_kp and nouveau_kp in keypoints_existants:
        st.caption(
            f"Un graphe existe déjà pour `{nouveau_kp}` — il sera "
            "remplacé pour les sessions traitées (même keypoint = même "
            "nom de fichier)."
        )

    kwargs = dict(kwargs)
    kwargs.pop("qc_plot", None)
    kwargs.pop("skip_existing", None)
    cmd = PL.prepare_vame_input(
        projet, qc_bodypart=nouveau_kp or "tail_base",
        qc_plot=True, skip_existing=False, **kwargs,
    )
    _job.bouton_lancer(
        projet, f"Régénérer les graphes pour `{nouveau_kp or 'tail_base'}`", cmd,
        cle="btn_nettoyage_qc_regen", type="secondary",
        disabled=not nouveau_kp.strip(),
        help=None if nouveau_kp.strip() else "Renseigne un keypoint.",
    )


def _section_brut_vs_nettoye(projet: Path, galerie: dict[str, dict[str, Path]], keypoint: str) -> None:
    st.markdown(lucide_title("video", "Brut vs nettoyé"), unsafe_allow_html=True)
    st.caption(
        "Le `_labeled.mp4` de DLC (poses brutes, avant nettoyage) à côté "
        "du graphe de trajectoire du keypoint choisi ci-dessus (poses "
        "nettoyées), pour la même session."
    )

    sessions_dispo = sorted(galerie.get(keypoint, {}))
    session_id = st.selectbox(
        "Session", options=sessions_dispo, key="nettoyage_qc_brut_session",
    )

    col_video, col_plot = st.columns(2)
    with col_video:
        st.caption("`_labeled.mp4` (DLC, brut)")
        session_dir = dlc_output_dir(projet) / session_id
        # Recherche récursive (rglob, pas glob) : en mode single-animal,
        # run_superanimal_cropped écrit ses _labeled.mp4 dans le
        # sous-dossier scratch cropped-raw/, pas directement dans
        # dlc-output/<session>/ (même piège que Task 12, ruling R12.2). En
        # mode custom, l'analyze_videos de DLC n'écrit JAMAIS de
        # _labeled.mp4 : une liste vide dans ce mode est normale, pas un
        # signe de problème — le message ci-dessous reste donc neutre,
        # jamais un `st.warning`/`st.error`.
        candidats = sorted(session_dir.rglob("*_labeled.mp4")) if session_dir.is_dir() else []
        if candidats:
            st.video(str(candidats[0]))
        else:
            st.caption(
                "Aucun `_labeled.mp4` pour cette session. Normal en mode "
                "`custom` (DLC n'en produit jamais dans ce mode) ; en "
                "mode `superanimal`/`single-animal`, vérifie plutôt que "
                "l'inférence DLC a bien tourné pour cette session (page "
                "**Pose (DLC)**)."
            )
    with col_plot:
        st.caption(f"Trajectoire nettoyée — `{keypoint}`")
        chemin_plot = galerie.get(keypoint, {}).get(session_id)
        if chemin_plot:
            st.image(str(chemin_plot), width="stretch")
        else:
            st.caption(f"Pas de graphe QC pour `{session_id}` / `{keypoint}`.")


def _section_galerie_qc(projet: Path) -> None:
    st.markdown(
        lucide_title("chart-column", "Galerie QC des trajectoires"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Critère de l'équipe VAME/LIN : tracer la trajectoire d'un "
        "keypoint sur toute la vidéo ne doit montrer **aucun saut "
        "anormal de position** — et ce **sans avoir eu à jeter de "
        "points**, ce qui cause beaucoup de problèmes en aval (VAME "
        "segmente mal des trous). Un graphe par session et par "
        "`--qc-bodypart`, sous `data/dlc-output/_qc_trajectories/`."
    )

    galerie = S.list_qc_trajectories(projet)
    if not galerie:
        st.caption(
            "Aucun graphe de contrôle trouvé sous `_qc_trajectories/` — "
            "lance un nettoyage avec « Graphes de contrôle » coché dans la "
            "section ci-dessus pour les produire."
        )
        return

    keypoints = sorted(galerie)
    defaut_kp = st.session_state.get("nettoyage_qc_bodypart") or "tail_base"
    index_defaut = keypoints.index(defaut_kp) if defaut_kp in keypoints else 0
    keypoint = st.selectbox(
        "Keypoint affiché", options=keypoints, index=index_defaut,
        key="nettoyage_qc_galerie_keypoint",
    )
    st.caption(
        "Le nom de fichier porte le keypoint : les graphes de plusieurs "
        "keypoints **coexistent** sans jamais s'écraser entre eux."
    )

    sessions_du_keypoint = galerie[keypoint]
    colonnes = st.columns(3)
    for i, (session_id, chemin) in enumerate(sorted(sessions_du_keypoint.items())):
        with colonnes[i % 3]:
            st.image(str(chemin), caption=session_id, width="stretch")

    st.divider()
    _section_regenerer_keypoint(projet, keypoints)

    st.divider()
    _section_brut_vs_nettoye(projet, galerie, keypoint)


# ============================================================
# Section 4 : outils avancés (dépannage, pas projet-aware)
# ============================================================

def _section_avancee(projet: Path) -> None:
    with st.expander("Outils avancés", expanded=False):
        st.caption(
            "Dépannages ponctuels, pas des étapes du parcours principal. "
            "Ces trois scripts ne connaissent pas la structure d'un projet "
            "EthoFlow — les chemins ci-dessous sont pré-remplis à partir du "
            "projet courant, mais restent des chemins de fichiers ordinaires."
        )

        # --- filter_keypoints -------------------------------------------------
        st.markdown("**`filter_keypoints.py`** — retire des keypoints trop peu fiables")
        col_in, col_out = st.columns(2)
        with col_in:
            fk_input = st.text_input(
                "Dossier d'entrée", value=str(dlc_output_dir(projet)),
                key="nettoyage_fk_input",
            )
        with col_out:
            fk_output = st.text_input(
                "Dossier de sortie",
                value=str(dlc_output_dir(projet).parent / "dlc-output-filtered"),
                key="nettoyage_fk_output",
            )
        fk_min_validity = st.number_input(
            "`--min-validity` (0 = désactivé, utiliser `--keep`/`--drop` à la place)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key="nettoyage_fk_min_validity",
        )
        fk_dry_run = st.checkbox("`--dry-run`", value=True, key="nettoyage_fk_dry_run")
        cmd_fk = PL.filter_keypoints(
            input_dir=fk_input, output_dir=fk_output,
            min_validity=fk_min_validity if fk_min_validity > 0 else None,
            dry_run=fk_dry_run,
        )
        _job.bouton_lancer(
            projet, "Lancer filter_keypoints", cmd_fk,
            cle="btn_nettoyage_filter_keypoints", type="secondary",
            disabled=not fk_input.strip() or not fk_output.strip(),
        )

        st.divider()

        # --- fill_nan_h5 --------------------------------------------------
        st.markdown("**`fill_nan_h5.py`** — comble agressivement les NaN résiduels")
        col_root, col_out2 = st.columns(2)
        with col_root:
            fn_root = st.text_input(
                "Dossier source (`--root`)", value=str(dlc_output_dir(projet)),
                key="nettoyage_fn_root",
            )
        with col_out2:
            fn_output = st.text_input(
                "Dossier de sortie (vide = réécriture en place)",
                value="", key="nettoyage_fn_output",
            )
        fn_dry_run = st.checkbox("`--dry-run`", value=True, key="nettoyage_fn_dry_run")
        cmd_fn = PL.fill_nan_h5(
            root=fn_root, output_dir=fn_output or None, dry_run=fn_dry_run,
        )
        _job.bouton_lancer(
            projet, "Lancer fill_nan_h5", cmd_fn,
            cle="btn_nettoyage_fill_nan", type="secondary",
            disabled=not fn_root.strip(),
        )

        st.divider()

        # --- trim_empty_arena -----------------------------------------------
        st.markdown(
            "**`trim_empty_arena.py`** — tronque les frames d'arène vide en "
            "début/fin de session, à partir d'un `validity_per_session.csv` "
            "produit par `analyze_vame.py --validity-source`."
        )
        te_csv = st.text_input(
            "`--validity-csv`", value="", key="nettoyage_te_csv",
            placeholder=str(projet / "results" / "validity_per_session.csv"),
        )
        col_h5_in, col_h5_out = st.columns(2)
        with col_h5_in:
            te_h5_in = st.text_input(
                "`--h5-input`", value=str(dlc_output_dir(projet)),
                key="nettoyage_te_h5_in",
            )
        with col_h5_out:
            te_h5_out = st.text_input(
                "`--h5-output`",
                value=str(dlc_output_dir(projet).parent / "dlc-output-trimmed"),
                key="nettoyage_te_h5_out",
            )
        col_v_in, col_v_out = st.columns(2)
        with col_v_in:
            te_v_in = st.text_input(
                "`--video-input`", value=str(projet / "data" / "cropped"),
                key="nettoyage_te_v_in",
            )
        with col_v_out:
            te_v_out = st.text_input(
                "`--video-output`", value=str(projet / "data" / "cropped-trimmed"),
                key="nettoyage_te_v_out",
            )
        te_dry_run = st.checkbox("`--dry-run`", value=True, key="nettoyage_te_dry_run")

        te_manque = not all([te_csv.strip(), te_h5_in.strip(), te_h5_out.strip(),
                             te_v_in.strip(), te_v_out.strip()])
        cmd_te = PL.trim_empty_arena(
            validity_csv=te_csv or "validity_per_session.csv",
            h5_input=te_h5_in, h5_output=te_h5_out,
            video_input=te_v_in, video_output=te_v_out,
            dry_run=te_dry_run,
        )
        _job.bouton_lancer(
            projet, "Lancer trim_empty_arena", cmd_te,
            cle="btn_nettoyage_trim_empty", type="secondary",
            disabled=te_manque,
            help="Renseigne les cinq chemins." if te_manque else None,
        )


def render() -> None:
    projet = require_project()

    st.title("Nettoyage")
    st.caption(
        "Étape 6b du pipeline : nettoyage des poses DLC avant VAME. "
        "Facultative — le pipeline tourne sans — mais elle évite qu'un "
        "keypoint mal tracké devienne un faux motif comportemental. Tourne "
        "dans l'env conda `dlc`."
    )
    st.caption(f"Projet détecté comme `{project_kind(projet)}`.")

    _section_arenes(projet)
    st.divider()
    _section_nettoyage(projet)
    st.divider()
    _job.panneau(projet)
    _job.historique(projet)
    st.divider()
    _section_resume(projet)
    st.divider()
    _section_avancee(projet)
