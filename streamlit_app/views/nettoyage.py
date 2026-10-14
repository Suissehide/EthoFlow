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
from lib.project import dlc_bodyparts
from lib import sessions as S
from views import _job
from views import _widgets
from views._widgets import champ_chemin

def _champ_keypoint(projet: Path, label: str, *, cle: str, defaut: str = "tail_base",
                    help: str | None = None) -> str:
    """Liste des keypoints du modèle DLC du projet ; champ libre seulement
    si aucun modèle lisible n'est configuré (une faute de frappe y passait
    sans erreur, et le graphe sortait vide)."""
    keypoints = dlc_bodyparts(projet)
    if not keypoints:
        return st.text_input(label, value=defaut, key=cle, width=320, help=help)
    index = keypoints.index(defaut) if defaut in keypoints else 0
    return st.selectbox(label, keypoints, index=index, key=cle, width=320, help=help)


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
    st.markdown(lucide_title("waypoints", "Séparer les animaux par arène"), unsafe_allow_html=True)

    if project_kind(projet) != "multi":
        st.caption("Un seul animal par vidéo dans ce projet : rien à séparer, "
                   "passe directement au nettoyage ci-dessous.")
        return

    st.caption(
        "Pose faite sur la vidéo entière (plusieurs animaux) : sépare d'abord "
        "chaque animal selon son arène. Le nettoyage travaille ensuite animal "
        "par animal."
    )
    choisies, tout = _widgets.selecteur_sessions(projet, cle="nettoyage_arenes",
                                                 colonne="split")
    clean = st.checkbox("Reconstituer les petits trous pendant la séparation",
                        value=True, key="nettoyage_arenes_clean")
    with st.expander("Réglages avancés"):
        seuil = st.number_input(
            "Confiance minimale", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            key="nettoyage_arenes_likelihood", width=220,
            help="En dessous, un point est considéré comme non détecté.")
        interp = st.number_input(
            "Plus long trou reconstitué (images)", min_value=1, value=25, step=1,
            key="nettoyage_arenes_interp", width=220,
            help="Les trous plus longs restent vides.")

    n = _widgets.nb_sessions(projet, choisies, tout)
    cmd = PL.assign_arenas(
        projet, sessions=choisies, all_sessions=tout,
        likelihood_threshold=float(seuil), interp_limit=int(interp), clean=clean,
    )
    _job.bouton_lancer(projet, _widgets.libelle_lancer(n), cmd, cle="btn_nettoyage_arenes",
                       disabled=n == 0, help="Choisis au moins une session." if n == 0 else None)
    _widgets.details_techniques(cmd)


# ============================================================
# Section 2 : nettoyage des poses (les 4 filtres)
# ============================================================

def _section_nettoyage(projet: Path) -> None:
    st.markdown(lucide_title("brush-cleaning", "Nettoyer les poses"), unsafe_allow_html=True)
    st.caption(
        "Chaque session passe par quatre filtres. Les points écartés sont "
        "reconstitués à partir de leurs voisins quand le trou est court ; "
        "les trous plus longs restent vides."
    )
    with st.expander("Ce que font les quatre filtres"):
        st.markdown(
            "1. **Lissage** : gomme les tremblements d'une ou deux images.\n"
            "2. **Confiance** : écarte les points dont le modèle n'est pas sûr.\n"
            "3. **Déplacements impossibles** : écarte les points qui sautent "
            "plus vite qu'une souris ne court. C'est le filtre le plus "
            "efficace : il attrape aussi les points sûrs d'eux mais faux.\n"
            "4. **Points collés** : repère un point qui retombe toujours au "
            "même endroit (reflet, coin d'arène), sans le confondre avec un "
            "animal réellement immobile."
        )

    choisies, tout = _widgets.selecteur_sessions(projet, cle="nettoyage_clean",
                                                 colonne="nettoyage")
    skip_existing = st.checkbox("Ignorer les sessions déjà nettoyées",
                                value=True, key="nettoyage_skip_existing")
    qc_plot = st.checkbox("Produire les graphes de contrôle", value=True,
                          key="nettoyage_qc_plot",
                          help="Un graphe de trajectoire par session, pour repérer "
                               "d'un coup d'œil un saut ou un point bloqué. Coûte "
                               "quelques secondes par session.")
    qc_bodypart = _champ_keypoint(
        projet, "Point suivi sur les graphes", cle="nettoyage_qc_bodypart",
        help="tail_base par défaut : le point le plus stable du corps, jamais caché "
             "sous l'animal. Un saut sur sa trajectoire est donc forcément une erreur "
             "de suivi, pas un vrai mouvement.")

    # --- Filtre 3 : il lui faut l'échelle -------------------------------
    echelle = px_per_cm(projet)
    if echelle is None:
        st.warning(
            "**Le filtre des déplacements impossibles est désactivé** : "
            "l'échelle n'est pas calibrée. C'est pourtant le plus efficace. "
            "Calibre-la dans **Vidéos & calibration → Échelle**, ou saisis-la "
            "ci-dessous pour ce nettoyage seulement."
        )
        saisie = st.number_input("Pixels par cm (pour ce nettoyage)", min_value=0.0,
                                 value=0.0, step=0.5, width=220,
                                 key="nettoyage_px_per_cm_fallback",
                                 help="Mesure-la de préférence sur une règle filmée avec "
                                      "ton setup. 0 = filtre désactivé.")
        echelle = saisie if saisie > 0 else None

    with st.expander("Réglages avancés"):
        col1, col2 = st.columns(2)
        with col1:
            window_length = st.number_input(
                "Lissage (images)", min_value=1, value=5, step=2,
                key="nettoyage_window_length",
                help="Fenêtre du filtre de lissage, en nombre impair d'images.")
            likelihood_threshold = st.number_input(
                "Confiance minimale", min_value=0.0, max_value=1.0, value=0.70,
                step=0.05, key="nettoyage_likelihood",
                help="La confiance que le modèle donne à chaque point, de 0 à 1. "
                     "Plus haut = plus sévère, plus de points reconstitués. "
                     "0,70 est la valeur recommandée par l'équipe de VAME. "
                     "Un point peut être sûr de lui et faux : les filtres 3 et 4 "
                     "sont là pour ça.")
            max_speed = st.number_input(
                "Vitesse maximale (m/s)", min_value=0.1, value=5.0, step=0.5,
                key="nettoyage_max_speed",
                help="Au-delà, un déplacement d'une image à l'autre est jugé "
                     "impossible. Si des sauts restent visibles sur les graphes, "
                     "descends à 4. Sans échelle, ce filtre ne tourne pas.")
        with col2:
            interp_limit = st.number_input(
                "Plus long trou reconstitué (images)", min_value=1, value=25,
                step=1, key="nettoyage_interp_limit",
                help="25 images ≈ 1 seconde à 25 images par seconde. Les trous plus "
                     "longs restent vides : ce sont souvent de vraies occlusions.")
            sticky_detection = st.checkbox(
                "Repérer les points collés", value=True, key="nettoyage_sticky",
                help="Filtre 4 : un point qui retombe souvent au même endroit, "
                     "dispersé dans le temps (reflet, coin d'arène).")
        if echelle is not None:
            st.caption(f"Échelle utilisée : {echelle} pixels par cm.")

    n = _widgets.nb_sessions(projet, choisies, tout, colonne="nettoyage",
                             ignorer_faites=skip_existing)
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
        projet, _widgets.libelle_lancer(n), cmd, cle="btn_nettoyage_clean",
        disabled=n == 0,
        help=None if n else ("Toutes ces sessions sont déjà nettoyées." if skip_existing
                             else "Choisis au moins une session."),
    )
    _widgets.details_techniques(cmd, notes=[
        "Sorties : `data/dlc-output/<session>/<session>_clean.h5` (un fichier "
        "nettoyé par session) et `data/dlc-output/_qc_trajectories/<session>_<point>.png`.",
        "Étape facultative : sans `_clean.h5`, VAME prend le `.h5` le plus abouti "
        "du dossier (`*_filtered.h5` avant la sortie brute de DLC).",
        "Pas de `_filtered.h5` produit par `dlc.filterpredictions` : la vidéo a "
        "sans doute été déplacée depuis la pose. Le `.h5` doit être à côté de la "
        "vidéo indiquée dans les métadonnées.",
    ])


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

    st.markdown(lucide_title("circle-check", "Résultat du nettoyage"), unsafe_allow_html=True)

    # Le dernier nettoyage réussi, pas le job courant : une inspection ou un
    # autre job lancé ensuite faisait disparaître ce résumé.
    job = _dernier_job_nettoyage(projet)
    if job is None:
        st.caption(
            "Après un nettoyage, cette section donne pour chaque session "
            "la part d'images utilisables et de points reconstitués."
        )
    else:
        log = runner.read_log(projet, job.job_id)
        resultats = _parse_resume(log or "")
        if not resultats:
            st.caption(
                "Nettoyage terminé, mais son résumé n'a pas pu être lu : "
                "voir son journal dans l'historique des jobs ci-dessus."
            )
        else:
            st.dataframe(
                [
                    {
                        "Session": r["session"],
                        "Images": r["n_frames"],
                        "Points reconstitués": r["n_repaired"],
                        "% utilisables": round(r["pct_useful"], 1),
                        "% reconstitués (≈)": round(r["pct_repaired"], 1),
                    }
                    for r in resultats
                ],
                width="stretch", hide_index=True,
            )
            pire = max(r["pct_repaired"] for r in resultats)
            if pire > 15:
                st.warning(
                    f"Jusqu'à {pire:.1f} % de points reconstitués sur une "
                    "session. Au-delà de 10 à 15 %, le problème vient du "
                    "modèle de pose, pas du nettoyage : le compenser ici le "
                    "masquerait. Mieux vaut ré-entraîner le modèle (voir "
                    "**Qualité DLC**)."
                )
            elif pire > 10:
                st.info(
                    f"Jusqu'à {pire:.1f} % de points reconstitués : proche "
                    "des 10 à 15 % au-delà desquels il vaut mieux améliorer "
                    "le modèle de pose que compenser au nettoyage."
                )
            else:
                st.caption(
                    f"Au plus {pire:.1f} % de points reconstitués, sous les "
                    "10 à 15 % : le modèle de pose est en bon état."
                )

    st.divider()
    _section_galerie_qc(projet)

    st.divider()
    _section_inspection(projet)


@st.cache_data(show_spinner="Lecture des fichiers de poses…", max_entries=8)
def _inspection_cache(projet: str, signature: tuple):
    # `signature` (chemins + dates de modification des .h5) invalide le
    # cache dès qu'un nettoyage ou une inférence réécrit un fichier.
    return S.inspection(Path(projet))


def _section_inspection(projet: Path) -> None:
    st.markdown(lucide_title("search", "Bilan qualité par session"), unsafe_allow_html=True)
    st.caption(
        "Pour chaque session : la part d'images où l'animal est détecté, les "
        "trous, et la fiabilité de chaque point du corps (sur les poses "
        "nettoyées quand elles existent). Ne modifie rien."
    )
    if not st.toggle("Afficher le bilan", key="nettoyage_inspection"):
        return
    signature = tuple(sorted((str(h), h.stat().st_mtime)
                             for h in dlc_output_dir(projet).glob("*/*.h5")))
    sessions, keypoints = _inspection_cache(str(projet), signature)
    if sessions.empty:
        st.info("Aucune pose estimée pour l'instant : lance d'abord l'étape Pose.")
        return
    st.dataframe(
        sessions, hide_index=True, width="stretch",
        column_config={"Couverture (%)": st.column_config.ProgressColumn(
            "Couverture (%)", min_value=0, max_value=100, format="%.1f %%")},
    )
    st.caption("Fiabilité de chaque point du corps (% des images), du moins au plus fiable :")
    st.dataframe(keypoints.loc[keypoints.mean(axis=1).sort_values().index],
                 width="stretch")
    with st.expander("Comment lire ce bilan"):
        st.markdown(
            "- **Verdict** : d'après la couverture. 90 % ou plus excellent, 80 % bon, "
            "70 % limite, en dessous insuffisant.\n"
            "- **Plus long trou** : un trou de plusieurs secondes est une occlusion "
            "ou une perte de suivi. VAME risque d'en faire un motif à lui tout seul.\n"
            "- **Points les moins fiables** (en haut du second tableau) : souvent une "
            "patte arrière ou le bout de la queue. Un point vers 40 % se retire "
            "(**Outils avancés → Retirer les points trop peu fiables**) plutôt que de "
            "polluer la segmentation.\n\n"
            "C'est ce bilan qu'il faut regarder avant de lancer des heures "
            "d'entraînement VAME."
        )


# ============================================================
# Section 3bis : galerie QC des trajectoires (Task 21)
# ============================================================

def _dernier_job_nettoyage(projet: Path):
    """Dernier job `prepare_vame_input_custom.py` réussi, courant ou passé.

    `runner.current` seul ne suffit pas : si un autre job (régénération
    des graphes, VAME…) a tourné après le nettoyage, ce dernier n'est plus
    le job « courant » mais reste celui dont le résumé et les seuils
    comptent.
    """
    from lib import runner

    for job in [runner.current(projet), *runner.history(projet, limit=20)]:
        if job is not None and job.script == "prepare_vame_input_custom.py" and job.state == "succeeded":
            return job
    return None


def _section_regenerer_keypoint(projet: Path, keypoints_existants: list[str]) -> None:
    st.markdown(
        lucide_title("brush-cleaning", "Graphes de contrôle sur un autre point"),
        unsafe_allow_html=True,
    )

    job = _dernier_job_nettoyage(projet)
    if job is None:
        st.caption(
            "Aucun nettoyage réussi pour l'instant : lance d'abord un "
            "nettoyage ci-dessus."
        )
        return

    kwargs = PL.parse_prepare_vame_input_args(job.argv)
    if kwargs is None:
        st.warning(
            "Les réglages du dernier nettoyage n'ont pas pu être relus. Avec "
            "des réglages devinés, les graphes ne seraient plus comparables : "
            "relance plutôt un nettoyage complet ci-dessus."
        )
        return

    st.caption(
        "Refait le dernier nettoyage avec **les mêmes réglages et les mêmes "
        "sessions**, en traçant un autre point du corps. Les graphes des "
        "autres points sont conservés."
    )
    st.caption(
        "Les sessions déjà nettoyées sont retraitées : c'est d'elles qu'on "
        "veut un nouveau graphe."
    )

    nouveau_kp = _champ_keypoint(projet, "Point du corps", cle="nettoyage_qc_regen_bodypart")
    if nouveau_kp and nouveau_kp in keypoints_existants:
        st.caption(
            f"Des graphes existent déjà pour `{nouveau_kp}` : ils seront "
            "remplacés."
        )

    kwargs = dict(kwargs)
    kwargs.pop("qc_plot", None)
    kwargs.pop("skip_existing", None)
    cmd = PL.prepare_vame_input(
        projet, qc_bodypart=nouveau_kp or "tail_base",
        qc_plot=True, skip_existing=False, **kwargs,
    )
    _job.bouton_lancer(
        projet, f"Tracer {nouveau_kp or 'tail_base'}", cmd,
        cle="btn_nettoyage_qc_regen", type="secondary",
        disabled=not nouveau_kp.strip(),
        help=None if nouveau_kp.strip() else "Indique un point du corps.",
    )


def _section_brut_vs_nettoye(projet: Path, galerie: dict[str, dict[str, Path]], keypoint: str) -> None:
    st.markdown(lucide_title("video", "Avant / après nettoyage"), unsafe_allow_html=True)
    st.caption(
        "La vidéo annotée par la pose (avant nettoyage) à côté de la "
        "trajectoire nettoyée du point choisi, pour la même session."
    )

    sessions_dispo = sorted(galerie.get(keypoint, {}))
    session_id = st.selectbox(
        "Session", options=sessions_dispo, key="nettoyage_qc_brut_session",
    )

    col_video, col_plot = st.columns(2)
    with col_video:
        st.caption("Vidéo annotée, avant nettoyage")
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
                "Pas de vidéo annotée pour cette session. Normal avec « Mon "
                "modèle entraîné », qui n'en produit pas ; avec SuperAnimal, "
                "vérifie que la pose a bien tourné pour cette session."
            )
    with col_plot:
        st.caption(f"Trajectoire nettoyée · {keypoint}")
        chemin_plot = galerie.get(keypoint, {}).get(session_id)
        if chemin_plot:
            st.image(str(chemin_plot), width="stretch")
        else:
            st.caption(f"Pas de graphe pour {session_id} · {keypoint}.")


def _section_galerie_qc(projet: Path) -> None:
    st.markdown(
        lucide_title("chart-column", "Trajectoires de contrôle"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Un bon nettoyage : la trajectoire d'un point sur toute la vidéo ne "
        "montre **aucun saut anormal**, sans avoir eu à jeter beaucoup de "
        "points (VAME découpe mal les trous)."
    )
    st.caption(
        "Des sauts restent visibles ? Baisse la vitesse maximale (4 m/s) ou "
        "monte la confiance minimale, dans les réglages avancés du nettoyage."
    )

    galerie = S.list_qc_trajectories(projet)
    if not galerie:
        st.caption(
            "Aucun graphe de contrôle trouvé : lance un nettoyage avec "
            "« Produire les graphes de contrôle » coché."
        )
        return

    keypoints = sorted(galerie)
    defaut_kp = st.session_state.get("nettoyage_qc_bodypart") or "tail_base"
    index_defaut = keypoints.index(defaut_kp) if defaut_kp in keypoints else 0
    keypoint = st.selectbox(
        "Point affiché", options=keypoints, index=index_defaut,
        key="nettoyage_qc_galerie_keypoint",
    )
    st.caption(
        "Les graphes de chaque point sont conservés séparément."
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
            "Dépannages ponctuels, hors du parcours normal. Les chemins sont "
            "pré-remplis pour ce projet, mais restent des dossiers ordinaires."
        )

        # --- filter_keypoints -------------------------------------------------
        st.markdown('<p class="ef-section">Retirer les points trop peu fiables</p>', unsafe_allow_html=True)
        # Empilés plutôt que côte à côte : `champ_chemin` crée déjà ses
        # propres colonnes pour loger le bouton, et Streamlit interdit de
        # les imbriquer.
        fk_input = champ_chemin(
            "Dossier d'entrée", cle="nettoyage_fk_input",
            valeur_defaut=str(dlc_output_dir(projet)),
        )
        fk_output = champ_chemin(
            "Dossier de sortie", cle="nettoyage_fk_output",
            valeur_defaut=str(dlc_output_dir(projet).parent / "dlc-output-filtered"),
        )
        fk_min_validity = st.number_input(
            "Fiabilité minimale d'un point (0 = ne rien retirer)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key="nettoyage_fk_min_validity",
            help="Retire les points du corps dont la fiabilité moyenne, sur tous "
                 "les fichiers, est sous ce seuil. Utile quand un point très peu "
                 "fiable (bout de queue) fait planter l'entraînement VAME.",
        )
        fk_dry_run = st.checkbox("Aperçu seulement (n'écrit rien)", value=True, key="nettoyage_fk_dry_run")
        cmd_fk = PL.filter_keypoints(
            input_dir=fk_input, output_dir=fk_output,
            min_validity=fk_min_validity if fk_min_validity > 0 else None,
            dry_run=fk_dry_run,
        )
        _job.bouton_lancer(
            projet, "Retirer les points", cmd_fk,
            cle="btn_nettoyage_filter_keypoints", type="secondary",
            disabled=not fk_input.strip() or not fk_output.strip(),
        )

        st.divider()

        # --- fill_nan_h5 --------------------------------------------------
        st.markdown('<p class="ef-section">Combler tous les trous restants</p>', unsafe_allow_html=True)
        st.caption("Bouche aussi les longs trous que le nettoyage laisse vides. "
                   "À utiliser seulement si VAME refuse des fichiers à cause d'eux.")
        fn_root = champ_chemin(
            "Dossier source", cle="nettoyage_fn_root",
            valeur_defaut=str(dlc_output_dir(projet)),
        )
        fn_output = champ_chemin(
            "Dossier de sortie (vide = réécriture en place)",
            cle="nettoyage_fn_output",
        )
        fn_dry_run = st.checkbox("Aperçu seulement (n'écrit rien)", value=True, key="nettoyage_fn_dry_run")
        cmd_fn = PL.fill_nan_h5(
            root=fn_root, output_dir=fn_output or None, dry_run=fn_dry_run,
        )
        _job.bouton_lancer(
            projet, "Combler les trous", cmd_fn,
            cle="btn_nettoyage_fill_nan", type="secondary",
            disabled=not fn_root.strip(),
        )

        st.divider()

        # --- trim_empty_arena -----------------------------------------------
        st.markdown(
            "**Couper l'arène vide** en début et fin de session, d'après le "
            "fichier de validité produit par les analyses."
        )
        te_csv = champ_chemin(
            "Fichier de validité (.csv)", cle="nettoyage_te_csv",
            mode="fichier", extensions=[".csv"],
            placeholder=str(projet / "results" / "validity_per_session.csv"),
        )
        te_h5_in = champ_chemin(
            "Dossier des poses", cle="nettoyage_te_h5_in",
            valeur_defaut=str(dlc_output_dir(projet)),
        )
        te_h5_out = champ_chemin(
            "Dossier des poses coupées", cle="nettoyage_te_h5_out",
            valeur_defaut=str(dlc_output_dir(projet).parent / "dlc-output-trimmed"),
        )
        te_v_in = champ_chemin(
            "`--video-input`", cle="nettoyage_te_v_in",
            valeur_defaut=str(projet / "data" / "cropped"),
        )
        te_v_out = champ_chemin(
            "`--video-output`", cle="nettoyage_te_v_out",
            valeur_defaut=str(projet / "data" / "cropped-trimmed"),
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

    _widgets.entete_etape(projet)

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
