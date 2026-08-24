"""Page Vidéos & calibration — étapes 4 et 6a du pipeline.

Trois besoins concrets (voir `lib/video.py`, qui porte toute la logique
testable) :

1. **Regarder une session avant de la traiter.** Vignette, lecteur, et les
   caractéristiques *réelles* de la vidéo (sonde OpenCV via `lib.video.probe`)
   confrontées à ce que déclare `metadata.yaml`. Un écart de fps fausse
   toutes les conversions frame → secondes en aval — se repérer ici coûte
   des secondes, se repérer après un run d'inférence coûte des heures.
2. **Re-pointer une vidéo déplacée.** Cas du README (Troubleshooting) : une
   metadata qui porte des chemins Windows sur une machine Linux, ou plus
   simplement un disque externe débranché. `find_relinks`/`apply_relinks`
   (lib/video.py) retrouvent et réécrivent `source_video`.
3. **Cropper les arènes (multi uniquement).** Voie B de l'étape 4 du README —
   crop puis DLC single-animal, l'alternative à la voie A (DLC multi-animal
   puis split, page Pose/Nettoyage).

Onglets ``Sessions`` et ``Crop`` construits ici ; la Task 20 ajoute
``Calibration arènes`` et ``Échelle px/cm`` au même `st.tabs(...)` dans
`render()`, sans restructurer le reste du fichier.

Aucune commande n'est exécutée ici : `lib.pipeline` construit,
`views._job.bouton_lancer` lance via `lib.runner`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import lib.pipeline as PL
from lib.config import arena_coords, project_kind, require_project
from lib.icons import lucide_title
from lib import sessions as S
from lib import video as V
from views import _job, _widgets

# Écart de fps (en images/seconde) au-delà duquel on alerte visiblement.
# Une valeur strictement nulle ferait ressortir des arrondis de sonde sans
# intérêt (24.999 vs 25) ; le README ne donne pas de seuil, celui-ci reste
# largement sous le premier palier de fps caméra usuel (24/25/30).
_TOLERANCE_FPS = 0.5


# ============================================================
# Sonde vidéo — mise en cache
# ============================================================

@st.cache_data(show_spinner=False)
def _probe_cache(chemin: str) -> V.VideoInfo:
    """Sonde mise en cache par chemin, pas par session.

    Streamlit relance tout le script à chaque interaction : sonder la vidéo
    de chaque session d'une liste à chaque rerun serait un aller-retour
    disque par session, à chaque clic. Seule la session affichée est sondée
    (`render()` n'appelle ceci que pour la session choisie dans le
    sélecteur), et le résultat est mis en cache par `chemin` — une session
    re-pointée obtient un nouveau chemin, donc une nouvelle entrée de cache,
    jamais une sonde périmée après un re-pointage.
    """
    return V.probe(Path(chemin))


def _fps_declare(meta: dict) -> float | None:
    camera = meta.get("camera")
    if not isinstance(camera, dict):
        return None
    valeur = camera.get("fps")
    if valeur is None:
        return None
    try:
        return float(valeur)
    except (TypeError, ValueError):
        return None


def _ecart_fps(meta: dict, info: V.VideoInfo) -> tuple[float, float] | None:
    """(fps déclaré, fps réel) si l'écart dépasse la tolérance, sinon `None`."""
    declare = _fps_declare(meta)
    if declare is None or info.fps is None:
        return None
    if abs(declare - info.fps) > _TOLERANCE_FPS:
        return declare, info.fps
    return None


def _fmt(valeur: float | int | str | None) -> str:
    """Chaîne d'affichage uniforme pour la colonne d'un tableau de comparaison.

    Les colonnes mélangent naturellement des int (dimensions, frames), des
    float (fps, durée) et des valeurs absentes : les laisser en types
    hétérogènes fait échouer la sérialisation Arrow de `st.dataframe` dès
    qu'une colonne contient à la fois un nombre et « — ». Tout passer en
    chaîne évite le problème et reste lisible, cette table n'étant jamais
    recalculée à partir de ce qu'elle affiche.
    """
    if valeur is None:
        return "—"
    if isinstance(valeur, float):
        return f"{valeur:.2f}"
    return str(valeur)


def _tableau_comparaison(meta: dict, info: V.VideoInfo) -> pd.DataFrame:
    """Déclaré (metadata `camera:`) vs réel (sonde) — fps, dimensions, frames, durée.

    Seuls fps/largeur/hauteur sont déclarés dans le schéma (README §3.4,
    clé `camera:`) : nombre de frames et durée n'y figurent jamais, donc
    « — » plutôt qu'une valeur inventée.
    """
    camera = meta.get("camera") if isinstance(meta.get("camera"), dict) else {}
    return pd.DataFrame([
        {
            "Caractéristique": "fps",
            "Déclaré (metadata)": _fmt(camera.get("fps")),
            "Réel (fichier)": _fmt(info.fps),
        },
        {
            "Caractéristique": "Largeur (px)",
            "Déclaré (metadata)": _fmt(camera.get("width")),
            "Réel (fichier)": _fmt(info.width),
        },
        {
            "Caractéristique": "Hauteur (px)",
            "Déclaré (metadata)": _fmt(camera.get("height")),
            "Réel (fichier)": _fmt(info.height),
        },
        {
            "Caractéristique": "Frames",
            "Déclaré (metadata)": "—",
            "Réel (fichier)": _fmt(info.n_frames),
        },
        {
            "Caractéristique": "Durée (s)",
            "Déclaré (metadata)": "—",
            "Réel (fichier)": _fmt(info.duration_s),
        },
    ])


# ============================================================
# Onglet Sessions — vignette, lecteur, comparaison, metadata, re-pointage
# ============================================================

def _section_video_session(session_id: str, meta: dict) -> None:
    st.markdown(lucide_title("video", f"Session {session_id}"), unsafe_allow_html=True)

    source = meta.get("source_video")
    chemin = Path(source) if source else None

    if chemin is None or not chemin.is_file():
        # Dégradation gracieuse (ruling brief) : jamais d'exception, on
        # affiche ce qu'on sait et on renvoie vers le re-pointage plus bas —
        # jamais une page vide ou cassée.
        st.warning(
            (f"Vidéo source introuvable : `{chemin}`." if chemin
             else "Aucune `source_video` dans la metadata de cette session.")
            + " Vidéo déplacée, disque débranché, ou changement de machine ? "
              "Voir la section **Re-pointer des vidéos manquantes** ci-dessous."
        )
    else:
        info = _probe_cache(str(chemin))
        if not info.exists:
            st.warning(
                f"`{chemin}` existe sur le disque mais n'a pas pu être ouvert "
                "comme vidéo (fichier corrompu, ou codec non supporté par "
                "OpenCV sur cette machine)."
            )
        else:
            col_vignette, col_lecteur = st.columns([1, 2])
            with col_vignette:
                png = V.frame_png_bytes(chemin, index=0, max_width=400)
                if png:
                    st.image(png, caption="Première frame", use_container_width=True)
                else:
                    st.caption("Vignette indisponible.")
            with col_lecteur:
                st.video(str(chemin))

            if info.fps is None:
                st.warning(
                    "Le fps réel n'a pas pu être lu dans le fichier — la "
                    "comparaison ci-dessous ne peut pas se faire sur cette ligne."
                )
            ecart = _ecart_fps(meta, info)
            if ecart:
                declare, reel = ecart
                st.error(
                    f"**Écart de fps** : {declare:g} déclaré dans la metadata "
                    f"contre {reel:.2f} mesuré dans le fichier. Toute "
                    "conversion frame → secondes en aval (VAME, analyses) "
                    "sera fausse tant que ce n'est pas corrigé — corrige la "
                    "metadata ou reconfirme le fps réel avant de lancer "
                    "l'inférence."
                )

            st.dataframe(
                _tableau_comparaison(meta, info),
                use_container_width=True, hide_index=True,
            )

    st.markdown(lucide_title("clipboard-list", "Metadata de la session"), unsafe_allow_html=True)
    champs = S.metadata_fields(meta)
    if champs:
        # `Valeur` en chaîne : les clés de metadata sont libres (§sync_from_excel),
        # une colonne mêlant int/float/bool/None ferait échouer la
        # sérialisation Arrow de `st.dataframe` au premier projet dont les
        # types varient d'une clé à l'autre (même piège que `_fmt` ci-dessus).
        st.dataframe(
            pd.DataFrame([{"Clé": k, "Valeur": _fmt(v)} for k, v in champs.items()]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("Aucun champ de metadata scalaire pour cette session.")


def _section_arenes_session(projet: Path, meta: dict) -> None:
    if project_kind(projet) != "multi":
        return
    st.markdown(lucide_title("waypoints", "Arènes"), unsafe_allow_html=True)
    df_arenes = S.arenes_dataframe(meta)
    if df_arenes.empty:
        st.caption("Aucune arène définie dans la metadata de cette session.")
    else:
        st.dataframe(df_arenes, use_container_width=True, hide_index=True)


def _section_relink(projet: Path, sessions_manquantes: list[str]) -> None:
    st.markdown(
        lucide_title("folder-open", "Re-pointer des vidéos manquantes"),
        unsafe_allow_html=True,
    )
    st.warning(
        f"{len(sessions_manquantes)} session(s) sans vidéo source valide : "
        f"{', '.join(sessions_manquantes)}."
    )
    st.caption(
        "Cas classique du Troubleshooting du README : la metadata porte des "
        "chemins d'une autre machine (Windows → Linux, ou l'inverse), ou "
        "d'un disque débranché depuis. Indique le dossier où se trouvent "
        "réellement les vidéos aujourd'hui — la recherche se fait par "
        "`<id>.mp4`, où `id` est la clé `id` de la metadata."
    )

    dossier = st.text_input(
        "Dossier des vidéos", key="videos_relink_dossier",
        placeholder="/chemin/vers/les/videos",
    )
    if not dossier:
        return

    dossier_path = Path(dossier)
    if not dossier_path.is_dir():
        st.error(f"« {dossier} » n'est pas un dossier.")
        return

    relinks = V.find_relinks(projet, dossier_path)
    if not relinks:
        st.info("Aucune vidéo correspondante trouvée dans ce dossier.")
        return

    st.dataframe(
        pd.DataFrame([
            {"Session": sid, "Vidéo retrouvée": str(chemin)}
            for sid, chemin in relinks
        ]),
        use_container_width=True, hide_index=True,
    )

    if st.button(f"Re-pointer {len(relinks)} session(s)", key="btn_relink_demander"):
        st.session_state["_confirmer_relink"] = str(dossier_path)

    # Confirmation avant écriture (apply_relinks réécrit metadata.yaml) : le
    # tableau ci-dessus est la preuve de ce qui va changer, ce bouton n'est
    # que la demande — le style suit la double confirmation déjà utilisée
    # pour la suppression de projet (views/projet.py).
    if st.session_state.get("_confirmer_relink") == str(dossier_path):
        st.warning(
            "Réécrire `source_video` dans le `metadata.yaml` de ces "
            "sessions ? Seule cette clé change, le reste de chaque fichier "
            "est préservé tel quel."
        )
        col_annuler, col_confirmer = st.columns(2)
        with col_annuler:
            if st.button("Annuler", key="btn_relink_annuler"):
                st.session_state.pop("_confirmer_relink", None)
                st.rerun()
        with col_confirmer:
            if st.button("Oui, re-pointer", key="btn_relink_confirmer", type="primary"):
                # Recalcul juste avant l'écriture : le dossier a pu changer
                # de contenu entre l'affichage du tableau et ce clic.
                relinks_confirmes = V.find_relinks(projet, dossier_path)
                n = V.apply_relinks(projet, relinks_confirmes)
                st.session_state.pop("_confirmer_relink", None)
                st.cache_data.clear()
                st.toast(f"{n} session(s) re-pointée(s).")
                st.rerun()


def _onglet_sessions(projet: Path) -> None:
    df = S.list_sessions(projet)
    if df.empty:
        st.info(
            "Aucune session dans ce projet pour l'instant — commence par la "
            "page **Données**."
        )
        return

    statuts = dict(zip(df["session_id"], df["vidéo"]))
    session_id = st.selectbox(
        "Session",
        options=list(df["session_id"]),
        format_func=lambda sid: f"{sid} (vidéo : {statuts[sid]})",
        key="videos_session_choisie",
    )

    meta = S.load_metadata(projet, session_id) or {}
    _section_video_session(session_id, meta)
    _section_arenes_session(projet, meta)

    manquantes = list(df.loc[df["vidéo"] == "manque", "session_id"])
    if manquantes:
        st.divider()
        _section_relink(projet, manquantes)


# ============================================================
# Onglet Crop — voie B de l'étape 4 (multi uniquement)
# ============================================================

def _onglet_crop(projet: Path) -> None:
    if project_kind(projet) != "multi":
        st.caption(
            "Projet détecté comme `single` — un seul animal par vidéo. Le "
            "crop par arène (`crop_arenes.py`) n'a de sens que pour isoler "
            "plusieurs animaux filmés dans le même cadre ; cette section ne "
            "s'applique donc pas ici, l'inférence DLC tourne directement sur "
            "la vidéo source (page **Pose (DLC)**)."
        )
        return

    st.markdown(lucide_title("clapperboard", "Crop des arènes"), unsafe_allow_html=True)
    st.caption(
        "Étape 4 du README — pour un projet multi-animal, deux voies "
        "possibles, à choisir une fois pour toutes avant l'inférence DLC :"
    )
    st.markdown(
        "- **Voie A — DLC multi-animal puis split.** `run_dlc_inference.py "
        "--mode superanimal` tourne directement sur la vidéo entière (pas de "
        "crop), puis `assign_arenas.py` sépare la sortie par arène. Plus "
        "rapide dans l'ensemble : un seul passage DLC par vidéo entière.\n"
        "- **Voie B — crop puis DLC single-animal.** Ce que fait cette "
        "section (`crop_arenes.py`), suivi de `run_dlc_inference.py --mode "
        "single-animal` sur chaque vidéo croppée. Sortie plus propre par "
        "arène, et **indispensable** si tu comptes labelliser des frames "
        "pour entraîner ou affiner un modèle DLC — la GUI DLC est bien plus "
        "simple en single-animal."
    )
    st.caption(
        "Le crop n'est donc pas « la » chose à faire par défaut : il ne sert "
        "que si tu choisis la voie B, ou si tu prépares une labellisation "
        "(README §4.2)."
    )

    coords = arena_coords(projet)
    if not coords:
        st.info(
            "Aucune coordonnée d'arène par défaut n'est configurée "
            "(`default_arenes_coords`) — `crop_arenes.py` lira les "
            "coordonnées propres à chaque session dans sa `metadata.yaml` "
            "(`arenes[].coords`). Si elles n'y sont pas non plus, calibre-les "
            "d'abord dans l'onglet **Calibration arènes**."
        )

    choisies, tout = _widgets.selecteur_sessions(projet, cle="videos_crop")
    tout_nouveau = st.checkbox(
        "Seulement les sessions pas encore croppées (`--all-new`)",
        value=False, key="videos_crop_all_new", disabled=tout,
        help="Ignoré si « Toutes les sessions » est coché.",
    )

    pas_de_sessions = not tout and not tout_nouveau and not choisies
    cmd = PL.crop_arenes(
        projet, sessions=choisies,
        all_sessions=tout, all_new=(tout_nouveau and not tout),
    )
    _job.bouton_lancer(
        projet, "Lancer le crop des arènes", cmd,
        cle="btn_videos_crop",
        disabled=pas_de_sessions,
        help=None if not pas_de_sessions else
             "Sélectionne au moins une session, coche « Toutes les sessions » "
             "ou « Seulement les nouvelles ».",
    )
    _job.panneau(projet)
    _job.historique(projet)


# ============================================================
# Page
# ============================================================

def render() -> None:
    projet = require_project()

    st.title("Vidéos & calibration")
    st.caption(
        "Étapes 4 et 6a du pipeline : regarder une session avant de la "
        "traiter, comparer ses caractéristiques réelles à ce que déclare la "
        "metadata, re-pointer une vidéo déplacée, et cropper les arènes pour "
        "la voie B du multi-animal."
    )
    st.caption(f"Projet détecté comme `{project_kind(projet)}`.")

    onglet_sessions, onglet_crop = st.tabs(["Sessions", "Crop"])
    with onglet_sessions:
        _onglet_sessions(projet)
    with onglet_crop:
        _onglet_crop(projet)
