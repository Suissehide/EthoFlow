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

4. **Calibrer au clic.** `scripts/calibrate_arenes.py` et
   `scripts/calibrate_scale.py` ouvrent une fenêtre OpenCV et demandent de
   cliquer — impossible à travers un serveur web. `streamlit_image_coordinates`
   reproduit le geste dans le navigateur ; l'écriture du résultat dans
   `configs/pipeline_config.yaml` passe par `lib.project.set_arena_coords`/
   `set_px_per_cm`, qui délèguent aux fonctions des scripts
   (`save_coords_default`/`write_scale`) — la vue ne réimplémente jamais
   la sérialisation YAML.

Quatre onglets : ``Sessions``, ``Crop``, ``Calibration arènes``,
``Échelle px/cm``.

Aucune commande n'est exécutée ici : `lib.pipeline` construit,
`views._job.bouton_lancer` lance via `lib.runner`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import lib.pipeline as PL
from lib.config import arena_coords, project_kind, px_per_cm, require_project, set_arena_coords, set_px_per_cm
from lib.icons import lucide_title
from lib import sessions as S
from lib import video as V
from views import _job, _widgets

# Labels des arènes dans l'ordre où elles sont cliquées — `crop_arenes.py:130`
# fait `x, y, w, h = coords`, donc l'ordre des clés n'a pas d'importance pour
# le CLI, mais A1→A4 est la convention du reste du projet (`calibrate_arenes.py`,
# metadata `arenes[].id`).
_LABELS_ARENES = ["A1", "A2", "A3", "A4"]

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
                    st.image(png, caption="Première frame", width="stretch")
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
                width="stretch", hide_index=True,
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
            width="stretch", hide_index=True,
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
        st.dataframe(df_arenes, width="stretch", hide_index=True)


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
        width="stretch", hide_index=True,
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
# Sélection commune session + frame (onglets Calibration arènes / Échelle)
# ============================================================

def _sessions_avec_video(projet: Path) -> list[str]:
    """Sessions dont `list_sessions` déclare une vidéo source valide (`vidéo == "OK"`)."""
    df = S.list_sessions(projet)
    if df.empty:
        return []
    return [sid for sid, statut in zip(df["session_id"], df["vidéo"]) if statut == "OK"]


def _choisir_frame_session(projet: Path, *, cle: str) -> tuple[str, np.ndarray] | None:
    """Sélecteur session + index de frame ; renvoie `(identifiant_source, frame_BGR)` ou `None`.

    `identifiant_source` sert de clé de remise à zéro de l'état de clics
    (nouvelle session ou nouvelle frame = anciens clics sans rapport).
    """
    sessions = _sessions_avec_video(projet)
    if not sessions:
        st.info(
            "Aucune session avec une vidéo source valide pour l'instant — "
            "voir l'onglet **Sessions** pour re-pointer une vidéo manquante."
        )
        return None

    session_id = st.selectbox("Session", options=sessions, key=f"{cle}_session")
    meta = S.load_metadata(projet, session_id) or {}
    chemin = Path(meta["source_video"])
    info = _probe_cache(str(chemin))
    if not info.n_frames:
        st.warning(f"`{chemin}` : nombre de frames illisible, impossible d'en extraire une.")
        return None

    defaut = min(info.n_frames // 2, info.n_frames - 1)
    frame_idx = st.number_input(
        "Index de la frame", min_value=0, max_value=info.n_frames - 1,
        value=defaut, key=f"{cle}_frame",
        help=f"Vidéo de {info.n_frames} frames.",
    )
    frame = V.grab_frame(chemin, index=int(frame_idx))
    if frame is None:
        st.warning(f"Impossible d'extraire la frame {frame_idx} de `{chemin}`.")
        return None
    return f"session:{chemin}#{frame_idx}", frame


# ============================================================
# Onglet Calibration arènes — deux clics par arène, ajustement au pixel
# ============================================================

def _reset_arenes_clics() -> None:
    st.session_state["calib_arenes_rects"] = []
    st.session_state["calib_arenes_premier_clic"] = None
    for label in _LABELS_ARENES:
        for champ in ("x", "y", "w", "h"):
            st.session_state.pop(f"calib_arenes_{label}_{champ}", None)


def _onglet_calibration_arenes(projet: Path) -> None:
    st.markdown(lucide_title("waypoints", "Calibration des arènes"), unsafe_allow_html=True)
    st.caption(
        "Deux clics par arène — les deux coins opposés du rectangle — dans "
        "l'ordre A1 → A2 → A3 → A4. Écrit `default_arenes_coords` dans "
        "`configs/pipeline_config.yaml` (via `calibrate_arenes.save_coords_default`), "
        "utilisé par `crop_arenes.py`/`assign_arenas.py` comme repli pour "
        "toute session sans coordonnées propres dans sa `metadata.yaml`."
    )

    choix = _choisir_frame_session(projet, cle="calib_arenes")
    if choix is None:
        return
    source_actuelle, frame = choix

    if st.session_state.get("calib_arenes_source") != source_actuelle:
        st.session_state["calib_arenes_source"] = source_actuelle
        _reset_arenes_clics()

    rects: list[list[int]] = st.session_state.setdefault("calib_arenes_rects", [])
    coords_actuelles = {_LABELS_ARENES[i]: r for i, r in enumerate(rects)}

    frame_rgb = V.to_rgb(frame)
    apercu = V.draw_arenas(frame_rgb, coords_actuelles) if coords_actuelles else frame_rgb

    if len(rects) >= len(_LABELS_ARENES):
        st.success(
            f"{len(_LABELS_ARENES)} arènes définies. Ajuste-les au pixel "
            "ci-dessous si besoin, puis enregistre."
        )
    else:
        label = _LABELS_ARENES[len(rects)]
        etape = "premier coin" if st.session_state.get("calib_arenes_premier_clic") is None else "coin opposé"
        st.caption(f"Clique le **{etape}** de l'arène **{label}** ({len(rects)}/{len(_LABELS_ARENES)}).")

    valeur = streamlit_image_coordinates(apercu, key=f"calib_arenes_clic__{source_actuelle}")

    if (valeur is not None and len(rects) < len(_LABELS_ARENES)
            and valeur.get("unix_time") != st.session_state.get("calib_arenes_dernier_clic")):
        st.session_state["calib_arenes_dernier_clic"] = valeur.get("unix_time")
        point = (int(valeur["x"]), int(valeur["y"]))
        premier = st.session_state.get("calib_arenes_premier_clic")
        if premier is None:
            st.session_state["calib_arenes_premier_clic"] = point
        else:
            rect = V.rect_from_two_points(premier, point)
            if rect[2] == 0 or rect[3] == 0:
                # Ruling R20.1 : paire dégénérée — même pixel, ou même x/y
                # seul — refusée à la source. Élargir min_value=1 → 0 sur le
                # number_input d'ajustement laisserait enregistrer une arène
                # d'aire nulle ; le premier clic reste enregistré, pas besoin
                # de tout recommencer l'arène.
                st.error(
                    f"Les deux clics de l'arène **{_LABELS_ARENES[len(rects)]}** "
                    "définissent un rectangle de largeur ou hauteur nulle "
                    "— reclique un coin opposé à un autre endroit."
                )
            else:
                rects.append(rect)
                st.session_state["calib_arenes_premier_clic"] = None
                st.rerun()  # redessine immédiatement avec le rectangle qui vient d'être posé

    if st.button("Recommencer les clics", key="calib_arenes_recommencer", disabled=not rects):
        _reset_arenes_clics()
        st.rerun()

    if rects:
        st.markdown(lucide_title("settings", "Ajustement fin (pixels)"), unsafe_allow_html=True)
        st.caption(
            "Recliquer précisément est pénible — corrige de quelques "
            "pixels ici plutôt que de recommencer."
        )
        for i, rect in enumerate(rects):
            label = _LABELS_ARENES[i]
            st.caption(f"**{label}**")
            col_x, col_y, col_w, col_h = st.columns(4)
            x = col_x.number_input("x", value=int(rect[0]), step=1, key=f"calib_arenes_{label}_x")
            y = col_y.number_input("y", value=int(rect[1]), step=1, key=f"calib_arenes_{label}_y")
            w = col_w.number_input("largeur", value=int(rect[2]), min_value=1, step=1, key=f"calib_arenes_{label}_w")
            h = col_h.number_input("hauteur", value=int(rect[3]), min_value=1, step=1, key=f"calib_arenes_{label}_h")
            rects[i] = [int(x), int(y), int(w), int(h)]

        if len(rects) == len(_LABELS_ARENES):
            if st.button("Enregistrer les 4 arènes", key="calib_arenes_enregistrer", type="primary"):
                coords = {_LABELS_ARENES[i]: rects[i] for i in range(len(_LABELS_ARENES))}
                set_arena_coords(projet, coords)
                st.cache_data.clear()
                st.toast("Coordonnées des 4 arènes enregistrées dans pipeline_config.yaml.")
        else:
            st.caption(
                f"{len(rects)}/{len(_LABELS_ARENES)} arènes définies — "
                f"encore {len(_LABELS_ARENES) - len(rects)} avant de pouvoir enregistrer."
            )


# ============================================================
# Onglet Échelle px/cm — deux clics sur une distance connue, ou saisie directe
# ============================================================

def _reset_echelle_clics() -> None:
    st.session_state["echelle_points"] = []


def _onglet_echelle(projet: Path) -> None:
    st.markdown(lucide_title("scan-line", "Échelle px/cm"), unsafe_allow_html=True)
    st.caption(
        "Nécessaire pour détecter les vitesses aberrantes dans "
        "`prepare_vame_input_custom.py` — sans échelle, un déplacement en "
        "pixels ne peut pas être converti en vitesse réelle."
    )
    st.info(
        "**Conseil du README** : photographie une règle plutôt que de "
        "mesurer les dimensions de l'arène — plus l'objet mesuré est "
        "grand, plus la distorsion de lentille fausse la conversion "
        "pixels → centimètres."
    )

    actuelle = px_per_cm(projet)
    if actuelle:
        st.caption(
            f"Échelle actuellement enregistrée : **{actuelle:.3f} px/cm** "
            f"(1 px = {10 / actuelle:.2f} mm)."
        )

    source = st.radio(
        "Source de l'image à cliquer",
        ["Frame d'une vidéo de session", "Photo importée (règle)"],
        key="echelle_source_mode", horizontal=True,
    )

    frame = None
    source_actuelle: str | None = None
    if source == "Frame d'une vidéo de session":
        choix = _choisir_frame_session(projet, cle="echelle")
        if choix is not None:
            source_actuelle, frame = choix
    else:
        fichier = st.file_uploader(
            "Photo d'une règle (ou de tout objet de longueur connue)",
            type=["png", "jpg", "jpeg"], key="echelle_upload",
        )
        if fichier is not None:
            octets = np.frombuffer(fichier.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(octets, cv2.IMREAD_COLOR)
            if frame is None:
                st.error(f"« {fichier.name} » n'a pas pu être lu comme image.")
            else:
                source_actuelle = f"image:{fichier.name}#{fichier.size}"

    if frame is not None and source_actuelle is not None:
        if st.session_state.get("echelle_source") != source_actuelle:
            st.session_state["echelle_source"] = source_actuelle
            _reset_echelle_clics()

        points: list[tuple[int, int]] = st.session_state.setdefault("echelle_points", [])
        frame_rgb = V.to_rgb(frame)
        apercu = V.draw_scale_line(frame_rgb, points[0], points[1]) if len(points) == 2 else frame_rgb

        if len(points) < 2:
            etape = "première" if not points else "seconde"
            st.caption(f"Clique la **{etape}** extrémité de la distance connue ({len(points)}/2).")

        valeur = streamlit_image_coordinates(apercu, key=f"echelle_clic__{source_actuelle}")

        if (valeur is not None and len(points) < 2
                and valeur.get("unix_time") != st.session_state.get("echelle_dernier_clic")):
            st.session_state["echelle_dernier_clic"] = valeur.get("unix_time")
            points.append((int(valeur["x"]), int(valeur["y"])))
            if len(points) == 2:
                st.rerun()  # redessine immédiatement le segment

        if st.button("Recommencer les clics", key="echelle_recommencer", disabled=not points):
            _reset_echelle_clics()
            st.rerun()

        if len(points) == 2:
            distance_px = V.distance_from_two_points(points[0], points[1])
            st.write(f"Distance mesurée : **{distance_px:.1f} px**")
            if distance_px <= 0:
                # Même racine que R20.1 côté arènes : deux clics au même
                # pixel donnent une distance nulle, donc un px_per_cm nul —
                # silencieux dans pipeline_config.yaml, il casserait sans
                # message le filtre de vitesses aberrantes du nettoyage
                # VAME. Refusé avant tout calcul, pas seulement avant
                # l'enregistrement (une division par une échelle nulle
                # plante déjà l'affichage « 1 px = X mm » juste en dessous).
                st.error(
                    "Les deux clics sont au même endroit — distance nulle, "
                    "impossible d'en tirer une échelle. Reclique deux "
                    "points distincts (bouton « Recommencer les clics » "
                    "ci-dessus)."
                )
            else:
                known_cm = st.number_input(
                    "Distance réelle entre les deux points cliqués (cm)",
                    min_value=0.01, value=10.0, step=0.5, key="echelle_known_cm",
                )
                valeur_calculee = distance_px / known_cm
                st.write(
                    f"Échelle calculée : **{valeur_calculee:.3f} px/cm** "
                    f"(1 px = {10 / valeur_calculee:.2f} mm)."
                )
                if st.button("Enregistrer cette échelle", key="echelle_enregistrer_clics", type="primary"):
                    set_px_per_cm(projet, valeur_calculee)
                    st.toast(f"px_per_cm = {valeur_calculee:.3f} enregistré dans pipeline_config.yaml.")
    else:
        st.caption(
            "Choisis une source d'image ci-dessus pour calibrer par les "
            "clics, ou saisis directement une valeur déjà connue plus bas."
        )

    st.divider()
    st.markdown(
        lucide_title("save", "Saisie directe d'une valeur déjà connue"),
        unsafe_allow_html=True,
    )
    st.caption("Équivalent de `calibrate_scale.py --set <valeur>` : écrit px_per_cm sans clic.")
    valeur_directe = st.number_input(
        "px/cm connu", min_value=0.0, value=0.0, step=0.1, key="echelle_valeur_directe",
    )
    if st.button(
        "Enregistrer cette valeur", key="echelle_enregistrer_directe",
        disabled=valeur_directe <= 0,
    ):
        set_px_per_cm(projet, valeur_directe)
        st.toast(f"px_per_cm = {valeur_directe:.3f} enregistré dans pipeline_config.yaml.")


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

    onglet_sessions, onglet_crop, onglet_calibration_arenes, onglet_echelle = st.tabs(
        ["Sessions", "Crop", "Calibration arènes", "Échelle px/cm"]
    )
    with onglet_sessions:
        _onglet_sessions(projet)
    with onglet_crop:
        _onglet_crop(projet)
    with onglet_calibration_arenes:
        _onglet_calibration_arenes(projet)
    with onglet_echelle:
        _onglet_echelle(projet)
