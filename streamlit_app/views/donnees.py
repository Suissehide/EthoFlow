"""Page Données — étapes 2 et 3 du pipeline.

Étape 2 (README) : remplir l'Excel maître généré par `create_project.py`.
Étape 3 : le transformer en `data/raw/<session>/metadata.yaml` via
`sync_from_excel.py`.

Aucune lecture d'Excel ici : le script est la seule autorité sur le
format (feuille `Sessions` ou `Subjects`/`Trials_Videos`/`Arena_Mapping`,
détection automatique du schéma). La page se contente de localiser le
fichier (même fonction que le script, `find_project_excel`), de l'offrir
au téléchargement/dépôt, et de construire les commandes via `lib.pipeline`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

import lib.pipeline as PL
from lib.config import SCRIPTS_DIR, project_kind, require_project
from lib.icons import lucide_title
from lib.sessions import list_sessions
from views import _job

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from sync_from_excel import find_project_excel  # noqa: E402


# ============================================================
# Section 1 : Excel maître
# ============================================================

def _section_excel(projet: Path) -> Path | None:
    st.markdown(lucide_title("file-spreadsheet", "Fichier Excel"), unsafe_allow_html=True)
    st.caption(
        "`create_project.py` a généré un classeur starter à la racine du "
        "projet — une ligne par vidéo à remplir, puis à redéposer ici."
    )

    with st.expander("Colonnes attendues", expanded=False):
        st.markdown(
            "- **`id`** (obligatoire) — nom du fichier vidéo sans extension "
            "(`970-M1` → cherche `970-M1.mp4`). C'est la clé unique de la "
            "session, elle devient le nom du dossier sous `data/raw/`.\n"
            "- **`mouse_id`** (recommandée) — identifie l'ANIMAL ; se répète "
            "d'une ligne à l'autre pour un même animal filmé à plusieurs "
            "timepoints dans un design longitudinal.\n"
            "- **`group`** (recommandée) — la variable de comparaison "
            "principale.\n"
            "- **toutes les autres colonnes sont libres** : `sync_from_excel.py` "
            "les recopie telles quelles dans `metadata.yaml`, et chacune "
            "devient un axe de comparaison utilisable dans `analyze_vame.py`. "
            "Inventer une colonne `regime_alimentaire` la fait arriver "
            "jusqu'au bout de la chaîne sans toucher au code.\n\n"
            "Pour un projet multi-animaux, le classeur a trois feuilles "
            "(`Subjects`, `Trials_Videos`, `Arena_Mapping`) au lieu d'une "
            "seule `Sessions` — `sync_from_excel.py` détecte le schéma "
            "tout seul, la page n'a pas à le demander."
        )

    excel_path = find_project_excel(projet)

    if excel_path is None:
        st.warning(
            f"Aucun Excel trouvé à la racine de `{projet.name}`. Il devrait "
            f"s'appeler `{projet.name}_sessions.xlsx` — relance "
            "`create_project.py` si besoin."
        )
    else:
        st.success(f"Excel trouvé : `{excel_path.name}`")
        st.download_button(
            "Télécharger",
            data=excel_path.read_bytes(),
            file_name=excel_path.name,
            key="btn_telecharger_excel",
        )

    st.markdown("**Déposer une version remplie**")

    # Astuce de clé versionnée (comme l'ancien navigateur de fichiers) :
    # après confirmation ou annulation, on incrémente la version pour que
    # Streamlit instancie un widget neuf plutôt que de garder le fichier
    # affiché comme « en attente » indéfiniment.
    version_key = "_donnees_upload_version"
    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    upload = st.file_uploader(
        "Remplace le fichier ci-dessus après confirmation — jamais en silence",
        type=["xlsx"],
        key=f"upload_excel_v{st.session_state[version_key]}",
        label_visibility="collapsed",
    )

    if upload is not None:
        cible = excel_path or (projet / f"{projet.name}_sessions.xlsx")
        st.warning(
            f"Écraser `{cible.name}` avec `{upload.name}` ? "
            "Cette action est irréversible."
        )
        col_annuler, col_confirmer = st.columns(2)
        with col_annuler:
            if st.button("Annuler", key="btn_annuler_upload_excel"):
                st.session_state[version_key] += 1
                st.rerun()
        with col_confirmer:
            if st.button("Oui, écraser", key="btn_confirmer_upload_excel", type="primary"):
                cible.write_bytes(upload.getvalue())
                st.session_state[version_key] += 1
                st.toast(f"`{cible.name}` remplacé")
                st.rerun()

    return excel_path


# ============================================================
# Section 2 : Sync avec aperçu
# ============================================================

def _section_sync(projet: Path, excel_path: Path | None) -> None:
    st.markdown(lucide_title("play", "Synchroniser"), unsafe_allow_html=True)
    st.caption(
        "Transforme chaque ligne de l'Excel en `data/raw/<session>/metadata.yaml`. "
        "Lance d'abord l'aperçu — il n'écrit rien — pour repérer un `id` "
        "qui ne correspond à aucun fichier vidéo avant le sync réel."
    )

    videos_dir = st.text_input(
        "Dossier des vidéos",
        value=st.session_state.get("_donnees_videos_dir", ""),
        placeholder="ex : E:/data/bottom_view/08062026",
        key="donnees_videos_dir",
    )
    st.session_state["_donnees_videos_dir"] = videos_dir

    col_ext, col_overwrite = st.columns([1, 2])
    with col_ext:
        ext = st.text_input("Extension vidéo", value="mp4", key="donnees_video_ext")
    with col_overwrite:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        overwrite = st.checkbox(
            "Écraser les metadata déjà générées (--overwrite)",
            value=False,
            key="donnees_overwrite",
        )

    pas_de_videos_dir = not videos_dir.strip()
    aide = "Renseigne le dossier des vidéos d'abord." if pas_de_videos_dir else None

    col_apercu, col_reel = st.columns(2)
    with col_apercu:
        cmd_apercu = PL.sync_from_excel(
            projet, videos_dir=videos_dir, excel=excel_path,
            video_ext=ext or "mp4", dry_run=True,
        )
        _job.bouton_lancer(
            projet, "Aperçu (dry-run)", cmd_apercu,
            cle="btn_sync_apercu", type="secondary",
            disabled=pas_de_videos_dir, help=aide,
        )
    with col_reel:
        cmd_reel = PL.sync_from_excel(
            projet, videos_dir=videos_dir, excel=excel_path,
            video_ext=ext or "mp4", overwrite=overwrite,
        )
        _job.bouton_lancer(
            projet, "Synchroniser", cmd_reel,
            cle="btn_sync_reel", type="primary",
            disabled=pas_de_videos_dir, help=aide,
        )

    _job.panneau(projet)
    _job.historique(projet)


# ============================================================
# Section 3 : Sessions synchronisées
# ============================================================

def _section_sessions(projet: Path) -> None:
    st.markdown(lucide_title("layout-dashboard", "Sessions synchronisées"), unsafe_allow_html=True)

    df = list_sessions(projet)
    if df.empty:
        st.info("Aucune session pour l'instant — lance le sync ci-dessus.")
        return

    if project_kind(projet) != "multi":
        df = df.drop(columns=["split"])

    n_total = len(df)
    n_video_ok = int((df["vidéo"] == "OK").sum())

    col1, col2 = st.columns(2)
    col1.metric("Sessions", n_total)
    col2.metric("Vidéos localisées", f"{n_video_ok}/{n_total}")

    st.dataframe(df, use_container_width=True, hide_index=True)

    if n_video_ok < n_total:
        st.caption(
            f"{n_total - n_video_ok} session(s) sans vidéo localisable — "
            "re-pointe les chemins depuis la page **Vidéos & calibration** "
            "(à venir)."
        )


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    projet = require_project()

    st.title("Données")
    st.caption(
        "Étapes 2 et 3 du pipeline : remplir l'Excel maître, puis en tirer "
        "les metadata.yaml des sessions."
    )

    excel_path = _section_excel(projet)
    st.divider()
    _section_sync(projet, excel_path)
    st.divider()
    _section_sessions(projet)
