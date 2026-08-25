"""Page Données — étapes 2 et 3 du pipeline.

Étape 2 (README) : remplir l'Excel maître généré par `create_project.py`.
Étape 3 : le transformer en `data/raw/<session>/metadata.yaml` via
`sync_from_excel.py`.

Aucune lecture d'Excel ici : le script est la seule autorité sur le
format (feuille `Sessions` ou `Subjects`/`Trials_Videos`/`Arena_Mapping`,
détection automatique du schéma). La page se contente de localiser le
fichier (`lib.config.excel_path`, qui délègue à la même fonction que le
script), de l'offrir au téléchargement/dépôt, et de construire les
commandes via `lib.pipeline`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import lib.pipeline as PL
from lib.config import excel_path, project_kind, require_project
from lib.icons import lucide_title
from lib.sessions import (
    arenes_dataframe,
    list_sessions,
    load_metadata,
    metadata_fields,
    save_arenes,
    save_metadata_fields,
)
from views import _job
from views._widgets import champ_chemin


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

    chemin_excel = excel_path(projet)

    if chemin_excel is None:
        st.warning(
            f"Aucun Excel trouvé à la racine de `{projet.name}`. Il devrait "
            f"s'appeler `{projet.name}_sessions.xlsx` — relance "
            "`create_project.py` si besoin."
        )
    else:
        st.success(f"Excel trouvé : `{chemin_excel.name}`")
        st.download_button(
            "Télécharger",
            data=chemin_excel.read_bytes(),
            file_name=chemin_excel.name,
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
        cible = chemin_excel or (projet / f"{projet.name}_sessions.xlsx")
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

    return chemin_excel


# ============================================================
# Section 2 : Sync avec aperçu
# ============================================================

def _section_sync(projet: Path, chemin_excel: Path | None) -> None:
    st.markdown(lucide_title("play", "Synchroniser"), unsafe_allow_html=True)
    st.caption(
        "Transforme chaque ligne de l'Excel en `data/raw/<session>/metadata.yaml`. "
        "Lance d'abord l'aperçu — il n'écrit rien — pour repérer un `id` "
        "qui ne correspond à aucun fichier vidéo avant le sync réel."
    )

    videos_dir = champ_chemin(
        "Dossier des vidéos",
        cle="donnees_videos_dir",
        valeur_defaut=st.session_state.get("_donnees_videos_dir", ""),
        placeholder="ex : E:/data/bottom_view/08062026",
        titre_dialogue="Dossier des vidéos à synchroniser",
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
            projet, videos_dir=videos_dir, excel=chemin_excel,
            video_ext=ext or "mp4", dry_run=True,
        )
        _job.bouton_lancer(
            projet, "Aperçu (dry-run)", cmd_apercu,
            cle="btn_sync_apercu", type="secondary",
            disabled=pas_de_videos_dir, help=aide,
        )
    with col_reel:
        cmd_reel = PL.sync_from_excel(
            projet, videos_dir=videos_dir, excel=chemin_excel,
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

    st.dataframe(df, width="stretch", hide_index=True)

    _section_metadata(projet, df["session_id"].tolist())

    if n_video_ok < n_total:
        st.caption(
            f"{n_total - n_video_ok} session(s) sans vidéo localisable — "
            "re-pointe les chemins depuis la page **Vidéos & calibration**."
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

    chemin_excel = _section_excel(projet)
    st.divider()
    _section_sync(projet, chemin_excel)
    st.divider()
    _section_sessions(projet)


# ============================================================
# Metadata d'une session — consultation et édition
# ============================================================

def _section_metadata(projet: Path, sessions: list[str]) -> None:
    """Voir et corriger la metadata d'une session sans rouvrir l'Excel.

    Les colonnes ne sont PAS codées en dur : elles sortent de ce que la
    session contient réellement, parce que `sync_from_excel.py` recopie
    toutes les colonnes de l'Excel, y compris celles que le chercheur
    invente. Une correction ponctuelle (une souris mal saisie, un groupe à
    rectifier) se fait ici ; une correction de fond se fait dans l'Excel
    puis un re-sync.
    """
    if not sessions:
        return

    st.divider()
    st.markdown(lucide_title("file-spreadsheet", "Metadata d'une session"),
                unsafe_allow_html=True)
    st.caption(
        "Édition ponctuelle. Pour une reprise de fond, corrige l'Excel et "
        "relance le sync avec `--overwrite`."
    )

    session = st.selectbox("Session", options=sessions, key="donnees_meta_session")
    meta = load_metadata(projet, session)
    if meta is None:
        st.error(f"`metadata.yaml` introuvable pour `{session}`.")
        return

    champs = metadata_fields(meta)
    if champs:
        df_champs = pd.DataFrame(
            [{"champ": c, "valeur": "" if v is None else str(v)}
             for c, v in champs.items()]
        )
        edite = st.data_editor(
            df_champs,
            column_config={
                "champ": st.column_config.TextColumn("Champ", disabled=True),
                "valeur": st.column_config.TextColumn("Valeur"),
            },
            hide_index=True, width="stretch",
            key=f"donnees_meta_editor_{session}",
        )
        if st.button("Enregistrer la metadata", key="donnees_meta_save"):
            modifs = {
                str(r["champ"]): r["valeur"] for _, r in edite.iterrows()
            }
            if save_metadata_fields(projet, session, modifs):
                st.success(f"Metadata de `{session}` enregistrée.")
                st.rerun()
            else:
                st.error("Écriture impossible.")
    else:
        st.caption("Cette session n'a aucun champ exploitable.")

    arenes = arenes_dataframe(meta)
    if not arenes.empty:
        st.markdown("**Arènes**")
        st.caption(
            "Les coordonnées viennent de `calibrate_arenes.py` et ne sont "
            "pas éditables ici — elles sont préservées à l'enregistrement."
        )
        colonnes_editables = [c for c in arenes.columns if c != "coords"]
        edite_ar = st.data_editor(
            arenes,
            column_config={
                "coords": st.column_config.TextColumn("Coords", disabled=True),
            },
            num_rows="dynamic", hide_index=True, width="stretch",
            key=f"donnees_arenes_editor_{session}",
        )
        if st.button("Enregistrer les arènes", key="donnees_arenes_save"):
            lignes = [
                {c: r[c] for c in colonnes_editables}
                for _, r in edite_ar.iterrows()
            ]
            if save_arenes(projet, session, lignes):
                st.success(f"Arènes de `{session}` enregistrées.")
                st.rerun()
            else:
                st.error("Écriture impossible.")
