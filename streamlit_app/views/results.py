"""Page Résultats — explore les sorties DLC, VAME et analyze_vame.

Trois onglets : output DLC, cluster_videos VAME, analyse statistique
(`<projet>/analysis/`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lib.config import dlc_output_dir, vame_output_dir, vame_projects_root
from lib.sessions import arenes_dataframe, list_sessions, load_metadata
from lib.vame_projects import (
    discover_projects,
    list_algos,
    list_sessions_in_project,
)


# ============================================================
# Tab "Output DLC"
# ============================================================

def _tab_dlc(session: str | None) -> None:
    if not session:
        st.info("Sélectionne une session.")
        return
    dlc_dir = dlc_output_dir() / session
    if not dlc_dir.exists():
        st.warning(f"Pas de sortie DLC pour `{session}` (`{dlc_dir}` absent).")
        return
    files = sorted(dlc_dir.iterdir())
    st.write(f"{len(files)} fichier(s) dans `{dlc_dir}`")
    for f in files[:50]:
        st.text(f.name)
    if len(files) > 50:
        st.caption(f"… (+{len(files) - 50} autres)")


# ============================================================
# Tab "VAME (cluster_videos)"
# ============================================================

def _tab_vame_videos(project: Path | None) -> None:
    if project is None:
        st.info("Sélectionne un projet VAME.")
        return
    algos = list_algos(project)
    if not algos:
        st.warning("Pas d'algo détecté dans ce projet — as-tu lancé `segment` ?")
        return
    algo = st.selectbox("Algo", options=algos, key="res_algo")
    sessions = list_sessions_in_project(project)
    if not sessions:
        st.warning("Pas de session segmentée dans ce projet.")
        return
    session = st.selectbox("Session du projet", options=sessions, key="res_vsession")
    cluster_dir = project / "results" / session / "VAME" / algo / "cluster_videos"
    if not cluster_dir.exists():
        st.warning(f"Pas de cluster_videos pour `{session}` / `{algo}`.")
        return
    videos = sorted(cluster_dir.glob("*.mp4"))
    st.caption(f"{len(videos)} cluster_videos dans `{cluster_dir}`")
    for v in videos:
        with st.expander(v.name, expanded=False):
            st.video(str(v))


# ============================================================
# Tab "Analyse"
# ============================================================

def _show_image_if_exists(path: Path, caption: str | None = None) -> bool:
    if path.exists():
        st.image(str(path), caption=caption or path.name, use_container_width=True)
        return True
    return False


def _csv_preview(path: Path) -> None:
    if not path.exists():
        return
    st.subheader(path.name)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Lecture impossible : {e}")
        return
    st.caption(f"{len(df)} lignes × {len(df.columns)} colonnes")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        f"Télécharger {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
        key=f"dl_{path.name}",
    )


def _tab_analysis(project: Path | None) -> None:
    if project is None:
        st.info("Sélectionne un projet VAME.")
        return
    analysis_dir = project / "analysis"
    if not analysis_dir.exists():
        st.warning(
            f"Pas de dossier `analysis/` dans `{project.name}`. "
            "Lance `analyze_vame` depuis la page **Lancer pipeline**."
        )
        return

    # Heatmap
    st.subheader("Heatmap d'usage")
    if not _show_image_if_exists(analysis_dir / "heatmap_usage.png"):
        st.caption("`heatmap_usage.png` absent.")

    # Plots par condition (pattern : mean_by_*.png, boxplots_top_by_*.png)
    st.subheader("Plots par condition")
    mean_plots = sorted(analysis_dir.glob("mean_by_*.png"))
    box_plots = sorted(analysis_dir.glob("boxplots_top_by_*.png"))
    if not mean_plots and not box_plots:
        # Fallback aux anciens noms
        mean_plots = sorted(analysis_dir.glob("mean_by_condition*.png"))
        box_plots = sorted(analysis_dir.glob("boxplots_top_motifs*.png"))
    for p in mean_plots + box_plots:
        _show_image_if_exists(p)
    if not (mean_plots or box_plots):
        st.caption("Aucun plot `mean_by_*.png` ou `boxplots_top_*.png`.")

    # CSV
    st.subheader("Tables CSV")
    _csv_preview(analysis_dir / "motif_usage_long.csv")
    _csv_preview(analysis_dir / "motif_usage.csv")
    _csv_preview(analysis_dir / "validity_per_session.csv")


# ============================================================
# Entry
# ============================================================

def render() -> None:
    st.title("Résultats")

    # Sélection session ethoflow (pour le tab DLC)
    df = list_sessions()
    if df.empty:
        st.info("Aucune session.")
        return

    col1, col2 = st.columns(2)
    with col1:
        treated = df[df["DLC"] == "OK"]["session_id"].tolist()
        session = st.selectbox(
            "Session EthoFlow (pour les sorties DLC)",
            options=treated or df["session_id"].tolist(),
            key="res_session",
        )
    with col2:
        root = vame_projects_root()
        projects = discover_projects(root)
        project = st.selectbox(
            "Projet VAME (pour les sorties VAME / analyse)",
            options=projects,
            format_func=lambda p: p.name if isinstance(p, Path) else str(p),
            key="res_project",
        ) if projects else None
        if not projects:
            st.caption(f"Aucun projet dans `{root}`.")

    meta = load_metadata(session) if session else None
    if meta:
        with st.expander("Contexte de la session (metadata)"):
            st.dataframe(arenes_dataframe(meta), use_container_width=True, hide_index=True)

    tabs = st.tabs(["Output DLC", "VAME (cluster_videos)", "Analyse"])
    with tabs[0]:
        _tab_dlc(session)
    with tabs[1]:
        _tab_vame_videos(project)
    with tabs[2]:
        _tab_analysis(project)
