"""Page Détails session — affiche la metadata et les arènes d'une session."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

from lib.sessions import arenes_dataframe, list_sessions, load_metadata


def render() -> None:
    st.title("Détails d'une session")

    df = list_sessions()
    if df.empty:
        st.info("Pas de session.")
        return

    session = st.selectbox("Session", options=df["session_id"].tolist())
    meta = load_metadata(session)
    if meta is None:
        st.error("metadata.yaml introuvable")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Timepoint", meta.get("timepoint", "—"))
    c2.metric("Date", meta.get("date", "—"))
    c3.metric("FPS", meta.get("camera", {}).get("fps", "—"))
    c4.metric("Trial n°", meta.get("trial_no", "—"))

    st.subheader("Arènes")
    st.dataframe(arenes_dataframe(meta), use_container_width=True, hide_index=True)

    with st.expander("metadata.yaml brut"):
        st.code(
            yaml.dump(meta, allow_unicode=True, sort_keys=False),
            language="yaml",
        )

    sv = meta.get("source_video")
    if sv:
        if Path(sv).exists():
            st.success(f"Vidéo source localisée : `{sv}`")
        else:
            st.warning(f"Vidéo source introuvable : `{sv}`")
