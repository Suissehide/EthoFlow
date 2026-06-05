"""Page Labellisation VAME — assigne des labels éthologiques aux motifs.

Pour un projet × algo donné, on visionne chaque `cluster_video` et on remplit un
YAML `<projet>/analysis/motif_labels_<algo>.yaml` consommable par `analyze_vame.py`.

Deux vues : par motif (lecture + label) et tableau (édition rapide).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from lib.config import ETHOGRAM, vame_projects_root
from lib.labels import labels_path, load_labels, motif_display, save_labels
from lib.vame_projects import (
    discover_projects,
    find_any_motif_video,
    list_algos,
    motif_usage_df,
    parse_algo_n,
    validity_per_session_df,
)


# ============================================================
# Helpers
# ============================================================

def _project_selector() -> tuple[Path | None, str | None]:
    """Sélecteur projet × algo. Renvoie (project, algo_full) ou (None, None)."""
    root = vame_projects_root()
    projects = discover_projects(root)
    if not projects:
        st.warning(
            f"Aucun projet VAME dans `{root}`.\n\n"
            "Change la racine dans la sidebar, ou lance `setup` d'abord depuis la page "
            "**Lancer pipeline**."
        )
        return None, None

    col1, col2 = st.columns([2, 1])
    with col1:
        project = st.selectbox(
            "Projet VAME",
            options=projects,
            format_func=lambda p: p.name,
            key="lm_project",
        )
    with col2:
        algos = list_algos(project)
        if not algos:
            st.error(
                "Pas d'algo détecté dans ce projet. As-tu lancé `segment` ?"
            )
            return project, None
        algo = st.selectbox("Algorithme", options=algos, key="lm_algo")
    return project, algo


def _stats_for_motif(usage_df: pd.DataFrame, motif_id: int) -> dict:
    """Stats agrégées pour un motif : usage moyen, sessions affectées, top session."""
    sub = usage_df[usage_df["motif"] == motif_id]
    if sub.empty:
        return {"mean_pct": 0.0, "n_sessions": 0, "top_session": "—", "top_freq": 0.0}
    return {
        "mean_pct": float(sub["frequency"].mean() * 100),
        "n_sessions": int((sub["count"] > 0).sum()),
        "top_session": str(sub.loc[sub["frequency"].idxmax(), "session"]),
        "top_freq": float(sub["frequency"].max() * 100),
    }


def _autosave_label(project: Path, algo: str, motif_id: int) -> None:
    """Callback on_change : lit le widget et persiste le YAML."""
    key = f"label_input_{motif_id}"
    value = st.session_state.get(key, "").strip()
    current = load_labels(project, algo)
    current[motif_id] = value
    save_labels(project, algo, current)
    st.toast(f"Sauvé : motif {motif_id} = {value or '(vide)'}")


# ============================================================
# Tab "Par motif"
# ============================================================

def _tab_individual(project: Path, algo: str, usage_df: pd.DataFrame, n_motifs: int) -> None:
    labels = load_labels(project, algo)

    if "current_motif" not in st.session_state:
        st.session_state["current_motif"] = 0

    motif_id = st.select_slider(
        "Motif",
        options=list(range(n_motifs)),
        value=min(st.session_state["current_motif"], n_motifs - 1),
        format_func=lambda m: motif_display(m, labels),
        key="lm_slider",
    )
    st.session_state["current_motif"] = motif_id

    col_video, col_side = st.columns([2, 1])

    with col_video:
        video = find_any_motif_video(project, algo, motif_id)
        if video and video.exists():
            st.video(str(video))
            st.caption(f"`{video.relative_to(project)}`")
        else:
            st.warning(
                f"Pas de cluster_video trouvée pour le motif {motif_id}. "
                "Lance `motif-videos` dans la page **Lancer pipeline**."
            )

    with col_side:
        stats = _stats_for_motif(usage_df, motif_id)
        st.metric("Usage moyen", f"{stats['mean_pct']:.2f} %")
        st.metric("Sessions affectées", stats["n_sessions"])
        st.metric(
            "Top session",
            stats["top_session"],
            delta=f"{stats['top_freq']:.1f} %",
        )

        # validity / empty-arena (si dispo)
        validity = validity_per_session_df(project)
        if validity is not None and not validity.empty:
            for col in ("empty_arena_fraction", "frac_empty", "empty_fraction"):
                if col in validity.columns:
                    avg = float(pd.to_numeric(validity[col], errors="coerce").mean() * 100)
                    st.metric("Empty-arena (moy. projet)", f"{avg:.1f} %")
                    break

        st.markdown("---")
        # Le widget est lié à `label_input_<motif_id>` ; on initialise sa valeur.
        widget_key = f"label_input_{motif_id}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = labels.get(motif_id, "")
        st.text_input(
            "Label éthologique",
            key=widget_key,
            on_change=_autosave_label,
            args=(project, algo, motif_id),
            help="Sauvegarde automatique en quittant le champ (Enter ou Tab).",
        )

        with st.expander("Vocabulaire suggéré", expanded=False):
            for cat, terms in ETHOGRAM.items():
                st.caption(cat)
                cols = st.columns(3)
                for i, term in enumerate(terms):
                    if cols[i % 3].button(
                        term,
                        key=f"voc_{motif_id}_{cat}_{term}",
                        use_container_width=True,
                    ):
                        st.session_state[widget_key] = term
                        _autosave_label(project, algo, motif_id)
                        st.rerun()


# ============================================================
# Tab "Vue tableau"
# ============================================================

def _build_table(
    project: Path, algo: str, usage_df: pd.DataFrame, n_motifs: int,
) -> pd.DataFrame:
    """Tableau motif × stats × label."""
    labels = load_labels(project, algo)
    validity = validity_per_session_df(project)

    rows: list[dict] = []
    for motif_id in range(n_motifs):
        stats = _stats_for_motif(usage_df, motif_id)
        empty = None
        if validity is not None and not validity.empty:
            for col in ("empty_arena_fraction", "frac_empty", "empty_fraction"):
                if col in validity.columns:
                    empty = float(pd.to_numeric(validity[col], errors="coerce").mean() * 100)
                    break
        rows.append({
            "motif": motif_id,
            "label": labels.get(motif_id, ""),
            "usage_moyen (%)": round(stats["mean_pct"], 2),
            "top_session": stats["top_session"],
            "top_freq (%)": round(stats["top_freq"], 2),
            "sessions_affectées": stats["n_sessions"],
            "empty_fraction (%)": round(empty, 2) if empty is not None else None,
        })
    return pd.DataFrame(rows)


def _tab_table(project: Path, algo: str, usage_df: pd.DataFrame, n_motifs: int) -> None:
    df = _build_table(project, algo, usage_df, n_motifs)
    edited = st.data_editor(
        df,
        column_config={
            "motif": st.column_config.NumberColumn("motif", disabled=True),
            "label": st.column_config.TextColumn(
                "label", help="Édite et tab/click ailleurs pour sauver."
            ),
            "usage_moyen (%)": st.column_config.NumberColumn(disabled=True),
            "top_session": st.column_config.TextColumn(disabled=True),
            "top_freq (%)": st.column_config.NumberColumn(disabled=True),
            "sessions_affectées": st.column_config.NumberColumn(disabled=True),
            "empty_fraction (%)": st.column_config.NumberColumn(disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key=f"lm_editor_{project.name}_{algo}",
    )

    # Détection des changements de label
    if not edited.equals(df):
        mapping = {
            int(row["motif"]): str(row["label"] or "")
            for _, row in edited.iterrows()
        }
        save_labels(project, algo, mapping)
        st.toast("Labels sauvés")

    # Export / import
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        path = labels_path(project, algo)
        if path.exists():
            st.download_button(
                "Télécharger le YAML",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/x-yaml",
            )
        else:
            st.caption("Pas encore de YAML — édite au moins un label.")
    with col2:
        uploaded = st.file_uploader(
            "Charger un YAML existant (écrase les labels actuels)",
            type=["yaml", "yml"],
            key="lm_upload",
        )
        if uploaded is not None:
            try:
                raw = yaml.safe_load(uploaded.read()) or {}
                mapping = {int(k): str(v) for k, v in raw.items()}
                save_labels(project, algo, mapping)
                st.success(f"{len(mapping)} labels chargés.")
                st.rerun()
            except Exception as e:
                st.error(f"YAML invalide : {e}")


# ============================================================
# Entry
# ============================================================

def render() -> None:
    st.title("Labellisation des motifs VAME")
    st.caption(
        "Pour chaque motif, visionne la cluster_video et attribue un label éthologique. "
        "Sauvegarde dans `<projet>/analysis/motif_labels_<algo>.yaml`, consommé par "
        "`analyze_vame.py --labels`."
    )

    project, algo = _project_selector()
    if project is None or algo is None:
        return

    # Récupère le nombre de motifs depuis le nom d'algo (ex: 'hmm-15' → 15)
    try:
        _, n_motifs = parse_algo_n(algo)
    except ValueError:
        st.error(f"Format d'algo invalide : {algo}")
        return

    usage_df = motif_usage_df(project, algo)
    if usage_df.empty:
        st.info(
            "Aucun `motif_usage_<session>.npy` détecté dans ce projet × algo. "
            "Les stats seront vides — la labellisation reste possible mais à l'aveugle."
        )

    tab_individual, tab_table = st.tabs(["Par motif", "Vue tableau"])
    with tab_individual:
        _tab_individual(project, algo, usage_df, n_motifs)
    with tab_table:
        _tab_table(project, algo, usage_df, n_motifs)
