"""Page Données — visualiser/éditer les metadata.yaml et importer depuis Excel."""
from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from lib.config import ROOT, SCRIPTS_DIR, current_project_name, data_root, raw_dir
from lib.icons import ACCENT, lucide_html, lucide_title
from lib.sessions import list_sessions


# ============================================================
# File / folder browser (réutilisable)
# ============================================================

def _file_picker(label: str, default: str, key: str, extensions: list[str] | None = None) -> str:
    """Champ texte + navigateur de fichiers intégré."""
    persist_key = f"{key}_persisted"
    version_key = f"{key}_version"
    if persist_key not in st.session_state:
        st.session_state[persist_key] = default
    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    value = st.session_state[persist_key]
    # Version dans la key du widget pour forcer un widget frais après sélection
    widget_key = f"{key}_input_v{st.session_state[version_key]}"

    col1, col2 = st.columns([4, 1])
    current = col1.text_input(label, value=value, key=widget_key)
    # Saisie manuelle → mettre à jour la persistance
    if current != value:
        st.session_state[persist_key] = current

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Parcourir", key=f"{key}_browse"):
            st.session_state[f"{key}_browsing"] = True

    if st.session_state.get(f"{key}_browsing", False):
        selected = _browse(current, key, extensions)
        if selected is not None:
            st.session_state[f"{key}_browsing"] = False
            st.session_state[persist_key] = selected
            # Incrémenter la version pour forcer un nouveau widget
            st.session_state[version_key] += 1
            st.rerun()
        if st.button("Annuler", key=f"{key}_cancel"):
            st.session_state[f"{key}_browsing"] = False
            st.rerun()

    return st.session_state[persist_key]


def _browse(start_path: str, key: str, extensions: list[str] | None = None) -> str | None:
    """Navigateur de fichiers/dossiers intégré."""
    p = Path(start_path)
    current_dir = p.parent if p.is_file() else p
    if not current_dir.is_dir():
        current_dir = Path.home()

    browse_key = f"{key}_browse_dir"
    if browse_key in st.session_state:
        current_dir = Path(st.session_state[browse_key])

    st.markdown(f"**Dossier :** `{current_dir}`")

    col_up, col_select = st.columns(2)
    with col_up:
        if current_dir.parent != current_dir:
            if st.button("Dossier parent", key=f"{key}_parent"):
                st.session_state[browse_key] = str(current_dir.parent)
                st.rerun()
    with col_select:
        if extensions is None:
            if st.button("Choisir ce dossier", key=f"{key}_select", type="primary"):
                if browse_key in st.session_state:
                    del st.session_state[browse_key]
                return str(current_dir)

    try:
        entries = sorted(current_dir.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        st.warning("Accès refusé.")
        return None

    for entry in entries[:40]:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if st.button(f"{entry.name}/", key=f"{key}_e_{entry.name}"):
                st.session_state[browse_key] = str(entry)
                st.rerun()
        elif extensions is None or entry.suffix.lower() in extensions:
            if st.button(entry.name, key=f"{key}_e_{entry.name}"):
                if browse_key in st.session_state:
                    del st.session_state[browse_key]
                return str(entry)

    return None


# ============================================================
# Section: Import Excel (optionnel, en haut)
# ============================================================

def _section_import_excel() -> None:
    st.markdown(lucide_title("folder-open", "Importer depuis Excel"), unsafe_allow_html=True)
    st.caption(
        "Optionnel — génère automatiquement les `metadata.yaml` depuis le fichier Excel maître. "
        "Tu peux aussi créer et éditer les metadata manuellement ci-dessous."
    )

    with st.expander("Configurer l'import Excel", expanded=False):
        default_excel = ROOT.parent / "data" / "OpenField_trials_CDUPLAA.xlsx"
        default_videos = ROOT.parent / "data"

        excel_path = _file_picker(
            "Fichier Excel",
            str(default_excel),
            "sync_excel",
            extensions=[".xlsx", ".xls"],
        )
        videos_dir = _file_picker(
            "Dossier des vidéos (.mp4)",
            str(default_videos),
            "sync_videos",
            extensions=None,
        )
        dry_run = st.checkbox("Dry-run (afficher sans écrire)", value=False)

        if st.button("Lancer le sync", type="primary", key="btn_sync"):
            cmd = [
                sys.executable, str(SCRIPTS_DIR / "sync_from_excel.py"),
                "--excel", excel_path,
                "--videos-dir", videos_dir,
            ]
            # Passer le projet courant au script
            project_path = st.session_state.get("current_project_path")
            if project_path:
                cmd.extend(["--project-dir", project_path])
            if dry_run:
                cmd.append("--dry-run")

            with st.status("Sync en cours...", expanded=True) as status:
                result = subprocess.run(cmd, capture_output=True, text=True)
                st.code(result.stdout or "(stdout vide)")
                if result.stderr:
                    st.error(result.stderr)
                ok = result.returncode == 0
                status.update(
                    label="Sync terminé" if ok else "Sync échoué",
                    state="complete" if ok else "error",
                )


# ============================================================
# Section: Metadata viewer / editor
# ============================================================

def _section_metadata() -> None:
    st.markdown(lucide_title("file-spreadsheet", "Metadata des sessions"), unsafe_allow_html=True)

    df = list_sessions()
    sessions_list = df["session_id"].tolist() if not df.empty else []

    # Ajouter les sessions créées dans cette session streamlit (pas encore sauvegardées)
    pending = st.session_state.get("_pending_sessions", [])
    all_sessions = sessions_list + [s for s in pending if s not in sessions_list]

    # ---- Sélection / création de session ----
    new_id = st.text_input(
        "Nouvelle session",
        placeholder="Tape un identifiant et appuie sur Entrée",
        key="new_session_id",
    )
    # Si l'utilisateur tape un nouvel id, l'ajouter à la liste
    if new_id and new_id.strip() and new_id.strip() not in all_sessions:
        clean_id = new_id.strip()
        if "_pending_sessions" not in st.session_state:
            st.session_state["_pending_sessions"] = []
        st.session_state["_pending_sessions"].append(clean_id)
        all_sessions.append(clean_id)
        # Sélectionner automatiquement la nouvelle session
        st.session_state["_select_session"] = clean_id
        st.session_state.pop("new_session_id", None)
        st.rerun()

    # Forcer la sélection si on vient de créer une session
    force_select = st.session_state.pop("_select_session", None)

    if not all_sessions:
        st.info("Tape un identifiant ci-dessus pour créer ta première session.")
        return

    default_idx = 0
    if force_select and force_select in all_sessions:
        default_idx = all_sessions.index(force_select)

    # Style boutons session
    st.markdown(
        """<style>
        .st-key-btn_del_session button {
            background: transparent !important;
            color: #ef4444 !important;
            border: 1px solid #ef4444 !important;
        }
        .st-key-btn_del_session button:hover {
            background: rgba(239,68,68,0.1) !important;
        }
        .st-key-btn_dup_confirm button {
            background: #3b82f6 !important;
            color: #fff !important;
            border: none !important;
        }
        .st-key-btn_dup_confirm button:hover {
            background: #2563eb !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    col_sel, col_dup, col_del = st.columns([4, 1, 1])
    with col_sel:
        session = st.selectbox("Session", options=all_sessions, index=default_idx, key="meta_session_sel")
    with col_dup:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Dupliquer", key="btn_dup_session"):
            st.session_state["_dup_session"] = session
    with col_del:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Supprimer", key="btn_del_session"):
            st.session_state["_del_session"] = session

    # ---- Dupliquer ----
    if st.session_state.get("_dup_session") == session:
        dup_name = st.text_input("Nom de la copie", value=f"{session}-copie", key="dup_name")
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("Annuler", key="btn_dup_cancel"):
                st.session_state.pop("_dup_session", None)
                st.rerun()
        with c2:
            if st.button("Confirmer", key="btn_dup_confirm", type="primary"):
                if dup_name and dup_name.strip():
                    clean_dup = dup_name.strip()
                    src = raw_dir() / session / "metadata.yaml"
                    if src.exists():
                        import shutil
                        dst_dir = raw_dir() / clean_dup
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst_dir / "metadata.yaml")
                        with open(dst_dir / "metadata.yaml") as f:
                            dup_meta = yaml.safe_load(f) or {}
                        dup_meta["session_id"] = clean_dup
                        with open(dst_dir / "metadata.yaml", "w") as f:
                            yaml.dump(dup_meta, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
                    st.session_state.pop("_dup_session", None)
                    st.session_state["_select_session"] = clean_dup
                    st.toast(f"Session '{clean_dup}' créée depuis '{session}'")
                    st.rerun()

    # ---- Supprimer ----
    if st.session_state.get("_del_session") == session:
        st.warning(f"Supprimer la session **{session}** et son metadata ? Cette action est irréversible.")
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("Annuler", key="btn_del_cancel"):
                st.session_state.pop("_del_session", None)
                st.rerun()
        with c2:
            if st.button("Oui, supprimer", key="btn_del_confirm", type="primary"):
                import shutil
                target = raw_dir() / session
                if target.exists():
                    shutil.rmtree(target)
                if "_pending_sessions" in st.session_state and session in st.session_state["_pending_sessions"]:
                    st.session_state["_pending_sessions"].remove(session)
                st.session_state.pop("_del_session", None)
                st.toast(f"Session '{session}' supprimée")
                st.rerun()

    # ---- Charger ou initialiser le metadata ----
    meta_path = raw_dir() / session / "metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
    else:
        meta = {
            "session_id": session,
            "date": date.today().strftime("%Y-%m-%d"),
            "source_video": None,
            "camera": {"fps": 25, "width": 1280, "height": 1024},
            "arenes": [],
            "notes": "",
        }

    # ---- Infos générales (éditable) ----
    st.markdown("---")
    st.markdown("**Informations générales**")
    col1, col2 = st.columns(2)

    # Date picker en français
    date_raw = meta.get("date", "")
    try:
        date_val = datetime.strptime(str(date_raw).split(" ")[0], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        date_val = date.today()

    picked_date = col1.date_input(
        "Date",
        value=date_val,
        format="DD/MM/YYYY",
        key="meta_date",
    )
    meta["date"] = picked_date.strftime("%Y-%m-%d")

    camera = meta.get("camera", {})
    camera["fps"] = col2.number_input("FPS", value=int(camera.get("fps", 25)), min_value=1, key="meta_fps")

    # ---- Notes ----
    meta["notes"] = st.text_area("Notes", value=meta.get("notes", ""), key="meta_notes")

    # ---- Dossier vidéos ----
    # Réinitialiser le picker quand on change de session
    if st.session_state.get("_meta_videos_dir_session") != session:
        st.session_state["_meta_videos_dir_session"] = session
        st.session_state.pop("meta_videos_dir_persisted", None)
        st.session_state.pop("meta_videos_dir_input", None)

    if meta.get("videos_dir"):
        videos_dir_default = meta["videos_dir"]
    elif meta.get("source_video"):
        p = Path(meta["source_video"]).parent
        videos_dir_default = str(p) if p.is_dir() else str(data_root())
    else:
        videos_dir_default = str(data_root())

    videos_dir = _file_picker(
        "Dossier des vidéos",
        videos_dir_default,
        "meta_videos_dir",
        extensions=None,
    )
    meta["videos_dir"] = videos_dir if videos_dir.strip() else None

    # Lister les vidéos (premier niveau uniquement)
    found_videos: list[str] = []
    if videos_dir and Path(videos_dir).is_dir():
        video_exts = {".mp4", ".avi", ".mkv", ".mov"}
        found_videos = sorted(
            f.name for f in Path(videos_dir).iterdir()
            if f.is_file() and f.suffix.lower() in video_exts
        )
        if found_videos:
            st.success(f"{len(found_videos)} vidéo(s) trouvée(s)")
        else:
            st.warning("Aucune vidéo trouvée dans ce dossier")
    elif videos_dir and videos_dir.strip():
        st.warning("Dossier introuvable")

    # ---- Arènes — groupées par vidéo ----
    st.markdown("**Arènes**")
    arenes = meta.get("arenes", [])

    # Construire le DataFrame — vidéo en première colonne
    arena_rows = [
        {
            "video": a.get("video", ""),
            "id": a.get("id", ""),
            "mouse_id": a.get("mouse_id"),
            "condition": a.get("condition", ""),
            "stress": bool(a.get("stress", False)),
            "angii": bool(a.get("angii", False)),
        }
        for a in arenes
    ]
    if not arena_rows:
        arena_rows = [{
            "video": found_videos[0] if found_videos else "",
            "id": "A1", "mouse_id": None, "condition": "",
            "stress": False, "angii": False,
        }]

    arena_df = pd.DataFrame(arena_rows)
    arena_df["stress"] = arena_df["stress"].astype(bool)
    arena_df["angii"] = arena_df["angii"].astype(bool)
    # Trier par vidéo puis arène pour regrouper
    arena_df = arena_df.sort_values(["video", "id"]).reset_index(drop=True)

    edited = st.data_editor(
        arena_df,
        column_order=["video", "id", "mouse_id", "condition", "stress", "angii"],
        column_config={
            "video": st.column_config.SelectboxColumn(
                "Vidéo",
                options=found_videos,
                help="Vidéo source contenant cette arène",
                width="medium",
            ) if found_videos else st.column_config.TextColumn("Vidéo", width="medium"),
            "id": st.column_config.TextColumn("Arène", width="small"),
            "mouse_id": st.column_config.NumberColumn("MouseID", width="small"),
            "condition": st.column_config.TextColumn("Condition"),
            "stress": st.column_config.CheckboxColumn("Stress"),
            "angii": st.column_config.CheckboxColumn("ANGII"),
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=f"arena_editor_{session}",
    )

    # Re-trier après édition et construire les arènes avec mouse_trial_code auto
    edited = edited.sort_values(["video", "id"]).reset_index(drop=True)

    # Palette de couleurs subtiles par vidéo
    _PALETTE = [
        "rgba(99,102,241,0.08)", "rgba(16,185,129,0.08)", "rgba(245,158,11,0.08)",
        "rgba(239,68,68,0.08)", "rgba(139,92,246,0.08)", "rgba(6,182,212,0.08)",
        "rgba(236,72,153,0.08)", "rgba(34,197,94,0.08)", "rgba(251,146,60,0.08)",
        "rgba(59,130,246,0.08)",
    ]
    _BORDER_PALETTE = [
        "#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
        "#06b6d4", "#ec4899", "#22c55e", "#fb923c", "#3b82f6",
    ]
    unique_videos = [v for v in edited["video"].unique() if v]
    video_colors = {v: (_PALETTE[i % len(_PALETTE)], _BORDER_PALETTE[i % len(_BORDER_PALETTE)]) for i, v in enumerate(unique_videos)}

    # Affichage groupé par vidéo avec couleurs + mouse_trial_code auto-généré
    new_arenes = []
    if not edited.empty:
        rows_html = ""
        current_video = None
        for _, row in edited.iterrows():
            video = row["video"] if pd.notna(row.get("video")) and row["video"] else None
            arena_id = row["id"] if pd.notna(row.get("id")) and row["id"] else ""
            mouse_id = int(row["mouse_id"]) if pd.notna(row.get("mouse_id")) else None

            # Ignorer les lignes vides dans le résumé
            if not video and not arena_id and mouse_id is None:
                continue

            # Auto-générer mouse_trial_code depuis le nom de la vidéo
            # Ex: vidéo "OF-M1-20251010-V01.mp4" + arène "A1" + mouse 15 → "OF-M1-20251010-V01_A1_M15"
            if video and arena_id:
                video_stem = Path(video).stem
                mtc = f"{video_stem}_{arena_id}"
                if mouse_id is not None:
                    mtc += f"_M{mouse_id}"
            else:
                mtc = None

            bg, border = video_colors.get(video, ("transparent", "transparent")) if video else ("transparent", "transparent")

            # Ligne vidéo header si changement
            if video and video != current_video:
                rows_html += (
                    f'<tr><td colspan="5" style="background:{bg};border-left:3px solid {border};'
                    f'padding:0.4rem 0.6rem;font-weight:600;font-size:0.85rem">{video}</td></tr>'
                )
                current_video = video

            stress_str = "oui" if row.get("stress") else ""
            angii_str = "oui" if row.get("angii") else ""
            mid_str = str(mouse_id) if mouse_id is not None else ""
            cond_str = row["condition"] if pd.notna(row.get("condition")) and row["condition"] else ""

            rows_html += (
                f'<tr style="background:{bg}">'
                f'<td style="padding:0.25rem 0.6rem;border-left:3px solid {border}">{arena_id}</td>'
                f'<td style="padding:0.25rem 0.4rem">{mid_str}</td>'
                f'<td style="padding:0.25rem 0.4rem">{cond_str}</td>'
                f'<td style="padding:0.25rem 0.4rem;font-family:monospace;font-size:0.8rem;color:#9ca3af">{mtc or ""}</td>'
                f'<td style="padding:0.25rem 0.4rem;font-size:0.8rem;color:#6b7280">'
                f'{"S" if stress_str else ""} {"A" if angii_str else ""}</td>'
                f'</tr>'
            )

            new_arenes.append({
                "id": arena_id,
                "video": video,
                "coords": None,
                "mouse_trial_code": mtc,
                "mouse_id": mouse_id,
                "condition": cond_str or None,
                "angii": bool(row.get("angii")) if pd.notna(row.get("angii")) else False,
                "stress": bool(row.get("stress")) if pd.notna(row.get("stress")) else False,
            })

        st.markdown("**Résumé**")
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;font-size:0.875rem">'
            f'<thead><tr style="border-bottom:1px solid #333">'
            f'<th style="text-align:left;padding:0.3rem 0.6rem">Arène</th>'
            f'<th style="text-align:left;padding:0.3rem 0.4rem">Mouse</th>'
            f'<th style="text-align:left;padding:0.3rem 0.4rem">Condition</th>'
            f'<th style="text-align:left;padding:0.3rem 0.4rem">Trial Code</th>'
            f'<th style="text-align:left;padding:0.3rem 0.4rem">Flags</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )
    arenes = new_arenes

    # ---- YAML brut (lecture seule) ----
    with st.expander("metadata.yaml brut"):
        preview = {**meta, "camera": camera, "arenes": arenes}
        st.code(yaml.dump(preview, allow_unicode=True, sort_keys=False), language="yaml")

    # ---- Sauvegarder ----
    just_saved = st.session_state.get("_meta_just_saved", False)
    if just_saved:
        st.session_state["_meta_just_saved"] = False

    # Bouton vert après sauvegarde, bleu sinon
    save_label = "Sauvegardé" if just_saved else "Sauvegarder"
    if just_saved:
        btn_bg = "#10b981"
        btn_hover = "#059669"
    else:
        btn_bg = "#3b82f6"
        btn_hover = "#2563eb"
    st.markdown(
        f"""<style>
        .st-key-meta_save button {{
            background: {btn_bg} !important;
            color: #fff !important;
            border: none !important;
        }}
        .st-key-meta_save button:hover {{
            background: {btn_hover} !important;
        }}
        .st-key-btn_sync button {{
            background: #3b82f6 !important;
            color: #fff !important;
            border: none !important;
        }}
        .st-key-btn_sync button:hover {{
            background: #2563eb !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )
    if st.button(save_label, type="primary", key="meta_save"):
        meta["camera"] = camera
        meta["arenes"] = arenes
        # Reconstruire source_video depuis videos_dir + video de la première arène (rétrocompat)
        if meta.get("videos_dir") and arenes:
            first_video = next((a["video"] for a in arenes if a.get("video")), None)
            if first_video:
                meta["source_video"] = str(Path(meta["videos_dir"]) / first_video)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            yaml.dump(meta, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        st.session_state["_meta_just_saved"] = True
        st.rerun()

    if just_saved:
        pname = current_project_name() or session
        st.success(f"metadata.yaml sauvegardé pour **{pname}**")


# ============================================================
# Entry
# ============================================================

def render() -> None:
    st.title("Données")
    st.caption("Visualiser et éditer les metadata des sessions, importer depuis Excel")

    _section_import_excel()
    st.divider()
    _section_metadata()
