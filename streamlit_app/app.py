"""
EthoFlow — interface web Streamlit.

Lancement :
    conda activate ethoflow
    streamlit run streamlit_app/app.py

L'app s'ouvre sur http://localhost:8501
Pour exposer sur le LAN : ajouter --server.address=0.0.0.0
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Localisation des données (chemins relatifs au repo)
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
CROPPED_DIR = DATA_ROOT / "cropped"
DLC_OUTPUT_DIR = DATA_ROOT / "dlc-output"
VAME_OUTPUT_DIR = DATA_ROOT / "vame-output"
SCRIPTS_DIR = ROOT / "scripts"


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="EthoFlow",
    page_icon="🐭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Helpers
# ============================================================

def load_metadata(session_id: str) -> dict | None:
    path = RAW_DIR / session_id / "metadata.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def list_sessions() -> pd.DataFrame:
    """Liste les sessions, leur metadata et leur statut d'avancement."""
    if not RAW_DIR.exists():
        return pd.DataFrame()

    rows = []
    for session_dir in sorted(RAW_DIR.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue
        session_id = session_dir.name
        meta = load_metadata(session_id) or {}
        source_video = meta.get("source_video")
        video_ok = bool(source_video and Path(source_video).exists())

        n_animals = sum(
            1 for a in meta.get("arenes", []) if a.get("mouse_id") is not None
        )

        # Une session est "splittée" si on trouve des .h5 nommés <session>_A*.h5
        dlc_dir = DLC_OUTPUT_DIR / session_id
        split_done = (
            dlc_dir.exists()
            and any(dlc_dir.glob(f"{session_id}_A*.h5"))
        )

        rows.append({
            "session_id": session_id,
            "timepoint": meta.get("timepoint", "—"),
            "date": meta.get("date", "—"),
            "animaux": n_animals,
            "vidéo": "OK" if video_ok else "manque",
            "DLC": "OK" if (DLC_OUTPUT_DIR / session_id).exists() else "—",
            "split": "OK" if split_done else "—",
            "VAME": "OK" if (VAME_OUTPUT_DIR / session_id).exists() else "—",
        })
    return pd.DataFrame(rows)


def arenes_dataframe(meta: dict) -> pd.DataFrame:
    rows = []
    for ar in meta.get("arenes", []):
        rows.append({
            "Arène": ar.get("id"),
            "MouseID": ar.get("mouse_id"),
            "Condition": ar.get("condition"),
            "Stress": "✓" if ar.get("stress") else "",
            "ANGII": "✓" if ar.get("angii") else "",
            "Coords": ar.get("coords") or "(à définir)",
            "MouseTrialCode": ar.get("mouse_trial_code") or "",
        })
    return pd.DataFrame(rows)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🐭 EthoFlow")
st.sidebar.caption("Analyse comportementale automatisée")

page = st.sidebar.radio(
    "Navigation",
    [
        "Tableau de bord",
        "Sync depuis Excel",
        "Détails session",
        "Lancer pipeline",
        "Résultats",
        "À propos",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Racine des données : `{DATA_ROOT}`")


# ============================================================
# Tableau de bord
# ============================================================

if page == "Tableau de bord":
    st.title("Tableau de bord")
    st.caption("Sessions présentes et statut d'avancement du pipeline")

    df = list_sessions()
    if df.empty:
        st.info(
            "Aucune session synchronisée. Va dans **Sync depuis Excel** "
            "pour générer les metadata à partir du fichier maître."
        )
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sessions", len(df))
        c2.metric("Vidéos OK", (df["vidéo"] == "OK").sum())
        c3.metric("DLC", (df["DLC"] == "OK").sum())
        c4.metric("Split arènes", (df["split"] == "OK").sum())
        c5.metric("VAME", (df["VAME"] == "OK").sum())
        st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# Sync depuis Excel
# ============================================================

elif page == "Sync depuis Excel":
    st.title("Synchroniser depuis l'Excel maître")
    st.caption(
        "Lit le fichier `OpenField_trials_*.xlsx` et génère un `metadata.yaml` "
        "par session dans `data/raw/`."
    )

    default_excel = ROOT.parent / "data" / "OpenField_trials_C DUPLAA.xlsx"
    default_videos = ROOT.parent / "data"

    excel_path = st.text_input("Chemin du fichier Excel", value=str(default_excel))
    videos_dir = st.text_input("Dossier des vidéos (.mp4)", value=str(default_videos))
    dry_run = st.checkbox("Dry-run (afficher sans écrire)", value=False)

    if st.button("Lancer le sync", type="primary"):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "sync_from_excel.py"),
            "--excel", excel_path,
            "--videos-dir", videos_dir,
        ]
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
# Détails session
# ============================================================

elif page == "Détails session":
    st.title("Détails d'une session")

    df = list_sessions()
    if df.empty:
        st.info("Pas de session.")
    else:
        session = st.selectbox("Session", options=df["session_id"].tolist())
        meta = load_metadata(session)
        if meta is None:
            st.error("metadata.yaml introuvable")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Timepoint", meta.get("timepoint", "—"))
            c2.metric("Date", meta.get("date", "—"))
            c3.metric("FPS", meta.get("camera", {}).get("fps", "—"))
            c4.metric("Trial n°", meta.get("trial_no", "—"))

            st.subheader("Arènes")
            st.dataframe(arenes_dataframe(meta), use_container_width=True, hide_index=True)

            with st.expander("metadata.yaml brut"):
                st.code(yaml.dump(meta, allow_unicode=True, sort_keys=False), language="yaml")

            sv = meta.get("source_video")
            if sv:
                if Path(sv).exists():
                    st.success(f"Vidéo source localisée : `{sv}`")
                else:
                    st.warning(f"Vidéo source introuvable : `{sv}`")


# ============================================================
# Lancer pipeline
# ============================================================

elif page == "Lancer pipeline":
    st.title("Lancer le pipeline")

    df = list_sessions()
    if df.empty:
        st.warning("Aucune session disponible.")
    else:
        sessions = st.multiselect("Sessions à traiter", options=df["session_id"].tolist())

        st.subheader("Étapes")
        c1, c2, c3 = st.columns(3)
        with c1:
            do_dlc = st.checkbox("1. Inférence DLC (multi-animal)", value=True)
        with c2:
            do_assign = st.checkbox("2. Split par arène", value=True)
        with c3:
            do_vame = st.checkbox("3. Analyse VAME", value=True)

        st.caption(
            "Étape optionnelle de pré-cropping (utile uniquement pour la "
            "labellisation d'un modèle custom) :"
        )
        do_crop = st.checkbox("Pré-crop des 4 arènes (optionnel)", value=False)

        st.markdown("---")
        if st.button("Lancer le pipeline", type="primary", disabled=not sessions):
            for session_id in sessions:
                with st.status(f"Traitement de `{session_id}`...", expanded=True) as status:
                    if do_crop:
                        st.write("→ Pré-crop des arènes (optionnel)")
                        result = subprocess.run(
                            [sys.executable, str(SCRIPTS_DIR / "crop_arenes.py"), session_id],
                            capture_output=True, text=True,
                        )
                        st.code(result.stdout or result.stderr or "(silencieux)")
                    if do_dlc:
                        st.write("→ Inférence DLC multi-animal (env conda 'dlc')")
                        st.info("À déclencher via `conda run -n dlc python scripts/run_dlc_inference.py`")
                    if do_assign:
                        st.write("→ Assignation arènes")
                        result = subprocess.run(
                            [sys.executable, str(SCRIPTS_DIR / "assign_arenas.py"), session_id],
                            capture_output=True, text=True,
                        )
                        st.code(result.stdout or result.stderr or "(silencieux)")
                    if do_vame:
                        st.write("→ Analyse VAME (env conda 'vame')")
                        st.info("À déclencher via `conda run -n vame python scripts/run_vame.py`")
                    status.update(label=f"`{session_id}` terminé", state="complete")


# ============================================================
# Résultats
# ============================================================

elif page == "Résultats":
    st.title("Résultats")

    df = list_sessions()
    if df.empty:
        st.info("Aucune session.")
    else:
        treated = df[df["DLC"] == "OK"]["session_id"].tolist()
        if not treated:
            st.warning("Aucune session avec inférence DLC complète.")
        else:
            session = st.selectbox("Session", options=treated)
            meta = load_metadata(session)
            if meta:
                st.subheader("Contexte")
                st.dataframe(arenes_dataframe(meta), use_container_width=True, hide_index=True)

            st.subheader("Fichiers générés")
            tabs = st.tabs(["DLC", "VAME"])
            with tabs[0]:
                dlc_dir = DLC_OUTPUT_DIR / session
                if dlc_dir.exists():
                    files = list(dlc_dir.iterdir())
                    st.write(f"{len(files)} fichier(s) dans `{dlc_dir}`")
                    for f in files[:20]:
                        st.text(f.name)
            with tabs[1]:
                vame_dir = VAME_OUTPUT_DIR / session
                if vame_dir.exists():
                    files = list(vame_dir.iterdir())
                    st.write(f"{len(files)} fichier(s) dans `{vame_dir}`")
                    for f in files[:20]:
                        st.text(f.name)
                else:
                    st.info("VAME pas encore exécuté pour cette session.")


# ============================================================
# À propos
# ============================================================

elif page == "À propos":
    st.title("À propos")
    st.markdown(
        """
        **EthoFlow** est une interface web légère pour orchestrer
        l'analyse comportementale de souris avec **DeepLabCut** et **VAME**.

        - Documentation : `docs/ETHOFLOW.md`
        - Source de vérité expérimentale : fichier Excel maître (`OpenField_trials_*.xlsx`)
        - Workflow : Excel → sync → metadata.yaml → crop → DLC → VAME → résultats

        Stack : Python · Streamlit · OpenCV · DeepLabCut · VAME
        """
    )
    st.caption(f"Lancé depuis : `{ROOT}`")
