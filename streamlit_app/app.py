"""
Pipeline souris — interface web Streamlit.

Lancement :
    conda activate pipeline-souris
    streamlit run streamlit_app/app.py

L'app s'ouvre sur http://localhost:8501
Pour exposer sur le LAN : ajouter --server.address=0.0.0.0
"""

import sys
import yaml
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

# Localisation des données (chemins relatifs au repo)
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
CROPPED_DIR = DATA_ROOT / "cropped"
DLC_OUTPUT_DIR = DATA_ROOT / "dlc-output"
VAME_OUTPUT_DIR = DATA_ROOT / "vame-output"
RESULTS_DIR = DATA_ROOT / "results"


# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Pipeline Souris",
    page_icon="🐭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Helpers
# ============================================================

def list_sessions():
    """Liste les sessions présentes dans data/raw/ avec leur statut d'avancement."""
    if not RAW_DIR.exists():
        return pd.DataFrame()

    rows = []
    for session_dir in sorted(RAW_DIR.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue
        session_id = session_dir.name
        metadata_file = session_dir / "metadata.yaml"
        has_video = any(session_dir.glob("*.mp4")) or any(session_dir.glob("*.avi"))

        rows.append({
            "session_id": session_id,
            "vidéo": "OK" if has_video else "manque",
            "metadata": "OK" if metadata_file.exists() else "manque",
            "cropped": "OK" if (CROPPED_DIR / session_id).exists() else "à faire",
            "DLC": "OK" if (DLC_OUTPUT_DIR / session_id).exists() else "à faire",
            "VAME": "OK" if (VAME_OUTPUT_DIR / session_id).exists() else "à faire",
        })
    return pd.DataFrame(rows)


def load_metadata(session_id: str):
    """Charge le metadata.yaml d'une session, ou retourne None."""
    path = RAW_DIR / session_id / "metadata.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🐭 Pipeline Souris")
st.sidebar.caption("Analyse comportementale automatisée")

page = st.sidebar.radio(
    "Navigation",
    [
        "Tableau de bord",
        "Nouvelle session",
        "Lancer pipeline",
        "Résultats",
        "À propos",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Racine des données : `{DATA_ROOT}`")


# ============================================================
# Page : Tableau de bord
# ============================================================

if page == "Tableau de bord":
    st.title("Tableau de bord")
    st.caption("Vue d'ensemble des sessions présentes dans `data/raw/`")

    df = list_sessions()
    if df.empty:
        st.info(
            "Aucune session trouvée. Va dans **Nouvelle session** pour en créer une, "
            "ou dépose manuellement une vidéo dans `data/raw/<session_id>/`."
        )
    else:
        # Stats rapides
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sessions totales", len(df))
        col2.metric("Croppées", (df["cropped"] == "OK").sum())
        col3.metric("DLC", (df["DLC"] == "OK").sum())
        col4.metric("VAME", (df["VAME"] == "OK").sum())

        st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# Page : Nouvelle session
# ============================================================

elif page == "Nouvelle session":
    st.title("Créer une nouvelle session")
    st.caption(
        "Crée le dossier de session et le fichier `metadata.yaml`. "
        "Tu pourras déposer la vidéo dedans ensuite."
    )

    with st.form("new_session"):
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date d'enregistrement", value=datetime.now())
            projet = st.text_input("Projet", placeholder="projet-X")
            chercheur = st.text_input("Chercheur", placeholder="nom.prenom")
        with col2:
            session_num = st.number_input("N° session du jour", min_value=1, value=1)
            protocole = st.text_input("Protocole", value="openfield-15min")
            fps = st.number_input("FPS", min_value=1, value=60)

        st.subheader("Arènes (4 par session)")
        st.caption("Renseigne l'animal et la condition de chaque arène.")

        arenes = []
        for i in range(1, 5):
            with st.expander(f"Arène {i}", expanded=(i == 1)):
                ca, cb = st.columns(2)
                with ca:
                    animal_id = st.text_input(
                        "Animal ID", key=f"animal_{i}", placeholder=f"M00{i}"
                    )
                with cb:
                    condition = st.text_input(
                        "Condition", key=f"cond_{i}", placeholder="control"
                    )
                arenes.append({
                    "id": f"arene-{i}",
                    "animal_id": animal_id,
                    "condition": condition,
                    "coords": None,  # à remplir manuellement ou via un outil de calibration
                })

        notes = st.text_area("Notes")

        submitted = st.form_submit_button("Créer la session", type="primary")
        if submitted:
            if not projet or not chercheur:
                st.error("Projet et chercheur sont obligatoires.")
            else:
                session_id = (
                    f"{date.strftime('%Y-%m-%d')}_{projet}_session-{session_num:03d}"
                )
                session_dir = RAW_DIR / session_id
                session_dir.mkdir(parents=True, exist_ok=True)

                metadata = {
                    "session_id": session_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "projet": projet,
                    "chercheur": chercheur,
                    "protocole": protocole,
                    "camera": {"resolution": "1920x1080", "fps": int(fps)},
                    "arenes": arenes,
                    "notes": notes,
                }
                with open(session_dir / "metadata.yaml", "w") as f:
                    yaml.dump(metadata, f, allow_unicode=True, sort_keys=False)

                st.success(f"Session créée : `{session_id}`")
                st.info(f"Dépose maintenant la vidéo (.mp4) dans `{session_dir}`")
                st.code(str(session_dir), language="text")


# ============================================================
# Page : Lancer pipeline
# ============================================================

elif page == "Lancer pipeline":
    st.title("Lancer le pipeline")

    df = list_sessions()
    if df.empty:
        st.warning("Aucune session disponible.")
    else:
        sessions = st.multiselect(
            "Sessions à traiter",
            options=df["session_id"].tolist(),
            help="Sélectionne une ou plusieurs sessions.",
        )

        st.subheader("Étapes")
        col1, col2, col3 = st.columns(3)
        with col1:
            do_crop = st.checkbox("1. Crop des arènes", value=True)
        with col2:
            do_dlc = st.checkbox("2. Inférence DLC", value=True)
        with col3:
            do_vame = st.checkbox("3. Analyse VAME", value=True)

        st.markdown("---")
        if st.button("Lancer le pipeline", type="primary", disabled=not sessions):
            for session_id in sessions:
                with st.status(f"Traitement de `{session_id}`...", expanded=True) as status:
                    if do_crop:
                        st.write("→ Crop des 4 arènes")
                        # TODO: subprocess.run([sys.executable, ROOT/"scripts/crop_arenes.py", session_id])
                        st.write("(à implémenter — appel à `scripts/crop_arenes.py`)")
                    if do_dlc:
                        st.write("→ Inférence DeepLabCut")
                        st.write("(à implémenter — appel à `scripts/run_dlc_inference.py`)")
                    if do_vame:
                        st.write("→ Analyse VAME")
                        st.write("(à implémenter — appel à `scripts/run_vame.py`)")
                    status.update(label=f"`{session_id}` terminé", state="complete")


# ============================================================
# Page : Résultats
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
            metadata = load_metadata(session)
            if metadata:
                st.subheader("Metadata")
                st.json(metadata)

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
# Page : À propos
# ============================================================

elif page == "À propos":
    st.title("À propos")
    st.markdown(
        """
        **Pipeline souris** est une interface web légère pour orchestrer
        l'analyse comportementale de souris avec **DeepLabCut** et **VAME**.

        - Documentation complète : `docs/PIPELINE_SOURIS_DOC.md`
        - Repo : (à pousser sur GitLab/GitHub)

        Stack : Python · Streamlit · OpenCV · DeepLabCut · VAME
        """
    )
    st.caption(f"Lancé depuis : `{ROOT}`")
