"""Page Lancer pipeline — orchestration DLC, post-DLC, VAME, analyse.

Chaque section a son propre bouton de lancement. Aucun bouton ne déclenche
plusieurs sections : on isole les actions pour laisser le chercheur valider
chaque étape avant la suivante.
"""
from __future__ import annotations

import streamlit as st

from lib.config import (
    cropped_dir,
    dlc_output_dir,
    vame_input_dir,
    vame_projects_root,
)
from lib.pipeline import (
    format_log_output,
    run_analyze_vame,
    run_dlc_inference,
    run_fill_nan,
    run_filter_keypoints,
    run_inspect_session,
    run_trim_empty,
    run_vame_stage,
)
from lib.sessions import list_sessions
from lib.vame_projects import (
    discover_projects,
    get_current_project_path,
    list_algos,
)


# ============================================================
# Sections
# ============================================================

def _section_sessions(df) -> tuple[list[str], bool]:
    """Sélection des sessions. Renvoie (selected, all_flag)."""
    st.header("1. Sélection des sessions")
    all_flag = st.checkbox(
        "Traiter toutes les sessions (équivalent `--all`)",
        value=False,
        help="Désactive le sélecteur ci-dessous et laisse les scripts décider.",
    )
    selected = st.multiselect(
        "Sessions à traiter",
        options=df["session_id"].tolist(),
        disabled=all_flag,
    )
    return selected, all_flag


def _section_dlc(selected: list[str], all_flag: bool) -> None:
    st.header("2. Inférence DLC")
    st.caption("Tourne dans l'env conda `dlc`. Voir `scripts/run_dlc_inference.py`.")

    mode_label = st.radio(
        "Mode",
        ["superanimal (multi)", "single-animal (cropped)", "custom"],
        horizontal=True,
        help="superanimal : SuperAnimal multi-animal sur vidéo entière. "
             "single-animal : SuperAnimal max_individuals=1 sur vidéos déjà croppées. "
             "custom : modèle DLC custom configuré dans pipeline_config.yaml.",
    )
    mode_map = {
        "superanimal (multi)": "superanimal",
        "single-animal (cropped)": "single-animal",
        "custom": "custom",
    }
    mode = mode_map[mode_label]

    col1, col2, col3 = st.columns(3)
    with col1:
        video_adapt = st.checkbox(
            "--video-adapt",
            value=(mode == "single-animal"),
            help="Fine-tuning court de SuperAnimal sur tes propres vidéos.",
        )
    with col2:
        video_adapt_bs = st.number_input(
            "--video-adapt-batch-size",
            min_value=1, max_value=8, value=2, step=1,
            disabled=not video_adapt,
            help="2 sur GPU 16 Go (RTX 5080), 4-8 sur 24 Go. Voir ETHOFLOW.md §5.4.",
        )
    with col3:
        skip_existing = st.checkbox("--skip-existing", value=True)

    if st.button("Lancer DLC", type="primary", key="btn_dlc",
                 disabled=not (selected or all_flag)):
        with st.status("DLC en cours...", expanded=True) as status:
            result = run_dlc_inference(
                session_ids=selected,
                mode=mode,
                video_adapt=video_adapt,
                video_adapt_batch_size=int(video_adapt_bs),
                skip_existing=skip_existing,
                all_sessions=all_flag,
            )
            format_log_output(result)
            ok = result.returncode == 0
            status.update(
                label="DLC terminé" if ok else "DLC échoué",
                state="complete" if ok else "error",
            )


def _section_post_dlc(selected: list[str]) -> None:
    st.header("3. Post-DLC — nettoyage des keypoints")
    st.caption("Tourne dans l'env conda `ethoflow`.")

    # ----- filter_keypoints -----
    with st.expander("filter_keypoints — virer les keypoints peu fiables", expanded=False):
        fk_in = st.text_input(
            "input-dir", value=str(vame_input_dir()), key="fk_in",
        )
        fk_out = st.text_input(
            "output-dir", value=str(vame_input_dir().parent / "vame-input-clean"),
            key="fk_out",
        )
        fk_keep = st.text_input(
            "keep (espace-séparé, vide = défaut du script)",
            value="", key="fk_keep",
            help="ex: nose left_ear right_ear neck mid_back mouse_center tail_base",
        )
        fk_min_validity = st.number_input(
            "min-validity (0 = ignorer)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key="fk_minv",
        )
        fk_dry = st.checkbox("dry-run", value=True, key="fk_dry")
        if st.button("Lancer filter_keypoints", key="btn_fk"):
            with st.status("filter_keypoints...", expanded=True) as status:
                result = run_filter_keypoints(
                    input_dir=fk_in,
                    output_dir=fk_out,
                    keep=fk_keep.split() if fk_keep.strip() else None,
                    min_validity=fk_min_validity if fk_min_validity > 0 else None,
                    dry_run=fk_dry,
                )
                format_log_output(result)
                ok = result.returncode == 0
                status.update(
                    label="filter_keypoints terminé" if ok else "filter_keypoints échoué",
                    state="complete" if ok else "error",
                )

    # ----- fill_nan_h5 -----
    with st.expander("fill_nan_h5 — imputer les NaN avant VAME", expanded=False):
        fn_root = st.text_input(
            "root", value=str(vame_input_dir().parent / "vame-input-clean"),
            key="fn_root",
        )
        fn_out = st.text_input(
            "output-dir (vide = en place)",
            value="", key="fn_out",
        )
        fn_dry = st.checkbox("dry-run", value=True, key="fn_dry")
        if st.button("Lancer fill_nan_h5", key="btn_fn"):
            with st.status("fill_nan_h5...", expanded=True) as status:
                result = run_fill_nan(
                    root=fn_root,
                    output_dir=fn_out if fn_out.strip() else None,
                    dry_run=fn_dry,
                )
                format_log_output(result)
                ok = result.returncode == 0
                status.update(
                    label="fill_nan_h5 terminé" if ok else "fill_nan_h5 échoué",
                    state="complete" if ok else "error",
                )

    # ----- inspect_session -----
    with st.expander("inspect_session — QC visuel par session", expanded=False):
        ins_input = st.text_input(
            "input-dir", value=str(vame_input_dir()), key="ins_in",
        )
        ins_all = st.checkbox(
            "--all (toutes les sessions de input-dir)",
            value=not bool(selected), key="ins_all",
        )
        ins_fps = st.number_input(
            "fps (0 = défaut du script)",
            min_value=0.0, max_value=240.0, value=0.0, step=1.0,
            key="ins_fps",
        )
        if st.button("Lancer inspect_session", key="btn_ins"):
            with st.status("inspect_session...", expanded=True) as status:
                result = run_inspect_session(
                    session_ids=selected,
                    input_dir=ins_input,
                    fps=ins_fps if ins_fps > 0 else None,
                    all_sessions=ins_all,
                )
                format_log_output(result)
                ok = result.returncode == 0
                status.update(
                    label="inspect_session terminé" if ok else "inspect_session échoué",
                    state="complete" if ok else "error",
                )

    # ----- trim_empty_arena -----
    with st.expander(
        "trim_empty_arena — tronquer les empty-arena en bord d'enregistrement",
        expanded=False,
    ):
        st.caption(
            "Optionnel. Voir ETHOFLOW.md §6.6. Nécessite un `validity_per_session.csv` "
            "préalablement produit par `analyze_vame --validity-source`."
        )
        te_validity = st.text_input("validity-csv", value="", key="te_validity")
        te_h5_in = st.text_input("h5-input", value=str(vame_input_dir()), key="te_h5_in")
        te_h5_out = st.text_input(
            "h5-output",
            value=str(vame_input_dir().parent / "vame-input-trimmed"),
            key="te_h5_out",
        )
        te_vid_in = st.text_input(
            "video-input", value=str(cropped_dir()), key="te_vid_in",
        )
        te_vid_out = st.text_input(
            "video-output",
            value=str(cropped_dir().parent / "cropped-trimmed"),
            key="te_vid_out",
        )
        te_dry = st.checkbox("dry-run", value=True, key="te_dry")
        if st.button("Lancer trim_empty_arena", key="btn_te",
                     disabled=not te_validity.strip()):
            with st.status("trim_empty_arena...", expanded=True) as status:
                result = run_trim_empty(
                    validity_csv=te_validity,
                    h5_input=te_h5_in,
                    h5_output=te_h5_out,
                    video_input=te_vid_in,
                    video_output=te_vid_out,
                    dry_run=te_dry,
                )
                format_log_output(result)
                ok = result.returncode == 0
                status.update(
                    label="trim terminé" if ok else "trim échoué",
                    state="complete" if ok else "error",
                )


def _section_vame() -> None:
    st.header("4. VAME — orchestration")
    st.caption("Tourne dans l'env conda `vame`. Une sous-commande = un bouton.")

    current = get_current_project_path()
    if current:
        st.info(f"Projet courant : `{current}`")
    else:
        st.warning(
            "Pas de projet VAME pointé par `.vame_config_path`. "
            "Lance d'abord `setup`, puis `use` pour basculer entre projets."
        )

    # ----- setup -----
    with st.expander("setup — initialiser un projet VAME", expanded=False):
        st_input = st.text_input(
            "input-dir (vide = data/vame-input/)",
            value="", key="vs_input",
        )
        st_cropped = st.text_input(
            "cropped-dir (vide = data/cropped/)",
            value="", key="vs_cropped",
        )
        st_name = st.text_input(
            "project-name", value="ethoflow-vame", key="vs_name",
        )
        st_pose_conf = st.slider(
            "pose-confidence", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            key="vs_conf",
        )
        st_force = st.checkbox("--force (écrase un projet existant)", value=False, key="vs_force")
        if st.button("Lancer VAME setup", key="btn_vsetup"):
            extra = ["--project-name", st_name, "--pose-confidence", str(st_pose_conf)]
            if st_input.strip():
                extra.extend(["--input-dir", st_input])
            if st_cropped.strip():
                extra.extend(["--cropped-dir", st_cropped])
            if st_force:
                extra.append("--force")
            with st.status("VAME setup...", expanded=True) as status:
                result = run_vame_stage("setup", extra)
                format_log_output(result)
                ok = result.returncode == 0
                status.update(
                    label="setup terminé" if ok else "setup échoué",
                    state="complete" if ok else "error",
                )

    # ----- les autres sous-commandes : un bouton chacune -----
    cols = st.columns(4)
    simple_stages = [
        ("Align", "align", "Alignement égocentrique + nettoyage"),
        ("Trainset", "trainset", "Crée le jeu d'entraînement"),
        ("Train", "train", "Entraînement du VAE (~30 min - 2 h, GPU)"),
        ("Evaluate", "evaluate", "Diagnostics post-entraînement"),
        ("Segment", "segment", "Segmentation des poses en motifs"),
        ("Motif-videos", "motif-videos", "Génère les cluster_videos par motif"),
        ("Community", "community", "Regroupement des motifs en communautés"),
        ("All", "all", "Enchaîne setup → segment (très long)"),
    ]
    for i, (label, stage, help_txt) in enumerate(simple_stages):
        with cols[i % 4]:
            if st.button(label, key=f"btn_v_{stage}", help=help_txt):
                with st.status(f"VAME {stage}...", expanded=True) as status:
                    result = run_vame_stage(stage)
                    format_log_output(result)
                    ok = result.returncode == 0
                    status.update(
                        label=f"VAME {stage} terminé" if ok else f"VAME {stage} échoué",
                        state="complete" if ok else "error",
                    )


def _section_analyze() -> None:
    st.header("5. Analyse statistique (analyze_vame)")
    st.caption("Tourne dans l'env conda `ethoflow`. Voir ETHOFLOW.md §6.6.")

    root = vame_projects_root()
    projects = discover_projects(root)
    if not projects:
        st.warning(
            f"Aucun projet VAME dans `{root}`. Change la racine dans la sidebar "
            "ou lance `setup` d'abord."
        )
        return

    project = st.selectbox(
        "Projet VAME",
        options=projects,
        format_func=lambda p: p.name,
        key="az_project",
    )
    algos = list_algos(project) or ["hmm-15"]
    algo_full = st.selectbox("Algo", options=algos, key="az_algo")

    # Parse algo-n pour passer les bons args au CLI
    try:
        from lib.vame_projects import parse_algo_n
        algo_name, n_clusters = parse_algo_n(algo_full)
    except ValueError:
        algo_name, n_clusters = "hmm", 15

    col1, col2 = st.columns(2)
    with col1:
        mask_empty = st.checkbox("--mask-empty", value=False, key="az_mask")
        validity_source = st.text_input(
            "--validity-source (dossier de h5 pré-fill, vide = ignorer)",
            value="", key="az_vsrc",
        )
    with col2:
        # Par défaut, pointe vers le YAML canonique du projet × algo s'il existe
        from lib.labels import labels_path
        default_labels = labels_path(project, algo_full)
        labels_default = str(default_labels) if default_labels.exists() else ""
        labels_input = st.text_input(
            "--labels (chemin YAML, vide = sans labels)",
            value=labels_default, key="az_labels",
        )
        min_edge_frames = st.number_input(
            "--min-edge-frames", min_value=1, max_value=10000, value=25, step=1,
            key="az_minedge",
        )

    if st.button("Lancer analyze_vame", type="primary", key="btn_analyze"):
        with st.status("analyze_vame...", expanded=True) as status:
            result = run_analyze_vame(
                project=project,
                algo=algo_name,
                n_clusters=n_clusters,
                labels_path=labels_input if labels_input.strip() else None,
                validity_source=validity_source if validity_source.strip() else None,
                mask_empty=mask_empty,
                min_edge_frames=int(min_edge_frames),
            )
            format_log_output(result)
            ok = result.returncode == 0
            status.update(
                label="analyze_vame terminé" if ok else "analyze_vame échoué",
                state="complete" if ok else "error",
            )


# ============================================================
# Entry
# ============================================================

def render() -> None:
    st.title("Lancer le pipeline")
    st.caption(
        "Chaque section déclenche un script indépendant. Lance dans l'ordre : "
        "DLC → post-DLC → VAME → analyse. Les boutons longs (`train`, `all`) "
        "tournent en arrière-plan, ferme pas l'onglet."
    )

    df = list_sessions()
    if df.empty:
        st.warning("Aucune session disponible. Lance `Sync depuis Excel` d'abord.")
        return

    selected, all_flag = _section_sessions(df)
    st.divider()
    _section_dlc(selected, all_flag)
    st.divider()
    _section_post_dlc(selected)
    st.divider()
    _section_vame()
    st.divider()
    _section_analyze()
