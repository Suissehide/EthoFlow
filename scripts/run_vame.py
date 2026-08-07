"""
Runner VAME — bootstrap d'un projet à partir de data/dlc-output/.

Étapes (sous-commandes séparées, à lancer dans l'ordre) :

    setup     init projet VAME à partir des paires (vidéo croppée, .h5)
    align     alignement égocentrique des poses
    trainset  création du jeu d'entraînement
    train     entraînement du VAE
    evaluate  diagnostics du modèle entraîné
    segment   segmentation des poses en motifs comportementaux
              (--n-clusters N pour autre chose que les 15 motifs par défaut)
    motif-videos  clips d'exemple par motif + génère data/vame/motif_labels.csv
    motif-labels  (re)génère seulement le CSV de labels
    community regroupement des motifs en communautés
    info      affiche le projet courant et le statut
    all       enchaîne setup → segment (long, plusieurs heures sur GPU)

Pré-requis :
    - Avoir crée l'env conda 'vame' (`conda env create -f environment-vame.yml`)
    - Avoir `data/dlc-output/<session>/<session>_A*.h5` (sorties d'assign_arenas
      ou de run_dlc_inference --mode single-animal) pour le topview,
      ou `data/dlc-output/<session>/<session>_clean.h5` pour le bottomview
    - Avoir les vidéos correspondantes dans `data/cropped/<session>/<session>_A*.mp4`
      → si tu n'as pas encore croppé, lance `python scripts/crop_arenes.py --all`
      depuis l'env ethoflow (rapide, ~2 min/session avec ffmpeg)

Usage typique :
    conda activate vame
    python scripts/run_vame.py --project-dir <project> setup
    python scripts/run_vame.py --project-dir <project> align
    python scripts/run_vame.py trainset
    python scripts/run_vame.py train
    python scripts/run_vame.py evaluate
    python scripts/run_vame.py segment
    python scripts/run_vame.py segment --n-clusters 25   # autre granularité
    python scripts/run_vame.py motif-videos              # clips + motif_labels.csv

Référence officielle (à garder ouverte pour ajuster les hyperparamètres) :
    https://github.com/LINCellularNeuroscience/VAME/blob/master/examples/demo.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Désactive l'accélération matérielle MSMF de cv2 sur Windows : VAME charge
# torch+CUDA AVANT d'appeler cv2.VideoCapture, et le hardware decoder MSMF
# (qui partage des ressources GPU avec CUVID) plante silencieusement quand
# torch a déjà claimé la GPU. Symptôme : motif_videos lève "Video capture
# could not be opened" alors que le fichier s'ouvre dans VLC et que cv2
# l'ouvre en standalone. setdefault pour ne pas écraser une valeur déjà
# posée par l'utilisateur. No-op hors Windows.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

# Import des chemins projet-aware
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_project_dir_arg,
    cropped_dir,
    data_dir,
    dlc_output_dir,
    raw_dir,
    resolve_project,
    vame_dir,
)

# Convention : un projet EthoFlow → un projet VAME, à plat dans data/vame/.
# Le projet VAME EST le dossier `<ethoflow_project>/data/vame/` lui-même
# (pas un sous-dossier). On obtient ça en passant à vame.init_new_project
# `working_directory=<project>/data/` et `project_name="vame"` —
# l'API crée alors `<project>/data/vame/`.
VAME_PROJECT_NAME = "vame"


def vame_config_yaml(project: Path) -> Path:
    """Path vers le config.yaml du projet VAME unique du projet EthoFlow."""
    return vame_dir(project) / "config.yaml"


# ============================================================
# Helpers
# ============================================================

def find_pairs(
    dlc_output_root: Path,
    cropped_dir: Path,
    raw_root: Path | None = None,
) -> list[tuple[Path, Path]]:
    """
    Liste toutes les paires (video, .h5) trouvées.

    Deux patterns supportés (essayés dans l'ordre pour chaque session) :

    Topview (multi-arena : 4 souris par vidéo source, croppées en 4 vidéos) :
        h5      : <dlc_output_root>/<session>/<session>_A*.h5
        video   : <cropped_dir>/<session>/<session>_A*.mp4

    Bottomview (single-animal : 1 souris par vidéo, pas de cropping) :
        h5      : <dlc_output_root>/<session>/<session>_clean.h5
        video   : `source_video` lu depuis <raw_root>/<session>/metadata.yaml

    Le paramètre `raw_root` n'est utilisé que pour le fallback bottomview.
    Si None, le fallback est ignoré (compat ascendante pour les appels
    existants qui ne donnent pas le raw_root).
    """
    import yaml as _yaml

    pairs: list[tuple[Path, Path]] = []
    if not dlc_output_root.exists():
        return pairs

    for session_dir in sorted(dlc_output_root.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue
        session_id = session_dir.name

        # --- Pattern 1 : topview cropped-by-arena ---
        topview_h5s = sorted(session_dir.glob(f"{session_id}_A*.h5"))
        if topview_h5s:
            for h5_path in topview_h5s:
                arena = h5_path.stem.rsplit("_", 1)[-1]  # "A1"
                video_path = cropped_dir / session_id / f"{session_id}_{arena}.mp4"
                if not video_path.exists():
                    print(f"  ⚠️  vidéo manquante : {video_path} — skip {h5_path.name}",
                          file=sys.stderr)
                    continue
                pairs.append((video_path, h5_path))
            continue  # session déjà résolue en topview

        # --- Pattern 2 : bottomview single-animal ---
        bottom_h5 = session_dir / f"{session_id}_clean.h5"
        if not bottom_h5.exists():
            continue
        if raw_root is None:
            print(f"  ⚠️  {session_id} : .h5 bottomview trouvé mais pas de raw_root "
                  f"pour résoudre la vidéo source", file=sys.stderr)
            continue
        meta_path = raw_root / session_id / "metadata.yaml"
        if not meta_path.exists():
            print(f"  ⚠️  {session_id} : metadata.yaml absent ({meta_path})",
                  file=sys.stderr)
            continue
        with open(meta_path) as f:
            meta = _yaml.safe_load(f) or {}
        src = meta.get("source_video")
        if not src:
            print(f"  ⚠️  {session_id} : pas de source_video dans la metadata",
                  file=sys.stderr)
            continue
        video_path = Path(src)
        if not video_path.exists():
            print(f"  ⚠️  vidéo source introuvable : {video_path} — skip {bottom_h5.name}",
                  file=sys.stderr)
            continue
        pairs.append((video_path, bottom_h5))

    return pairs


def load_config_pointer(project: Path) -> str:
    """Retourne le chemin du config.yaml du projet VAME.

    Depuis qu'on a flatté le layout (data/vame/ IS le projet), il n'y a
    plus de pointer .vame_config_path à maintenir : le path est
    structurellement déterminé par celui du projet EthoFlow. Cette
    fonction subsiste pour rétro-compat avec les callsites existants.
    """
    cfg = vame_config_yaml(project)
    if not cfg.exists():
        raise FileNotFoundError(
            f"Pas de projet VAME pour {project}.\n"
            f"   Attendu : {cfg}\n"
            f"   Crée-le avec : python scripts/run_vame.py "
            f"--project-dir {project} setup"
        )
    return str(cfg)


def load_vame_config(project: Path) -> dict:
    """vame-py 0.13 attend un config: dict (pas un chemin) dans la plupart des appels."""
    import vame
    return vame.read_config(load_config_pointer(project))


def detect_pose_ref_index(h5_path: Path) -> list[int]:
    """Trouve les indices de nose et tail_base parmi les keypoints du .h5."""
    import pandas as pd
    df = pd.read_hdf(h5_path)
    bp = df.columns.get_level_values("bodyparts").unique().tolist()
    # On cherche des noms compatibles avec ce qu'on a pu rencontrer
    def find(candidates, default):
        for name in candidates:
            if name in bp:
                return bp.index(name)
        return default
    nose = find(["nose", "Nose", "snout"], 0)
    tail = find(["tail_base", "tailbase", "TailBase", "tail", "tail_root"],
                len(bp) - 1)
    print(f"  → keypoints détectés : {bp}")
    print(f"  → pose_ref_index = [{nose} ({bp[nose]}), {tail} ({bp[tail]})]")
    return [nose, tail]


# ============================================================
# Commandes
# ============================================================

def cmd_setup(args) -> None:
    try:
        import vame
    except ImportError:
        print("❌ VAME non installé. Active l'env conda 'vame'.", file=sys.stderr)
        sys.exit(1)

    project = resolve_project(args)
    input_dir = Path(args.input_dir) if args.input_dir else dlc_output_dir(project)
    crop_dir = Path(args.cropped_dir) if args.cropped_dir else cropped_dir(project)
    raw_root = raw_dir(project)

    pairs = find_pairs(input_dir, crop_dir, raw_root=raw_root)
    if not pairs:
        print("❌ Aucune paire (vidéo, .h5) trouvée.\n"
              "   Patterns supportés :\n"
              f"   - topview    : {input_dir}/<session>/<session>_A*.h5\n"
              f"                + {crop_dir}/<session>/<session>_A*.mp4\n"
              f"   - bottomview : {input_dir}/<session>/<session>_clean.h5\n"
              f"                + source_video lu dans {raw_root}/<session>/metadata.yaml",
              file=sys.stderr)
        sys.exit(1)

    print(f"{len(pairs)} paires trouvées :")
    for v, h in pairs[:5]:
        print(f"  {v.name}  +  {h.name}")
    if len(pairs) > 5:
        print(f"  ... (+{len(pairs) - 5} autres)")

    # Auto-rekey : VAME (via movement) attend la clé HDF5 'df_with_missing'.
    # Les .h5 produits avant le fix avaient key='df' et VAME crashe dessus.
    # On corrige en place avant d'appeler init_new_project.
    if not args.no_auto_rekey:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from rekey_h5 import is_already_correct, rekey
        except ImportError as e:
            print(f"⚠️  Impossible d'importer rekey_h5 ({e}), skip auto-rekey",
                  file=sys.stderr)
        else:
            to_fix = [h for _, h in pairs if not is_already_correct(h)]
            if to_fix:
                print(f"\n🔧 Auto-rekey : {len(to_fix)} fichier(s) à corriger "
                      f"(ancienne clé 'df' → 'df_with_missing')")
                for h in to_fix:
                    status = rekey(h)
                    if status == "rekeyed":
                        print(f"   ✓ {h.name}")
                    else:
                        print(f"   ⚠️  {h.name} : {status}", file=sys.stderr)

    # On veut que `<project>/data/vame/` SOIT le projet VAME (pas un parent
    # qui contient `<project>/data/vame/<name>/`). On obtient ça en passant
    # working_directory=<project>/data/ et project_name="vame" — vame crée
    # alors `<project>/data/vame/`.
    project_dir = vame_dir(project)
    working_directory = data_dir(project)
    working_directory.mkdir(parents=True, exist_ok=True)

    # Refus d'écraser un projet existant — sauf avec --force, qui supprime
    # tout le contenu pour rattraper une tentative ratée.
    if project_dir.exists() and any(project_dir.iterdir()):
        if not args.force:
            print(f"❌ Projet VAME déjà présent : {project_dir}\n"
                  f"   Relance avec --force pour écraser (le dossier sera vidé).",
                  file=sys.stderr)
            sys.exit(1)
        import shutil
        print(f"⚠️  --force : suppression de {project_dir}")
        shutil.rmtree(project_dir)

    videos = [str(v) for v, _ in pairs]
    poses = [str(p) for _, p in pairs]

    # Symlink vs copy : sur Windows, les symlinks demandent des privilèges
    # admin ou le Developer Mode — sinon on prend une OSError 1314. Par défaut
    # on copie sur Windows et on symlinke ailleurs ; --copy-videos / --no-copy-videos
    # forcent un choix.
    import platform
    if args.copy_videos is None:
        copy_videos = (platform.system() == "Windows")
    else:
        copy_videos = args.copy_videos
    if copy_videos:
        print("ℹ️  Mode COPY (les vidéos sont copiées dans le projet, "
              "pas de symlinks).")

    print(f"\nCréation du projet VAME dans {project_dir}...")
    # vame-py 0.13 : init_new_project retourne (config_path, config_dict)
    result = vame.init_new_project(
        project_name=VAME_PROJECT_NAME,
        poses_estimations=poses,
        source_software="DeepLabCut",
        working_directory=str(working_directory),
        videos=videos,
        video_type=".mp4",
        copy_videos=copy_videos,
    )
    # Compatibilité défensive : ancienne API renvoyait juste un chemin
    if isinstance(result, tuple):
        config_path = result[0]
    else:
        config_path = result

    # VAME copie/symlinke les vidéos dans data/raw/ avec leur nom d'origine
    # (ex: 1001.mp4) mais nomme les .h5 d'après la session (BV-1001.h5).
    # Les étapes downstream (motif_videos en particulier) construisent les
    # chemins vidéo à partir du nom de SESSION et cherchent donc BV-1001.mp4.
    # On aligne ici post-init pour éviter le mismatch.
    project_path = Path(config_path).parent
    raw_vame = project_path / "data" / "raw"
    if raw_vame.exists():
        n_renamed = 0
        for video_src, h5_src in pairs:
            expected_name = h5_src.stem + ".mp4"     # BV-1001.mp4
            current_name = Path(video_src).name      # 1001.mp4
            if current_name == expected_name:
                continue
            actual = raw_vame / current_name
            expected = raw_vame / expected_name
            if actual.exists() and not expected.exists():
                actual.rename(expected)
                n_renamed += 1
        if n_renamed:
            print(f"\nℹ️  {n_renamed} vidéo(s) renommée(s) pour matcher le "
                  f"nom de session attendu par VAME downstream.")

    # Ajuste pose_confidence dans le config.yaml du projet — par défaut VAME le
    # met à 0.99, ce qui mask la majorité des points SuperAnimal (typiquement
    # 0.5-0.95). On s'aligne sur notre seuil de pré-cleaning (0.6) pour ne
    # rejeter que les points dont DLC était vraiment peu sûr.
    if args.pose_confidence is not None:
        cfg = vame.read_config(config_path)
        old = cfg.get("pose_confidence", "?")
        cfg["pose_confidence"] = args.pose_confidence
        vame.write_config(config_path, cfg)
        print(f"\nℹ️  pose_confidence : {old} → {args.pose_confidence}")

    print(f"\n✅ Projet VAME créé.\n   config.yaml : {config_path}")
    print("\nLe `config.yaml` contient les hyperparamètres du modèle. Tu peux\n"
          "l'éditer avant l'entraînement (taille de fenêtre, learning rate, etc).\n"
          "\nÉtape suivante :  python scripts/run_vame.py align")


def cmd_align(args) -> None:
    """
    vame-py >= 0.x : remplacé par `preprocessing` qui regroupe alignement
    égocentrique + filtrage outliers + lissage. Signature probable :
        vame.preprocessing(
            config,
            centered_reference_keypoint=<nom kp queue>,
            orientation_reference_keypoint=<nom kp nez>,
        )
    Si la signature diffère sur ta version, lance :
        python -c "import vame; help(vame.preprocessing)"
    et corrige les noms d'arguments dans la fonction ci-dessous.
    """
    import vame
    project = resolve_project(args)
    config = load_vame_config(project)

    # Les keypoints sont déjà dans le config.yaml du projet (copiés par
    # init_new_project), pas besoin de relire les .h5 originaux. Du coup
    # cmd_align ne dépend plus du dossier dlc-output, ce qui simplifie
    # l'organisation quand on a plusieurs runs DLC distincts.
    bp = list(config.get("keypoints") or [])
    if not bp:
        print("❌ Pas de 'keypoints' dans le config.yaml du projet VAME.",
              file=sys.stderr)
        sys.exit(1)

    def find(candidates, default):
        for name in candidates:
            if name in bp:
                return name
        return default

    nose_kp = find(["nose", "Nose", "Snout", "snout"], bp[0])
    tail_kp = find(["tail_base", "tailbase", "Tailbase", "TailBase", "tail"],
                   bp[len(bp) // 2])

    print(f"  → keypoints détectés : {bp}")
    print(f"  → reference keypoints : center={tail_kp}, orientation={nose_kp}")

    # Défauts EthoFlow-aware :
    # - lowconf_cleaning ON : un no-op si fill_nan_h5 a tourné (likelihood=1.0),
    #   peu coûteux à laisser quand ce n'est pas le cas.
    # - egocentric_alignment ON : indispensable.
    # - outlier_cleaning OFF par défaut : son IQR peut ré-introduire des NaN
    #   qui font crasher savgol et empoisonnent le training.
    # - savgol_filtering OFF par défaut : crashe sur NaN ; on lisse plutôt
    #   en amont via notre fill_nan_h5 (interpolation).
    steps = {
        "run_lowconf_cleaning":     not args.no_lowconf_cleaning,
        "run_egocentric_alignment": not args.no_alignment,
        "run_outlier_cleaning":     args.with_outlier_cleaning,
        "run_savgol_filtering":     args.with_savgol,
        "run_rescaling":            args.rescaling,
    }
    enabled = [k for k, v in steps.items() if v]
    skipped = [k for k, v in steps.items() if not v]
    print(f"  → étapes actives : {enabled}")
    if skipped:
        print(f"  → étapes désactivées : {skipped}")

    vame.preprocessing(
        config,
        centered_reference_keypoint=tail_kp,
        orientation_reference_keypoint=nose_kp,
        **steps,
    )
    print("\n✅ Preprocessing terminé. "
          "Étape suivante : python scripts/run_vame.py trainset")


def _ensure_position_processed(project_path: Path) -> None:
    """
    VAME's create_trainset attend une variable 'position_processed' dans chaque
    .nc de data/processed/. Normalement créée par la dernière étape de
    preprocessing (savgol), mais cette dernière soit n'a pas tourné, soit
    a produit une variable 100% NaN à cause des NaN d'entrée. Dans les deux
    cas on a besoin d'une version utilisable.

    On la (re)fabrique TOUJOURS à partir de 'position_egocentric_aligned',
    en bouchant les NaN dans cet ordre :
      - ffill sur l'axe time (propage la dernière valeur valide vers l'avant)
      - bfill (propage la prochaine valeur valide vers l'arrière)
      - fillna(0) pour les keypoints totalement NaN dans une session
    """
    import shutil
    import numpy as np
    import xarray as xr

    processed_dir = project_path / "data" / "processed"
    if not processed_dir.exists():
        return

    n_fixed = 0
    for nc_path in sorted(processed_dir.glob("*.nc")):
        with xr.open_dataset(nc_path) as ds:
            if "position_egocentric_aligned" not in ds.data_vars:
                print(f"  ⚠️  {nc_path.name} : pas de 'position_egocentric_aligned', skip",
                      file=sys.stderr)
                continue

            aligned = ds["position_egocentric_aligned"]
            n_before = int(np.isnan(aligned.values).sum())
            filled = aligned.ffill(dim="time").bfill(dim="time").fillna(0.0)
            n_after = int(np.isnan(filled.values).sum())

            # Vérifier l'état actuel de position_processed pour info
            existing_status = "absent"
            if "position_processed" in ds.data_vars:
                cur = ds["position_processed"].values
                n_cur_nan = int(np.isnan(cur).sum())
                pct = 100 * n_cur_nan / cur.size if cur.size else 0
                existing_status = f"existant ({pct:.1f}% NaN, écrasé)"

            new_ds = ds.assign(position_processed=filled).load()

        # to_netcdf ne peut pas écrire dans un fichier ouvert
        tmp = nc_path.with_suffix(".nc.tmp")
        new_ds.to_netcdf(tmp)
        new_ds.close()
        shutil.move(str(tmp), str(nc_path))
        n_fixed += 1
        total = aligned.size
        pct_before = 100 * n_before / total if total else 0
        pct_after = 100 * n_after / total if total else 0
        print(f"  ✓ {nc_path.name:50s} "
              f"NaN: {pct_before:.1f}% → {pct_after:.1f}%  "
              f"(was: {existing_status})")

    if n_fixed:
        print(f"\n→ {n_fixed} fichier(s) (re)généré(s) avec position_processed propre.")


def _fix_seq_mean_std(project_path: Path) -> None:
    """
    Recalcule seq_mean.npy et seq_std.npy depuis train_seq.npy.

    VAME's create_trainset écrit parfois des NaN scalaires dans seq_mean et
    seq_std (probablement à cause d'un calcul fait sur une variable
    intermédiaire qui contient encore des NaN — `position_cleaned_lowconf`
    ou similaire — avant que notre `_ensure_position_processed` n'ait fini
    son boulot). La normalisation `(train_seq - mean) / std` produit ensuite
    du NaN partout → loss NaN → training mort à l'epoch 50.

    Comme `train_seq.npy` lui-même est propre, on recalcule mean et std
    directement à partir de lui.
    """
    import numpy as np

    train_dir = project_path / "data" / "train"
    train_seq_path = train_dir / "train_seq.npy"
    mean_path = train_dir / "seq_mean.npy"
    std_path = train_dir / "seq_std.npy"

    if not train_seq_path.exists():
        print(f"  ⚠️  Pas de train_seq.npy à {train_dir}", file=sys.stderr)
        return

    train_seq = np.load(train_seq_path)
    train_nan = int(np.isnan(train_seq).sum())
    if train_nan > 0:
        print(f"  ⚠️  train_seq.npy contient {train_nan} NaN — "
              f"le fix mean/std ne suffira pas. À investiguer.",
              file=sys.stderr)

    needs_fix = False
    for path, label in [(mean_path, "seq_mean"), (std_path, "seq_std")]:
        if not path.exists():
            needs_fix = True
            continue
        arr = np.load(path)
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"  ❌ {label}.npy contient NaN ou Inf")
            needs_fix = True

    if not needs_fix:
        print(f"  ✓ seq_mean / seq_std déjà propres")
        return

    new_mean = float(np.nanmean(train_seq))
    new_std = float(np.nanstd(train_seq))
    np.save(mean_path, np.array(new_mean))
    np.save(std_path, np.array(new_std))
    print(f"  ✓ Recalculé depuis train_seq : mean={new_mean:.4f}, std={new_std:.4f}")


def cmd_trainset(args) -> None:
    import vame
    project = resolve_project(args)
    config_path = load_config_pointer(project)
    project_path = Path(config_path).parent

    # Bouche le trou si savgol n'a pas tourné (cas typique : --no-savgol pour
    # éviter le crash sur NaN). Idempotent — ne fait rien si position_processed
    # existe déjà.
    print("Vérification de 'position_processed' dans les .nc preprocessés...")
    _ensure_position_processed(project_path)

    vame.create_trainset(load_vame_config(project))

    # VAME peut écrire un seq_mean / seq_std contenant des NaN — la
    # normalisation (train_seq - mean) / std produit alors du NaN partout
    # et tue le training. On recalcule depuis train_seq.npy (propre).
    print("\nVérification de seq_mean / seq_std...")
    _fix_seq_mean_std(project_path)

    print("\n✅ Trainset créé. Étape suivante : python scripts/run_vame.py train")


def cmd_train(args) -> None:
    import vame

    if args.no_cluster_loss:
        # Monkey-patch : vame.model.rnn_vae.cluster_loss() appelle torch.svd()
        # sur une gram matrix qui est régulièrement mal conditionnée (latents
        # collapsés, données peu diverses, etc.) et fait planter le training
        # avant le premier epoch — même quand le poids kmeans_loss vaut 0.
        # Comme la fonction n'est utile que si on veut une régularisation par
        # cluster, on la remplace par un no-op qui renvoie 0.
        import torch
        import vame.model.rnn_vae as rnn_vae

        def _zero_cluster_loss(latent, kloss, klmbda, bsize):
            return torch.tensor(0.0, device=latent.device)

        rnn_vae.cluster_loss = _zero_cluster_loss
        print("⚠️  cluster_loss désactivée (monkey-patch) pour contourner les "
              "crashes SVD sur latents mal conditionnés.")

    print("Entraînement du VAE — peut prendre plusieurs heures sur GPU.")
    print("Les hyperparamètres sont dans le config.yaml du projet VAME.")
    project = resolve_project(args)
    vame.train_model(load_vame_config(project))
    print("\n✅ Entraînement terminé.")


def cmd_evaluate(args) -> None:
    import vame
    project = resolve_project(args)
    vame.evaluate_model(load_vame_config(project))
    print("\n✅ Évaluation terminée — vois les figures dans le dossier du projet.")


def set_n_clusters(project: Path, n: int) -> None:
    """Écrit `n_clusters` dans le config.yaml du projet VAME.

    C'est ce nombre qui décide combien de motifs la segmentation produit
    (défaut VAME : 15). Le changer et relancer `segment` produit un
    nouveau dossier `results/<session>/<model>/<algo>-<n>/` — les
    résultats précédents ne sont pas écrasés.
    """
    import yaml as _yaml
    cfg_path = vame_config_yaml(project)
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f) or {}
    key = "n_clusters" if "n_clusters" in cfg or "n_cluster" not in cfg \
        else "n_cluster"
    old = cfg.get(key)
    cfg[key] = int(n)
    with open(cfg_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"ℹ  {key} : {old} → {n}  (dans {cfg_path.name})")


def cmd_segment(args) -> None:
    """vame-py >= 0.x : `pose_segmentation` renommée en `segment_session`."""
    import vame
    project = resolve_project(args)
    if getattr(args, "n_clusters", None):
        set_n_clusters(project, args.n_clusters)
    print("Segmentation des poses en motifs comportementaux...")
    vame.segment_session(load_vame_config(project))
    print("\n✅ Segmentation terminée.")
    print("\nÉtapes suivantes possibles :")
    print("  - python scripts/run_vame.py motif-videos   # clips par motif + motif_labels.csv")
    print("  - python scripts/run_vame.py community      # regroupement de motifs en communautés")
    print("  - python scripts/analyze_vame.py            # comparaison par condition (custom EthoFlow)")


MOTIF_LABELS_COLUMNS = [
    "motif_id", "label", "category", "confidence",
    "qc_inspected_sessions", "notes", "usage_pct", "video",
]

# Catégories du référentiel ETHOGRAM, rappelées en commentaire du CSV pour
# que l'utilisateur n'ait pas à les chercher au moment de remplir.
ETHOGRAM_CATEGORIES = [
    "Locomotion", "Stationary", "Vertical exploration", "Sniffing",
    "Grooming", "Exploration", "Specific behaviors", "Transitions",
]


def read_vame_clusters(project: Path) -> tuple[int | None, list[str]]:
    """Lit (n_clusters, algorithmes de segmentation) du config VAME."""
    import yaml as _yaml
    cfg_path = vame_config_yaml(project)
    if not cfg_path.exists():
        return None, ["hmm"]
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f) or {}
    n = cfg.get("n_clusters", cfg.get("n_cluster"))
    algos = cfg.get("segmentation_algorithms") or ["hmm"]
    if isinstance(algos, str):
        algos = [algos]
    return (int(n) if n else None), list(algos)


def motif_usage_pct(vame_project: Path, algo: str,
                     n_clusters: int) -> list[float]:
    """Fréquence d'usage de chaque motif, en % du total, toutes sessions.

    Sert à pré-remplir le CSV : un motif à 0.1 % ne mérite probablement pas
    qu'on passe 10 min à le nommer, un motif à 20 % oui.
    """
    try:
        import numpy as np
    except ImportError:
        return [float("nan")] * n_clusters
    total = None
    for npy in (vame_project / "results").glob(
            f"*/*/{algo}-{n_clusters}/motif_usage_*.npy"):
        try:
            arr = np.asarray(np.load(npy), dtype=float).ravel()
        except Exception:
            continue
        if arr.size < n_clusters:
            arr = np.pad(arr, (0, n_clusters - arr.size))
        arr = arr[:n_clusters]
        total = arr if total is None else total + arr
    if total is None or total.sum() <= 0:
        return [float("nan")] * n_clusters
    return list(100.0 * total / total.sum())


def find_motif_video(vame_project: Path, motif_id: int) -> str:
    """Chemin (relatif au projet VAME) d'un clip d'exemple du motif."""
    for pattern in (f"**/*motif_{motif_id}.mp4", f"**/*motif_{motif_id}.avi",
                    f"**/*motif_{motif_id}_*.mp4"):
        for p in (vame_project / "results").glob(pattern):
            return str(p.relative_to(vame_project))
    return ""


def write_motif_labels_csv(project: Path, vame_project: Path,
                            overwrite: bool = False) -> Path | None:
    """Écrit `data/vame/motif_labels.csv` pré-rempli, une ligne par motif.

    Les colonnes `label` et `category` sont laissées **vides** : c'est le
    travail de l'utilisateur après visionnage des clips. Le reste est
    pré-rempli (id, fréquence d'usage, chemin du clip) pour lui éviter de
    fabriquer le fichier à la main.

    Si le fichier existe déjà, on ne l'écrase pas (le travail
    d'annotation est précieux) sauf `overwrite=True`.
    """
    import csv

    out_path = vame_project / "motif_labels.csv"
    n_clusters, algos = read_vame_clusters(project)
    if not n_clusters:
        print("⚠  n_clusters introuvable dans le config VAME — "
              "CSV de labels non généré.")
        return None

    if out_path.exists() and not overwrite:
        print(f"ℹ  {out_path.name} existe déjà — laissé intact "
              f"(--regen-labels pour le regénérer).")
        return out_path

    algo = algos[0]
    usage = motif_usage_pct(vame_project, algo, n_clusters)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(MOTIF_LABELS_COLUMNS)
        for i in range(n_clusters):
            pct = usage[i]
            w.writerow([
                i, "", "", "", "", "",
                "" if pct != pct else f"{pct:.2f}",   # NaN → vide
                find_motif_video(vame_project, i),
            ])

    print(f"\n✅ CSV de labels pré-rempli : {out_path}")
    print(f"   {n_clusters} lignes (algo={algo}), colonnes `label` et "
          f"`category` à remplir après visionnage.")
    print(f"   Catégories ETHOGRAM : {', '.join(ETHOGRAM_CATEGORIES)}")
    print("   Un motif jamais interprétable ? mets `artifact` en category, "
          "analyze_vame.py l'exclura.")
    return out_path


def cmd_motif_videos(args) -> None:
    """Génère une courte vidéo d'exemple pour chaque motif comportemental."""
    import vame
    project = resolve_project(args)
    print("Génération des vidéos par motif (peut être long)...")
    vame.motif_videos(load_vame_config(project))
    print("\n✅ Vidéos générées.")
    vame_project = Path(load_config_pointer(project)).parent
    sample_path = vame_project / "results" / "<session>" / "<model>" / "<algo>-<n>" / "motif_videos"
    print(f"   Cherche dans : {sample_path}")

    write_motif_labels_csv(project, vame_project,
                            overwrite=getattr(args, "regen_labels", False))


def cmd_motif_labels(args) -> None:
    """(Re)génère seulement le CSV de labels, sans refaire les vidéos."""
    project = resolve_project(args)
    vame_project = Path(load_config_pointer(project)).parent
    write_motif_labels_csv(project, vame_project,
                            overwrite=getattr(args, "regen_labels", False))


def cmd_community(args) -> None:
    """Regroupe les motifs en communautés (motifs sémantiquement proches)."""
    import vame
    project = resolve_project(args)
    print("Construction des communautés de motifs...")
    vame.community(load_vame_config(project))
    print("\n✅ Communautés calculées.")
    vame_project = Path(load_config_pointer(project)).parent
    print(f"   Résultats dans : {vame_project / 'results'}")


def cmd_info(args) -> None:
    """Affiche l'état du projet VAME (unique) du projet EthoFlow."""
    project = resolve_project(args)
    cfg_yaml = vame_config_yaml(project)

    if cfg_yaml.exists():
        print(f"Projet VAME : {cfg_yaml}")
        try:
            import vame
            cfg = vame.read_config(str(cfg_yaml))
            print(f"  → {len(cfg.get('session_names') or [])} session(s) "
                  f"importée(s) dans le projet")
            print(f"  → {len(cfg.get('keypoints') or [])} keypoint(s)")
        except Exception:
            pass
    else:
        print(f"Pas de projet VAME pour {project}.")
        print(f"  Crée-le avec : python scripts/run_vame.py "
              f"--project-dir {project} setup")

    # Scan optionnel d'un dossier dlc-output (pour planifier un futur setup)
    input_dir = Path(args.input_dir) if args.input_dir else dlc_output_dir(project)
    crop_dir = Path(args.cropped_dir) if args.cropped_dir else cropped_dir(project)
    if input_dir.exists():
        pairs = find_pairs(input_dir, crop_dir, raw_root=raw_dir(project))
        sessions = {p[1].stem.rsplit("_", 1)[0] for p in pairs}
        print(f"\nDans {input_dir} :")
        print(f"  → {len(pairs)} paire(s) (vidéo + h5) disponibles")
        print(f"  → {len(sessions)} session(s) distincte(s)")


def cmd_all(args) -> None:
    cmd_setup(args)
    cmd_align(args)
    cmd_trainset(args)
    cmd_train(args)
    cmd_evaluate(args)
    cmd_segment(args)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner VAME pour EthoFlow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_project_dir_arg(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup", help="Init projet VAME")
    p_setup.add_argument("--input-dir", default=None,
                         help="Dossier des .h5 (défaut: data/dlc-output/)")
    p_setup.add_argument("--cropped-dir", default=None,
                         help="Dossier des vidéos croppées (défaut: data/cropped/)")
    p_setup.add_argument("--force", action="store_true",
                         help="Supprimer un projet du même nom s'il existe déjà")
    p_setup.add_argument("--no-auto-rekey", action="store_true",
                         help="Ne pas re-clé-er automatiquement les .h5 à la "
                              "clé 'df_with_missing' (par défaut auto-corrigé)")
    p_setup.add_argument("--pose-confidence", type=float, default=0.6,
                         help="Seuil de confiance VAME (lowconf_cleaning) — "
                              "défaut 0.6, aligné sur notre pré-filtrage. "
                              "Mets None pour garder le 0.99 par défaut de VAME.")
    copy_grp = p_setup.add_mutually_exclusive_group()
    copy_grp.add_argument("--copy-videos", dest="copy_videos",
                          action="store_const", const=True, default=None,
                          help="Copier les vidéos dans le projet (au lieu de symlink). "
                               "Défaut auto : copy sur Windows, symlink ailleurs.")
    copy_grp.add_argument("--no-copy-videos", dest="copy_videos",
                          action="store_const", const=False,
                          help="Forcer le symlink (échouera sur Windows sans Developer Mode)")

    p_align = sub.add_parser("align", help="Preprocessing VAME (alignement + nettoyage)")
    # Flags 'no-' désactivent ce qui est ON par défaut
    p_align.add_argument("--no-lowconf-cleaning", action="store_true",
                         help="Désactive le lowconf_cleaning (ON par défaut, mais "
                              "no-op après fill_nan_h5 car likelihood=1.0 partout)")
    p_align.add_argument("--no-alignment", action="store_true",
                         help="Désactive l'alignement égocentrique (ON par défaut, "
                              "ne le désactive que pour debug)")
    # Flags 'with-' activent ce qui est OFF par défaut (parce que ces étapes
    # font régulièrement planter le pipeline EthoFlow)
    p_align.add_argument("--with-outlier-cleaning", action="store_true",
                         help="Active le nettoyage des outliers IQR (OFF par défaut "
                              "car peut ré-introduire des NaN qui crashent savgol)")
    p_align.add_argument("--with-savgol", action="store_true",
                         help="Active le filtre Savitzky-Golay (OFF par défaut car "
                              "crashe sur NaN ; à n'activer que sur données 100%% propres)")
    p_align.add_argument("--rescaling", action="store_true",
                         help="Activer le rescaling (désactivé par défaut)")
    sub.add_parser("trainset", help="Création du trainset")
    p_train = sub.add_parser("train", help="Entraînement du VAE (long)")
    p_train.add_argument("--no-cluster-loss", action="store_true",
                         help="Désactive vame.cluster_loss (no-op). À utiliser "
                              "quand le SVD plante au premier epoch.")
    sub.add_parser("evaluate", help="Évaluation du modèle")
    p_seg = sub.add_parser("segment", help="Segmentation en motifs")
    p_seg.add_argument("--n-clusters", type=int, default=None,
                       help="Nombre de motifs à produire (défaut VAME : 15). "
                            "Écrit la valeur dans le config VAME avant de "
                            "segmenter. Chaque valeur crée son propre dossier "
                            "de résultats, rien n'est écrasé.")
    p_mv = sub.add_parser("motif-videos",
                          help="Vidéos d'exemple par motif + motif_labels.csv")
    p_mv.add_argument("--regen-labels", action="store_true",
                      help="Écrase motif_labels.csv s'il existe déjà "
                           "(attention : perd les labels saisis).")
    p_ml = sub.add_parser("motif-labels",
                          help="(Re)génère seulement data/vame/motif_labels.csv")
    p_ml.add_argument("--regen-labels", action="store_true",
                      help="Écrase le fichier existant.")
    sub.add_parser("community",   help="Regroupement des motifs en communautés")

    p_info = sub.add_parser("info", help="État du projet VAME unique du projet EthoFlow")
    p_info.add_argument("--input-dir", default=None,
                        help="Scanne ce dossier pour montrer combien de paires "
                             "seraient utilisées par un futur setup")
    p_info.add_argument("--cropped-dir", default=None)

    sub.add_parser("all", help="Tout enchaîner (très long)")

    args = parser.parse_args()
    {
        "setup":        cmd_setup,
        "align":        cmd_align,
        "trainset":     cmd_trainset,
        "train":        cmd_train,
        "evaluate":     cmd_evaluate,
        "segment":      cmd_segment,
        "motif-videos": cmd_motif_videos,
        "motif-labels": cmd_motif_labels,
        "community":    cmd_community,
        "info":         cmd_info,
        "all":          cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
