"""Prépare un kit diagnostic à envoyer à Felix / Tony (équipe VAME) pour QC DLC.

Contenu du kit produit :

    dlc_feedback_kit_<date>/
    ├── clips/                       # extraits vidéo courts au milieu de chaque session
    │   ├── <session>_clip_pcut03.mp4    (labeled à pcutoff=0.3, keypoints permissifs)
    │   ├── <session>_clip_pcut06.mp4    (labeled à pcutoff=0.6 = défaut DLC, propre)
    │   └── ...
    ├── config/
    │   └── config.yaml              # config DLC du projet
    ├── labeled_data/                # sous-échantillon zippé du dossier labeled-data
    │   └── labeled_data_subset.zip
    ├── README.txt                   # infos pipeline + questions ouvertes
    └── dlc_feedback_kit_<date>.zip  # zip final prêt à envoyer

Choix par défaut :
    - Extraits pris au MILIEU des sessions (typiquement animal habitué, exploration
      naturelle plutôt que freeze initial ou fatigue de fin).
    - Durée de clip : 20 s à 30 fps = ~600 frames, léger même en H.264.
    - pcutoff=0.3 permet de VOIR ce que le modèle prédit même quand il n'est pas
      sûr. À 0.6 (défaut DLC), les pattes floues disparaissent complètement de
      la vidéo annotée, ce qui masque le vrai comportement du modèle.
    - Deux versions labeled par session (0.3 et 0.6) pour montrer la différence.

Usage :
    # Basique — 1 session, extrait au milieu, pcutoff 0.3 (recommandé pour Felix)
    python scripts/prepare_dlc_feedback_kit.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --sessions BV-970

    # Plusieurs sessions (mix good + bad tracking demandé par Felix)
    python scripts/prepare_dlc_feedback_kit.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --sessions BV-970 BV-971 BV-975

    # Timestamps custom (si un moment particulier illustre mieux le problème)
    python scripts/prepare_dlc_feedback_kit.py \\
        --project-dir D:/ethoflow/projects/bottomview-MCC-2026-06 \\
        --sessions BV-970 --clip-start 620

    # Sans re-générer les labeled videos (si elles existent déjà)
    python scripts/prepare_dlc_feedback_kit.py --project-dir <...> \\
        --sessions BV-970 --skip-labeled-videos

Pré-requis :
    - conda activate dlc  (a besoin de deeplabcut pour create_labeled_video)
    - ffmpeg dans le PATH (pour découper les clips)
    - Le .h5 DLC doit exister pour chaque session dans <project>/data/dlc-output/<session>/
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    add_project_dir_arg,
    dlc_output_dir,
    pipeline_config_path,
    raw_dir,
    resolve_project,
)


def get_video_duration_sec(video: Path) -> float | None:
    """Utilise ffprobe pour lire la durée d'une vidéo. None si échoue."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            stderr=subprocess.STDOUT, text=True,
        )
        return float(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def find_labeled_video(session_out: Path, pcutoff_tag: str) -> Path | None:
    """Cherche un labeled video existant pour un pcutoff donné.

    DLC nomme la sortie <original>_labeled.mp4 et n'encode pas toujours
    pcutoff dans le nom. On stocke une version renommée pour éviter les
    collisions entre passes.
    """
    for p in session_out.glob(f"*_labeled_{pcutoff_tag}.mp4"):
        return p
    # Fallback : nom DLC standard, plus récent
    candidates = list(session_out.glob("*_labeled.mp4"))
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_mtime)
    return None


def generate_labeled_video(dlc_config: str, source_video: Path,
                             session_out: Path, pcutoff: float) -> Path | None:
    """Regénère la labeled video au pcutoff demandé.

    Réutilise le .h5 existant (rapide, ~30 s pour 20 min de vidéo). Renomme
    la sortie en <stem>_labeled_p03.mp4 (ou p06 etc.) pour cohabiter.
    """
    try:
        import deeplabcut as dlc
    except ImportError:
        print(f"❌ deeplabcut non trouvé — active `conda activate dlc`",
              file=sys.stderr)
        sys.exit(1)

    # Vérifie qu'un .h5 existe (sinon create_labeled_video ne peut rien faire)
    if not list(session_out.glob("*.h5")):
        print(f"⚠  {session_out.name} : pas de .h5 DLC, skip labeled video")
        return None

    print(f"    Régénération labeled video pcutoff={pcutoff}...")
    dlc.create_labeled_video(
        dlc_config,
        [str(source_video)],
        destfolder=str(session_out),
        pcutoff=pcutoff,
        draw_skeleton=True,
    )

    # DLC vient d'écraser <stem>_labeled.mp4 → on renomme pour tagger le pcutoff
    tag = f"p{int(pcutoff * 100):02d}"
    latest = sorted(
        session_out.glob("*_labeled.mp4"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not latest:
        print(f"    ⚠  DLC n'a pas produit de labeled video")
        return None
    labeled = latest[0]
    tagged = labeled.with_name(f"{labeled.stem}_{tag}.mp4")
    labeled.rename(tagged)
    return tagged


def extract_clip(video: Path, start_sec: float, duration_sec: float,
                  out_path: Path) -> bool:
    """Découpe un clip via ffmpeg avec re-encode H.264 (compatible partout)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec), "-i", str(video),
        "-t", str(duration_sec),
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        # Filter: garde audio si présent, muet sinon
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    ⚠  ffmpeg échec pour {out_path.name} : {e}",
              file=sys.stderr)
        return False


def read_session_metadata(project: Path, session_id: str) -> dict | None:
    """Lit metadata.yaml d'une session, None si absent."""
    meta_path = raw_dir(project) / session_id / "metadata.yaml"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return yaml.safe_load(f) or {}


def read_dlc_config_path(project: Path) -> str | None:
    """Lit dlc_project_config depuis pipeline_config.yaml."""
    cfg_path = pipeline_config_path(project)
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("dlc_project_config")


def zip_labeled_data_subset(dlc_project: Path, out_zip: Path,
                              max_sessions: int) -> int:
    """Zippe un sous-ensemble du dossier labeled-data/ du projet DLC.

    Prend au plus `max_sessions` sous-dossiers (souvent 1 par vidéo labellisée),
    pris dans l'ordre alphabétique pour reproductibilité.
    """
    labeled_data_dir = dlc_project / "labeled-data"
    if not labeled_data_dir.exists():
        print(f"⚠  labeled-data/ introuvable dans {dlc_project}")
        return 0
    subdirs = sorted([d for d in labeled_data_dir.iterdir() if d.is_dir()])
    if not subdirs:
        return 0
    subset = subdirs[:max_sessions]
    print(f"  Zippage labeled-data subset : {len(subset)} sous-dossier(s)")
    n_files = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in subset:
            for f in d.rglob("*"):
                if f.is_file():
                    arcname = f.relative_to(labeled_data_dir.parent)
                    zf.write(f, arcname)
                    n_files += 1
    return n_files


def build_readme(kit_dir: Path, project: Path, sessions: list[str],
                  dlc_project: Path, exposure_ms: float | None) -> None:
    """Écrit un README.txt qui explique le kit à Felix / Tony."""
    lines = [
        "DLC bottom-view feedback kit",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Project (EthoFlow): {project.name}",
        f"Project (DLC):      {dlc_project.name}",
        "",
        "=" * 60,
        "Contents",
        "=" * 60,
        "",
        "clips/",
        "    Two labeled clips per session, extracted from the middle",
        "    of the recording (typically the exploratory phase).",
        "    - *_p30.mp4 : pcutoff=0.3, permissive display — shows what",
        "                  the network is predicting even when uncertain.",
        "                  This is where paw failures are visible.",
        "    - *_p60.mp4 : pcutoff=0.6, DLC default — clean production view.",
        "",
        "config/",
        "    config.yaml of the DLC project (architecture, bodyparts,",
        "    train/test split, iteration counters, etc.).",
        "",
        "labeled_data/",
        "    Subset of the labeled-data/ folder from the DLC project.",
        "    Contains ground-truth CSVs + labeled frames for the first",
        "    few videos of the training set.",
        "",
        "=" * 60,
        "Pipeline overview",
        "=" * 60,
        "",
        "Backbone         : HRNet-w32 (SuperAnimal Quadruped transfer)",
        "Detector         : Faster R-CNN ResNet50 FPN v2",
        "Framework        : DeepLabCut 3.x (pytorch backend)",
        "Bodyparts        : 12 keypoints (nose, ears, front paws L/R,",
        "                   hind paws L/R, tail base, tail mid, tail tip,",
        "                   center, left flank)",
        "Camera view      : bottom-view IR through transparent floor",
        "Camera framerate : 30 fps",
    ]
    if exposure_ms is not None:
        lines.append(f"Camera exposure  : {exposure_ms:.1f} ms")
    else:
        lines.append("Camera exposure  : (to be reported)")
    lines += [
        "",
        "Sessions included in this kit:",
    ]
    for sid in sessions:
        lines.append(f"    - {sid}")
    lines += [
        "",
        "=" * 60,
        "What we would appreciate feedback on",
        "=" * 60,
        "",
        "1. Camera exposure — do you see any evidence of paw motion blur",
        "   in the p30 clips? If so we suspect this is a hardware fix",
        "   rather than a training-data fix (per Felix's initial note).",
        "",
        "2. Iterative refinement timing — are we running outlier extraction",
        "   too early, given that our current training set only covers a",
        "   limited number of animals? What breadth of animals / behaviors",
        "   did the training set for your VAME publications typically cover?",
        "",
        "3. Any obvious labeling error or bodypart definition inconsistency",
        "   visible in the labeled_data/ archive?",
        "",
        "Thank you for taking the time to look at this.",
        "",
    ]
    (kit_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_project_dir_arg(parser, required=True)
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Sessions à inclure (ex : BV-970 BV-971)")
    parser.add_argument("--clip-start", type=float, default=None,
                        help="Timestamp de début en secondes (défaut : "
                             "milieu de la session). Applique à toutes les "
                             "sessions passées.")
    parser.add_argument("--clip-duration", type=float, default=20.0,
                        help="Durée du clip en secondes (défaut : 20)")
    parser.add_argument("--pcutoffs", nargs="+", type=float,
                        default=[0.3, 0.6],
                        help="Seuils pcutoff à générer (défaut : 0.3 et 0.6)")
    parser.add_argument("--skip-labeled-videos", action="store_true",
                        help="N'appelle pas create_labeled_video, réutilise "
                             "les *_labeled_p*.mp4 existants (rapide)")
    parser.add_argument("--labeled-data-sessions", type=int, default=3,
                        help="Nombre de sous-dossiers labeled-data à zipper "
                             "(défaut : 3, met 0 pour ne pas zipper)")
    parser.add_argument("--exposure-ms", type=float, default=None,
                        help="Temps d'exposition caméra en ms, injecté "
                             "dans le README (info clé pour Felix)")
    parser.add_argument("--kit-name", default=None,
                        help="Nom du dossier de sortie (défaut : "
                             "dlc_feedback_kit_YYYYMMDD)")
    parser.add_argument("--no-zip", action="store_true",
                        help="Ne crée pas le zip final (utile pour debug)")
    args = parser.parse_args()

    project = resolve_project(args)
    dlc_config_str = read_dlc_config_path(project)
    if not dlc_config_str:
        print(f"❌ dlc_project_config manquant dans {pipeline_config_path(project)}",
              file=sys.stderr)
        sys.exit(1)
    dlc_config = Path(dlc_config_str)
    dlc_project = dlc_config.parent

    if not args.skip_labeled_videos and not dlc_config.exists():
        print(f"❌ Config DLC introuvable : {dlc_config}", file=sys.stderr)
        print(f"   → utilise --skip-labeled-videos si tu veux juste réutiliser",
              file=sys.stderr)
        sys.exit(1)

    kit_name = args.kit_name or f"dlc_feedback_kit_{datetime.now():%Y%m%d}"
    kit_dir = project / "outputs" / kit_name
    kit_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = kit_dir / "clips"; clips_dir.mkdir(exist_ok=True)
    config_dir = kit_dir / "config"; config_dir.mkdir(exist_ok=True)
    labeled_data_dir = kit_dir / "labeled_data"; labeled_data_dir.mkdir(exist_ok=True)

    print(f"Projet EthoFlow : {project}")
    print(f"Projet DLC      : {dlc_project}")
    print(f"Kit de sortie   : {kit_dir}\n")

    # --- 1) Génère / retrouve les labeled videos + extrait un clip par pcutoff
    print(f"[1/4] Génération des clips ({len(args.sessions)} sessions × "
          f"{len(args.pcutoffs)} pcutoffs)")
    for session_id in args.sessions:
        print(f"\n  · {session_id}")
        meta = read_session_metadata(project, session_id)
        if meta is None:
            print(f"    ⚠  metadata absente, skip session")
            continue
        source_video = Path(meta.get("source_video", ""))
        if not source_video.exists():
            print(f"    ⚠  source_video introuvable : {source_video}")
            continue
        session_out = dlc_output_dir(project) / session_id
        if not session_out.exists():
            print(f"    ⚠  dossier DLC output absent : {session_out}")
            continue

        # Timestamp de découpe : middle par défaut
        duration = get_video_duration_sec(source_video)
        if args.clip_start is not None:
            clip_start = args.clip_start
        elif duration is not None:
            clip_start = max(0, (duration - args.clip_duration) / 2)
            print(f"    Durée session : {duration:.0f}s  →  clip start = "
                  f"{clip_start:.0f}s (milieu)")
        else:
            clip_start = 300.0  # fallback 5 min
            print(f"    ⚠  ffprobe indispo, fallback clip_start=300s")

        for pcutoff in args.pcutoffs:
            tag = f"p{int(pcutoff * 100):02d}"
            if args.skip_labeled_videos:
                labeled = find_labeled_video(session_out, tag)
                if labeled is None:
                    print(f"    ⚠  pcutoff={pcutoff} : pas de labeled video "
                          f"préexistante, skip")
                    continue
            else:
                labeled = generate_labeled_video(
                    str(dlc_config), source_video, session_out, pcutoff,
                )
                if labeled is None:
                    continue

            clip_out = clips_dir / f"{session_id}_clip_{tag}.mp4"
            print(f"    Découpe clip {tag} → {clip_out.name}")
            if extract_clip(labeled, clip_start, args.clip_duration, clip_out):
                size_mb = clip_out.stat().st_size / 1e6
                print(f"    ✅  {size_mb:.1f} MB")

    # --- 2) Copie du config.yaml DLC
    print(f"\n[2/4] Copie du config DLC")
    if dlc_config.exists():
        shutil.copy(dlc_config, config_dir / "config.yaml")
        print(f"  ✅ {config_dir / 'config.yaml'}")
    else:
        print(f"  ⚠  {dlc_config} introuvable")

    # --- 3) Sous-échantillon de labeled-data/
    print(f"\n[3/4] Sous-échantillon labeled-data/")
    if args.labeled_data_sessions > 0:
        zip_path = labeled_data_dir / "labeled_data_subset.zip"
        n = zip_labeled_data_subset(dlc_project, zip_path,
                                     args.labeled_data_sessions)
        if n > 0:
            size_mb = zip_path.stat().st_size / 1e6
            print(f"  ✅ {zip_path.name} : {n} fichiers, {size_mb:.1f} MB")
    else:
        print(f"  · skip (--labeled-data-sessions=0)")

    # --- 4) README + zip final
    print(f"\n[4/4] README + zip final")
    build_readme(kit_dir, project, args.sessions, dlc_project,
                  args.exposure_ms)
    print(f"  ✅ {kit_dir / 'README.txt'}")

    if not args.no_zip:
        final_zip = kit_dir.parent / f"{kit_dir.name}.zip"
        print(f"  Zippage final → {final_zip.name}")
        with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in kit_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(kit_dir.parent))
        size_mb = final_zip.stat().st_size / 1e6
        print(f"  ✅ {final_zip}  ({size_mb:.1f} MB)")

    print(f"\n✅ Kit prêt : {kit_dir}\n")
    print(f"Prochaine étape : envoie {kit_dir.name}.zip à Felix + Tony.")


if __name__ == "__main__":
    main()
