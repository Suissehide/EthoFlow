"""
Crop des 4 arènes d'une vidéo brute en N vidéos single-animal.

Utilisé dans deux flows :

1. **Chemin B (single-animal cropped)** : `run_dlc_inference.py --mode single-animal`
   lit les vidéos produites ici pour faire l'inférence DLC arène par arène.
2. **Labellisation / fine-tuning de modèle DLC custom** : la GUI DLC est plus
   simple à utiliser sur des vidéos single-animal que sur multi-animal.

(Le chemin A — multi-animal sur vidéo entière + assign_arenas — n'utilise pas
crop_arenes.)

Lit le `metadata.yaml` de la session pour récupérer le chemin de la vidéo
source et les coordonnées des arènes (ou retombe sur les
`default_arenes_coords` de `pipeline_config.yaml`), puis utilise ffmpeg
pour extraire chaque ROI dans une vidéo séparée.

Convention de nommage des sorties :
    data/cropped/<session_id>/<session_id>_<arene_id>.mp4
    ex: data/cropped/OF-M1-20251010-V01/OF-M1-20251010-V01_A1.mp4

Les arènes vides (mouse_id == null) sont ignorées.

Usage:
    python scripts/crop_arenes.py <session_id>
    python scripts/crop_arenes.py <s1> <s2> <s3>     # plusieurs sessions
    python scripts/crop_arenes.py --all              # toutes les sessions
    python scripts/crop_arenes.py --all-new          # uniquement celles sans crop

Requirements:
    - ffmpeg installé et accessible dans le PATH
    - PyYAML
"""
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CROPPED_DIR = ROOT / "data" / "cropped"
CONFIG_PATH = ROOT / "configs" / "pipeline_config.yaml"


def load_default_coords() -> dict:
    """Charge default_arenes_coords depuis pipeline_config.yaml si présent."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("default_arenes_coords", {}) or {}


def crop_arenes(session_id: str) -> None:
    session_dir = RAW_DIR / session_id
    metadata_path = session_dir / "metadata.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata absent : {metadata_path}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    # Vidéo source : chemin absolu dans le metadata, ou fallback dans le dossier
    source_video_str = metadata.get("source_video")
    if source_video_str:
        source_video = Path(source_video_str)
    else:
        candidates = list(session_dir.glob("*.mp4")) + list(session_dir.glob("*.avi"))
        if not candidates:
            raise FileNotFoundError(
                f"`source_video` absent du metadata et aucune vidéo dans {session_dir}"
            )
        source_video = candidates[0]

    if not source_video.exists():
        raise FileNotFoundError(f"Vidéo source introuvable : {source_video}")
    print(f"Source : {source_video}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg introuvable dans le PATH.")

    output_dir = CROPPED_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    arenes = metadata.get("arenes", [])
    if not arenes:
        raise ValueError("Pas d'arènes dans le metadata.yaml")

    default_coords = load_default_coords()
    if default_coords:
        print(f"Coords par défaut chargées depuis {CONFIG_PATH.name}")

    # Offset temporel optionnel : skipper les N premières secondes (utile quand
    # la manip commence après quelques secondes de placement des souris).
    start_time_s = float(metadata.get("start_time_s") or 0)
    end_time_s = metadata.get("end_time_s")
    if start_time_s > 0:
        print(f"⏱  start_time_s = {start_time_s}s — les premières secondes "
              f"seront ignorées")
    if end_time_s is not None:
        print(f"⏱  end_time_s   = {end_time_s}s")

    n_cropped = n_skipped = 0
    for arene in arenes:
        arene_id = arene["id"]

        # Skip arènes vides (pas de souris affectée)
        if arene.get("mouse_id") is None:
            print(f"⏭  {arene_id} : arène vide, skip")
            n_skipped += 1
            continue

        coords = arene.get("coords") or default_coords.get(arene_id)
        if not coords:
            print(f"⚠️  {arene_id} : pas de coords (ni metadata, ni default), skip. "
                  f"Lancer `calibrate_arenes.py` pour les définir.")
            n_skipped += 1
            continue

        x, y, w, h = coords
        output_name = f"{session_id}_{arene_id}.mp4"
        output_path = output_dir / output_name

        cmd = ["ffmpeg", "-y"]
        # -ss AVANT -i : seek rapide (à la frame-clé la plus proche, suffisamment
        # précis pour ce cas d'usage)
        if start_time_s > 0:
            cmd += ["-ss", str(start_time_s)]
        cmd += ["-i", str(source_video)]
        if end_time_s is not None:
            duration = float(end_time_s) - start_time_s
            cmd += ["-t", str(duration)]
        cmd += [
            "-vf", f"crop={w}:{h}:{x}:{y}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "copy",
            str(output_path),
        ]
        print(f"\n=== {arene_id} → {output_name} ===")
        subprocess.run(cmd, check=True)
        n_cropped += 1

    print(f"\n✅ Crop terminé : {n_cropped} arène(s) traitée(s), "
          f"{n_skipped} ignorée(s) — {output_dir}")


def list_all_sessions() -> list[str]:
    """Sessions présentes dans data/raw/."""
    if not RAW_DIR.exists():
        return []
    return sorted(
        d.name for d in RAW_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def is_cropped(session_id: str) -> bool:
    """Vrai si data/cropped/<session>/ contient déjà au moins une .mp4."""
    out = CROPPED_DIR / session_id
    return out.exists() and any(out.glob("*.mp4"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop des arènes — single ou batch.")
    parser.add_argument("session_ids", nargs="*",
                        help="Un ou plusieurs session_id")
    parser.add_argument("--all", action="store_true",
                        help="Traiter toutes les sessions de data/raw/")
    parser.add_argument("--all-new", action="store_true",
                        help="Traiter uniquement les sessions sans crop existant")
    args = parser.parse_args()

    if args.all:
        sessions = list_all_sessions()
    elif args.all_new:
        sessions = [s for s in list_all_sessions() if not is_cropped(s)]
    elif args.session_ids:
        sessions = list(args.session_ids)
    else:
        parser.print_help()
        sys.exit(1)

    if not sessions:
        print("Aucune session à traiter.")
        sys.exit(0)

    if len(sessions) > 1:
        print(f"{len(sessions)} session(s) à cropper : {sessions}\n")

    n_ok = n_fail = 0
    for i, session_id in enumerate(sessions, 1):
        if len(sessions) > 1:
            print(f"\n{'='*60}\n[{i}/{len(sessions)}] {session_id}\n{'='*60}")
        try:
            crop_arenes(session_id)
            n_ok += 1
        except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
            print(f"❌ {session_id} : {e}", file=sys.stderr)
            n_fail += 1
            continue

    if len(sessions) > 1:
        print(f"\n✅ Batch terminé : {n_ok} OK, {n_fail} échec(s) sur {len(sessions)}")
    if n_fail > 0:
        sys.exit(1)
