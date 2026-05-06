"""
Crop des 4 arènes d'une vidéo brute en N vidéos single-animal.

⚠️  Outil OPTIONNEL, hors du pipeline d'inférence par défaut.

Le pipeline standard fait de l'inférence DLC multi-animal directement sur
la vidéo entière, puis utilise `assign_arenas.py` pour splitter la sortie.
Pas besoin de cropper la vidéo source pour ça.

Ce script reste utile pour :
- la labellisation manuelle dans la GUI DLC (plus simple en single-animal)
- le fine-tuning ou l'entraînement d'un modèle custom
- l'inspection visuelle d'une arène isolée pour debug

Lit le `metadata.yaml` de la session pour récupérer le chemin de la vidéo
source et les coordonnées des arènes, puis utilise ffmpeg pour extraire
chaque ROI dans une vidéo séparée.

Convention de nommage des sorties :
    data/cropped/<session_id>/<session_id>_<arene_id>.mp4
    ex: data/cropped/OF-M1-20251010-V01/OF-M1-20251010-V01_A1.mp4

Les arènes vides (mouse_id == null) sont ignorées.

Usage:
    python scripts/crop_arenes.py <session_id>

Exemple:
    python scripts/crop_arenes.py OF-M1-20251010-V01

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

        cmd = [
            "ffmpeg", "-y",
            "-i", str(source_video),
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    crop_arenes(sys.argv[1])
