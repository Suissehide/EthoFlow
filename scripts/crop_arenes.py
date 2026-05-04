"""
Crop des 4 arènes d'une vidéo brute en 4 vidéos single-animal.

Lit le `metadata.yaml` de la session pour récupérer les coordonnées des arènes,
puis utilise ffmpeg pour extraire chaque ROI dans une vidéo séparée.

Usage:
    python scripts/crop_arenes.py <session_id>

Exemple:
    python scripts/crop_arenes.py 2026-05-04_projet-X_session-001

Requirements:
    - ffmpeg installé et accessible dans le PATH
    - PyYAML
"""
import sys
import yaml
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CROPPED_DIR = ROOT / "data" / "cropped"


def crop_arenes(session_id: str) -> None:
    session_dir = RAW_DIR / session_id
    metadata_path = session_dir / "metadata.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata absent : {metadata_path}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    # Trouver la vidéo source (premier .mp4 ou .avi du dossier)
    videos = list(session_dir.glob("*.mp4")) + list(session_dir.glob("*.avi"))
    if not videos:
        raise FileNotFoundError(f"Aucune vidéo dans {session_dir}")
    source_video = videos[0]
    print(f"Source : {source_video}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg introuvable dans le PATH.")

    output_dir = CROPPED_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    arenes = metadata.get("arenes", [])
    if not arenes:
        raise ValueError("Pas d'arènes dans le metadata.yaml")

    for arene in arenes:
        arene_id = arene["id"]
        coords = arene.get("coords")
        if not coords:
            print(f"⚠️  Coords manquantes pour {arene_id}, skip")
            continue

        x, y, w, h = coords
        output_path = output_dir / f"{arene_id}.mp4"

        # ffmpeg : crop=w:h:x:y ; on garde l'audio si présent (-c:a copy)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(source_video),
            "-vf", f"crop={w}:{h}:{x}:{y}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "copy",
            str(output_path),
        ]
        print(f"\n=== {arene_id} → {output_path.name} ===")
        subprocess.run(cmd, check=True)

    print(f"\n✅ Crop terminé : {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    crop_arenes(sys.argv[1])
