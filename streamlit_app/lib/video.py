"""Sonde vidéo, extraction de frames et re-pointage des vidéos déplacées — sans Streamlit.

Trois besoins concrets, tous formulés par le chercheur :

1. Regarder une session avant de la traiter : une vignette et les
   caractéristiques *réelles* de la vidéo (fps, nb de frames, durée,
   dimensions) lues dans le fichier, à comparer à ce qu'affirme la
   metadata. Un décalage de fps corrompt toutes les conversions
   frame → secondes en aval, et c'est bien moins coûteux à repérer ici
   qu'après un run d'inférence.
2. Re-pointer les vidéos qui ont changé de machine ou de disque : le
   README a une entrée Troubleshooting pour une metadata qui porte des
   chemins Windows sur une machine Linux. `find_relinks` retrouve les
   fichiers déplacés et `apply_relinks` réécrit `source_video` dans
   chaque `metadata.yaml`.
3. Dessiner les rectangles d'arène sur une frame, pour la page de
   calibrage au clic.
4. Convertir des paires de clics en géométrie (rectangle d'arène,
   distance en pixels) pour les onglets de calibration — Task 20. Pure et
   testable sans Streamlit ni `streamlit-image-coordinates`, qui n'est
   qu'une source de points `(x, y)` pour ces fonctions.

Ce module ne doit JAMAIS importer Streamlit, ni directement ni
transitivement : les pages l'appellent pour chaque session d'une liste,
et une exception ou une dépendance lourde ici casserait la page entière.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from lib.project import SCRIPTS_DIR
from lib.sessions import load_metadata, session_ids

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from paths import raw_dir  # noqa: E402


@dataclass
class VideoInfo:
    """Caractéristiques réelles d'une vidéo, lues dans le fichier — pas dans la metadata."""
    path: Path
    exists: bool
    fps: float | None
    n_frames: int | None
    width: int | None
    height: int | None
    duration_s: float | None


def _vide(path: Path) -> VideoInfo:
    """`VideoInfo` pour un fichier absent ou illisible : jamais d'exception."""
    return VideoInfo(path=Path(path), exists=False, fps=None, n_frames=None,
                      width=None, height=None, duration_s=None)


def probe(path: Path) -> VideoInfo:
    """Sonde une vidéo pour ses caractéristiques réelles.

    Ne lève jamais : une vidéo manquante ou corrompue est un cas courant
    ici (les pages sondent chaque session d'une liste), pas une
    exception. `cap.release()` est garanti par le `finally`, sinon les
    descripteurs de fichier fuient à chaque rerun Streamlit.
    """
    path = Path(path)
    if not path.is_file():
        return _vide(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return _vide(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not fps or fps <= 0:
            return VideoInfo(path=path, exists=True, fps=None,
                              n_frames=n_frames or None, width=width or None,
                              height=height or None, duration_s=None)
        duration_s = n_frames / fps if n_frames else None
        return VideoInfo(path=path, exists=True, fps=fps, n_frames=n_frames,
                          width=width, height=height, duration_s=duration_s)
    finally:
        cap.release()


def grab_frame(path: Path, index: int = 0) -> np.ndarray | None:
    """Extrait la frame `index`, ou `None` si la vidéo/l'index est invalide."""
    path = Path(path)
    if not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


def frame_png_bytes(path: Path, index: int = 0, max_width: int | None = None) -> bytes | None:
    """Frame encodée en PNG, redimensionnée à `max_width` si fournie, ou `None`."""
    frame = grab_frame(path, index=index)
    if frame is None:
        return None
    if max_width and frame.shape[1] > max_width:
        ratio = max_width / frame.shape[1]
        nouvelle_taille = (max_width, max(1, int(round(frame.shape[0] * ratio))))
        frame = cv2.resize(frame, nouvelle_taille, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        return None
    return buf.tobytes()


def draw_arenas(frame: np.ndarray, coords: dict[str, list[int]]) -> np.ndarray:
    """Dessine les rectangles d'arène sur une copie de `frame` — l'original n'est jamais modifié.

    Une page appelle ceci répétitivement sur la même frame mise en
    cache pendant que l'utilisateur ajuste les coordonnées : muter
    l'original ferait s'accumuler les rectangles.
    """
    sortie = frame.copy()
    for label, rect in coords.items():
        if not rect or len(rect) < 4:
            continue
        x, y, w, h = rect[:4]
        cv2.rectangle(sortie, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(sortie, str(label), (int(x) + 4, int(y) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return sortie


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR (convention OpenCV, celle de `grab_frame`/`draw_arenas`) → RGB.

    `streamlit_image_coordinates` construit l'image affichée via
    `PIL.Image.fromarray`, qui interprète un tableau 3 canaux comme du RGB.
    Sans cette conversion, une frame couleur s'affiche avec le rouge et le
    bleu inversés dans le navigateur (les rectangles verts de `draw_arenas`
    restent verts par coïncidence — canal G symétrique — mais une vraie
    photo, elle, se verrait fausse).
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rect_from_two_points(p1: tuple[int, int], p2: tuple[int, int]) -> list[int]:
    """Deux clics sur des coins opposés → `[x, y, w, h]` (format `crop_arenes.py`).

    Peu importe quel coin est cliqué en premier ou l'ordre haut/bas,
    gauche/droite : `x, y` est toujours le coin haut-gauche du rectangle.
    """
    x0, y0 = p1
    x1, y1 = p2
    x, y = min(x0, x1), min(y0, y1)
    w, h = abs(x1 - x0), abs(y1 - y0)
    return [int(x), int(y), int(w), int(h)]


def distance_from_two_points(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Distance euclidienne en pixels entre deux clics — pour l'échelle px/cm."""
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def draw_scale_line(frame: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> np.ndarray:
    """Dessine le segment de calibration d'échelle sur une copie de `frame`.

    Même précaution que `draw_arenas` : ne jamais muter l'original, une
    page rappelle cette fonction à chaque rerun sur la même frame mise en
    cache pendant que l'utilisateur clique ses deux points.
    """
    sortie = frame.copy()
    pt1 = (int(p1[0]), int(p1[1]))
    pt2 = (int(p2[0]), int(p2[1]))
    cv2.line(sortie, pt1, pt2, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(sortie, pt1, 5, (255, 0, 255), -1)
    cv2.circle(sortie, pt2, 5, (255, 0, 255), -1)
    distance = distance_from_two_points(p1, p2)
    cv2.putText(sortie, f"{distance:.1f} px", (pt1[0] + 8, pt1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    return sortie


def find_relinks(project: Path, videos_dir: Path, ext: str = "mp4") -> list[tuple[str, Path]]:
    """Sessions dont la `source_video` actuelle est absente, avec le fichier retrouvé sous `videos_dir`.

    Cherche `<videos_dir>/<id>.<ext>` où `id` vient de `metadata["id"]`,
    à défaut du nom du dossier de session. Ne retourne que les sessions
    dont la vidéo retrouvée existe réellement — une session dont la
    vidéo actuelle est valide ne doit jamais être proposée au
    re-pointage.
    """
    project = Path(project)
    videos_dir = Path(videos_dir)
    resultats: list[tuple[str, Path]] = []
    for session_id in session_ids(project):
        meta = load_metadata(project, session_id)
        if meta is None:
            continue
        source = meta.get("source_video")
        if source and Path(source).is_file():
            continue
        video_id = meta.get("id") or session_id
        candidat = videos_dir / f"{video_id}.{ext}"
        if candidat.is_file():
            resultats.append((session_id, candidat))
    return resultats


def apply_relinks(project: Path, relinks: list[tuple[str, Path]]) -> int:
    """Réécrit `source_video` dans le `metadata.yaml` de chaque session, sans toucher aux autres clés.

    `sort_keys=False` préserve l'ordre d'origine : la metadata porte les
    colonnes expérimentales du chercheur, les réordonner ou les perdre
    serait un dégât silencieux.
    """
    project = Path(project)
    compte = 0
    for session_id, nouveau_chemin in relinks:
        meta_path = raw_dir(project) / session_id / "metadata.yaml"
        if not meta_path.is_file():
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        meta["source_video"] = str(nouveau_chemin)
        meta_path.write_text(
            yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        compte += 1
    return compte
