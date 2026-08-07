"""Calibre l'échelle pixels → centimètres de ton setup caméra.

Nécessaire pour la détection de vitesse aberrante dans
`prepare_vame_input_custom.py`. Sans cette échelle, on ne peut pas
convertir un déplacement en pixels en une vitesse en m/s, et donc pas
juger si un saut de label est physiquement plausible.

Recommandation Tony (VAME/LIN) :

    « Tu peux prendre une photo d'une règle avec ton setup et convertir
    les pixels en cm ou en mètres. Ensuite tu peux calculer la vitesse
    en m/s et fixer que, disons, 4 ou 5 m/s est la vitesse maximale que
    tu autoriserais. Tu peux aussi prendre les dimensions connues de
    l'arène, mais plus l'objet est grand, plus la distorsion de lentille
    a d'influence. »

D'où la préférence pour une règle (petit objet, distorsion minime) sur
les dimensions de l'arène.

---------------------------------------------------------------------
Usage
---------------------------------------------------------------------

    # Interactif — propose la liste des vidéos du projet
    python scripts/calibrate_scale.py

    # Sur une session précise du projet (pas de chemin à taper)
    python scripts/calibrate_scale.py --session BV-970 --known-cm 10

    # Depuis une photo de règle
    python scripts/calibrate_scale.py \\
        --image D:/EthoFlow/calibration/regle.png --known-cm 10

    # Depuis une vidéo quelconque
    python scripts/calibrate_scale.py \\
        --video D:/data/bottom_view/970.mp4 --frame 0 --known-cm 10

    # Si tu connais déjà la valeur, écris-la directement
    python scripts/calibrate_scale.py --project-dir <...> --set 12.5

Sans argument, le script liste les vidéos des sessions du projet et te
laisse en choisir une — inutile de retrouver un chemin.

Une fenêtre s'ouvre : **clique deux points** séparés par la distance
réelle connue (les deux extrémités du segment de règle que tu mesures),
puis ferme la fenêtre. Le script calcule px/cm et l'écrit dans
`configs/pipeline_config.yaml` du projet, sous la clé `px_per_cm`.

`prepare_vame_input_custom.py` lit ensuite cette valeur automatiquement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import pipeline_config_path, raw_dir  # noqa: E402
from interactive import (  # noqa: E402
    add_no_prompt_arg,
    prompt,
    prompt_existing_path,
    resolve_or_prompt_project,
)


def list_session_videos(project: Path) -> list[tuple[str, Path]]:
    """Liste les (session_id, vidéo source) du projet, vidéos existantes.

    Évite de faire taper un chemin complet alors que le projet connaît
    déjà ses vidéos via les metadata.yaml.
    """
    out = []
    rd = raw_dir(project)
    if not rd.exists():
        return out
    for session_dir in sorted(rd.iterdir()):
        meta_path = session_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f) or {}
        except Exception:
            continue
        src = meta.get("source_video")
        if not src:
            continue
        p = Path(src)
        if p.exists():
            out.append((meta.get("session_id") or session_dir.name, p))
    return out


def pick_two_points(image_path: Path | None = None,
                     video_path: Path | None = None,
                     frame_index: int = 0) -> tuple[float, float] | None:
    """Ouvre une fenêtre et laisse l'utilisateur cliquer deux points.

    Renvoie la distance en pixels entre les deux clics, ou None si
    l'utilisateur a fermé sans cliquer deux fois.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("❌ OpenCV requis pour la sélection interactive. "
              "Utilise --set <valeur> si tu connais déjà px/cm.",
              file=sys.stderr)
        return None

    if image_path is not None:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Impossible de lire l'image {image_path}", file=sys.stderr)
            return None
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo {video_path}",
                  file=sys.stderr)
            return None
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, img = cap.read()
        cap.release()
        if not ok:
            print(f"❌ Impossible de lire la frame {frame_index}",
                  file=sys.stderr)
            return None

    points: list[tuple[int, int]] = []
    display = img.copy()
    win = "Clique 2 points a distance connue  |  r = reset  |  q/ESC = valider"

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 255, 0), 2)
                d = float(np.hypot(points[1][0] - points[0][0],
                                    points[1][1] - points[0][1]))
                cv2.putText(display, f"{d:.1f} px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print("\nFenêtre ouverte :")
    print("  · clique les 2 extrémités de la distance connue")
    print("  · 'r' pour recommencer")
    print("  · 'q' ou ESC pour valider et fermer\n")

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            points.clear()
            display = img.copy()
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    if len(points) != 2:
        print("⚠ Il faut exactement 2 points.", file=sys.stderr)
        return None
    import math
    dist_px = math.hypot(points[1][0] - points[0][0],
                          points[1][1] - points[0][1])
    return dist_px, 0.0


def write_scale(project: Path, px_per_cm: float) -> Path:
    """Écrit px_per_cm dans configs/pipeline_config.yaml."""
    cfg_path = pipeline_config_path(project)
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    cfg["px_per_cm"] = round(float(px_per_cm), 3)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return cfg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Projet EthoFlow où écrire l'échelle. "
                             "Demandé si absent.")
    parser.add_argument("--session", type=str, default=None,
                        help="ID de session du projet (ex : BV-970). Prend "
                             "sa vidéo source depuis la metadata — pas de "
                             "chemin à taper.")
    parser.add_argument("--image", type=Path, default=None,
                        help="Image de calibration (photo d'une règle).")
    parser.add_argument("--video", type=Path, default=None,
                        help="Vidéo dont on extrait une frame pour calibrer.")
    parser.add_argument("--frame", type=int, default=0,
                        help="Index de la frame à extraire avec --video.")
    parser.add_argument("--known-cm", type=float, default=None,
                        help="Distance réelle en cm entre les 2 points "
                             "que tu vas cliquer.")
    parser.add_argument("--set", type=float, default=None, dest="set_value",
                        help="Écrit directement cette valeur px/cm sans "
                             "passer par la sélection interactive.")
    add_no_prompt_arg(parser)
    args = parser.parse_args()

    project = resolve_or_prompt_project(args)

    # ---- Mode direct ----
    if args.set_value is not None:
        cfg_path = write_scale(project, args.set_value)
        print(f"✅ px_per_cm = {args.set_value} écrit dans {cfg_path}")
        return

    # ---- Source de l'image ----
    image, video = args.image, args.video

    # --session : reprend la vidéo de cette session depuis sa metadata
    if video is None and image is None and args.session:
        matches = [(sid, v) for sid, v in list_session_videos(project)
                   if sid == args.session or sid.endswith(f"-{args.session}")]
        if not matches:
            print(f"❌ Session '{args.session}' introuvable dans {project}, "
                  f"ou sa vidéo n'existe plus.", file=sys.stderr)
            sys.exit(1)
        video = matches[0][1]
        print(f"ℹ  Vidéo de la session {matches[0][0]} : {video.name}")

    if image is None and video is None:
        if args.no_prompt:
            print("❌ --session, --image, --video ou --set requis en mode "
                  "--no-prompt.", file=sys.stderr)
            sys.exit(1)

        # Menu des vidéos déjà connues du projet — évite de taper un chemin
        sessions = list_session_videos(project)
        if sessions:
            print("Sur quelle vidéo veux-tu calibrer ?")
            shown = sessions[:15]
            for i, (sid, v) in enumerate(shown, start=1):
                print(f"  {i}. {sid}  ({v.name})")
            if len(sessions) > len(shown):
                print(f"     … et {len(sessions) - len(shown)} autre(s)")
            n = len(shown)
            print(f"  {n + 1}. Une photo de règle (image)")
            print(f"  {n + 2}. Une autre vidéo (chemin libre)")
            while True:
                choice = prompt("Choix", default="1")
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= n:
                        video = shown[idx - 1][1]
                        break
                    if idx == n + 1:
                        image = prompt_existing_path("Chemin de l'image",
                                                      must_exist=True)
                        break
                    if idx == n + 2:
                        video = prompt_existing_path("Chemin de la vidéo",
                                                      must_exist=True)
                        break
                match = [v for sid, v in sessions if sid == choice]
                if match:
                    video = match[0]
                    break
                print("  ⚠ choix invalide")
        else:
            print("Aucune session avec vidéo dans ce projet.")
            print("Source de calibration :")
            print("  1. Une image (photo d'une règle sur le plancher)")
            print("  2. Une vidéo")
            choice = prompt("Source", default="1", choices=["1", "2"])
            if choice == "1":
                image = prompt_existing_path("Chemin de l'image",
                                              must_exist=True)
            else:
                video = prompt_existing_path("Chemin de la vidéo",
                                              must_exist=True)

    # ---- Distance connue ----
    known_cm = args.known_cm
    if known_cm is None:
        if args.no_prompt:
            print("❌ --known-cm requis en mode --no-prompt.", file=sys.stderr)
            sys.exit(1)
        known_cm = float(prompt(
            "Distance réelle entre les 2 points que tu vas cliquer, en cm",
            default="10",
        ))

    result = pick_two_points(image, video, args.frame)
    if result is None:
        sys.exit(1)
    dist_px, _ = result

    px_per_cm = dist_px / known_cm
    print(f"\nDistance mesurée : {dist_px:.1f} px pour {known_cm} cm")
    print(f"Échelle          : {px_per_cm:.3f} px/cm")
    print(f"                   (1 px = {1 / px_per_cm * 10:.2f} mm)")

    cfg_path = write_scale(project, px_per_cm)
    print(f"\n✅ Écrit dans {cfg_path}")
    print("\nÉtape suivante — la détection de vitesse est maintenant active :")
    print(f"  python scripts/prepare_vame_input_custom.py "
          f"--project-dir {project}")


if __name__ == "__main__":
    main()
