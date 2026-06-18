"""Test du preprocessing avec MOG2 (background adaptatif) pour gérer les
reflets dynamiques que le template statique de preprocess_test.py ne sait
pas modéliser.

Pourquoi MOG2 plutôt qu'un template fixe :

    preprocess_test.py calcule UN background unique (médiane ou percentile)
    sur quelques centaines de frames, puis fait |frame - background|. Si
    l'arène a des reflets qui CHANGENT au cours du temps (sol vernis +
    illumination IR mobile selon position de la souris), un template fixe
    n'arrive pas à les capturer → ils ressortent comme foreground et
    polluent l'image.

    MOG2 maintient un modèle gaussien par pixel et l'update au fil de la
    vidéo. Chaque pixel apprend SA propre distribution de valeurs typiques :
    si un pixel oscille régulièrement entre 200 et 240 (à cause des reflets),
    cet intervalle entre dans son "background". Seul ce qui sort de cet
    intervalle (= la souris) est marqué foreground.

Deux modes en sortie :

    - Par défaut : masque binaire (MOG2 décide foreground/background) +
      CLAHE sur la zone foreground uniquement. Souris sur fond noir net.
    - Avec --no-mask : utilise `getBackgroundImage()` de MOG2 comme template
      adaptatif pour faire un absdiff classique. Préserve plus de détails
      de gradient mais peut laisser des artefacts de reflets résiduels.

Paramètres clés à tweaker selon le résultat :

    --history          plus long = modèle plus stable, moins réactif aux
                       reflets dynamiques mais aussi moins réactif si la
                       souris stationne (elle finit par "rentrer" dans le BG)
    --var-threshold    plus bas = plus sensible (capture une souris peu
                       contrastée mais aussi plus de bruit)
    --morph-kernel     ouverture morphologique : enlève les petits points
                       parasites. Trop grand = érode aussi la souris (et
                       les pattes !)
    --warmup           nb de frames aléatoires pour entraîner MOG2 avant le
                       sample. Plus c'est haut, plus le modèle de background
                       est précis dès le premier frame du sample.

Usage :
    python scripts/dlc_bottomview/preprocess_test_mog2.py \\
        --video "E:\\data\\bottom_view\\08062026\\970.mp4"

    # tweak (si la souris disparaît quand elle stationne) :
    python scripts/dlc_bottomview/preprocess_test_mog2.py \\
        --video "E:\\data\\bottom_view\\08062026\\970.mp4" \\
        --history 1000 --var-threshold 12

    # mode --no-mask pour comparer avec preprocess_test.py classique :
    python scripts/dlc_bottomview/preprocess_test_mog2.py \\
        --video "E:\\data\\bottom_view\\08062026\\970.mp4" \\
        --no-mask
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def warmup_mog2(
    fgbg: cv2.BackgroundSubtractorMOG2,
    video_path: Path,
    n_warmup: int,
) -> None:
    """Entraîne MOG2 sur des frames aléatoires uniformément distribuées
    dans la vidéo. Chaque pixel apprend SA distribution de valeurs.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(total, min(n_warmup, total), replace=False))
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgbg.apply(gray)
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test preprocessing MOG2 + CLAHE pour arènes avec "
        "reflets dynamiques.",
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("preprocess_test_mog2"),
    )
    parser.add_argument("--start", type=float, default=60.0,
                        help="Début du sample en secondes")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Durée du sample en secondes")
    parser.add_argument(
        "--history", type=int, default=500,
        help="MOG2 history — nb frames sur lesquelles le modèle adaptatif "
        "raisonne (défaut: 500, soit ~16s à 30 fps)",
    )
    parser.add_argument(
        "--var-threshold", type=float, default=16.0,
        help="MOG2 varThreshold — seuil de variance au-dessus duquel un "
        "pixel est foreground (défaut: 16)",
    )
    parser.add_argument(
        "--warmup", type=int, default=500,
        help="Nb de frames de warmup MOG2 avant le sample (défaut: 500)",
    )
    parser.add_argument(
        "--morph-kernel", type=int, default=3,
        help="Taille du kernel pour morpho-ouverture sur le mask "
        "(défaut: 3 ; mettre 0 pour désactiver)",
    )
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Au lieu d'appliquer le mask binaire, fait absdiff avec le "
        "background adaptatif (préserve les gradients)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=-1.0,
        help="Learning rate MOG2 (défaut: -1 = auto basé sur --history). "
        "Mets une valeur faible (ex: 0.0001) pour figer le modèle quasi "
        "complètement pendant le processing — utile si la souris stationne "
        "longtemps et finit par être absorbée dans le background (ghosting). "
        "Recommandé : warmup suffisant + learning-rate très bas.",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"❌ Vidéo introuvable : {args.video}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1) MOG2 setup + warmup -----
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=False,
    )
    print(
        f"MOG2 warmup sur {args.warmup} frames "
        f"(history={args.history}, varThreshold={args.var_threshold})..."
    )
    warmup_mog2(fgbg, args.video, args.warmup)
    print("✅ Warmup done.\n")

    bg = fgbg.getBackgroundImage()
    bg_path = args.out_dir / f"{args.video.stem}_background_mog2.png"
    cv2.imwrite(str(bg_path), bg)
    print(f"  Background estimé : {bg_path}\n")

    # ----- 2) Process le sample -----
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start * fps)
    n_frames = int(args.duration * fps)
    n_frames = min(n_frames, total - start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    side_path = args.out_dir / f"{args.video.stem}_sidebyside_mog2.mp4"
    side_writer = cv2.VideoWriter(
        str(side_path), fourcc, fps, (width * 2, height), isColor=True,
    )
    proc_path = args.out_dir / f"{args.video.stem}_preprocessed_mog2.mp4"
    proc_writer = cv2.VideoWriter(
        str(proc_path), fourcc, fps, (width, height), isColor=False,
    )

    clahe = cv2.createCLAHE(
        clipLimit=args.clip_limit,
        tileGridSize=(args.tile_size, args.tile_size),
    )
    kernel = (
        np.ones((args.morph_kernel, args.morph_kernel), np.uint8)
        if args.morph_kernel > 0 else None
    )

    lr_label = (
        "auto" if args.learning_rate < 0 else f"{args.learning_rate:g}"
    )
    print(
        f"Processing {n_frames} frames depuis t={args.start}s, "
        f"mode={'absdiff (--no-mask)' if args.no_mask else 'binary mask'}, "
        f"learning_rate={lr_label}..."
    )
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # MOG2 continue à updater son modèle pendant qu'on consomme.
        # learning_rate très bas = quasi figé = pas d'absorption de la souris.
        fgmask = fgbg.apply(gray, learningRate=args.learning_rate)

        if args.no_mask:
            # absdiff avec le background adaptatif courant
            current_bg = fgbg.getBackgroundImage()
            enhanced = clahe.apply(cv2.absdiff(gray, current_bg))
        else:
            # morpho-ouverture pour nettoyer les petits points
            if kernel is not None:
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            # mask binaire : souris sur fond noir
            masked = cv2.bitwise_and(gray, gray, mask=fgmask)
            enhanced = clahe.apply(masked)

        # Side-by-side
        proc_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "ORIGINAL", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        mode_label = "MOG2 + ABSDIFF" if args.no_mask else "MOG2 + MASK"
        cv2.putText(proc_bgr, mode_label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        side_writer.write(np.hstack([frame, proc_bgr]))
        proc_writer.write(enhanced)

        if i % 200 == 0 and i > 0:
            print(f"  {i}/{n_frames} ({100 * i / n_frames:.0f}%)")

    cap.release()
    side_writer.release()
    proc_writer.release()

    print(f"\n✅ Background MOG2     : {bg_path}")
    print(f"✅ Side-by-side video  : {side_path}")
    print(f"✅ Preprocessed video  : {proc_path}")
    print(
        "\nCheck visuel :\n"
        "  - La souris reste-t-elle visible quand elle traverse les zones\n"
        "    à reflets (les bords proches des panneaux IR) ?\n"
        "  - Disparaît-elle quand elle stationne ? (Si oui : --history plus haut)\n"
        "  - Y a-t-il des petites taches blanches autres que la souris ?\n"
        "    (Si oui : --morph-kernel 5 ou plus)\n"
        "  - Mode mask vs no-mask : lequel donne le rendu le plus propre\n"
        "    pour DLC ? (DLC apprécie généralement le contraste binaire,\n"
        "    mais peut perdre du signal de pattes peu différenciées)\n"
    )


if __name__ == "__main__":
    main()
