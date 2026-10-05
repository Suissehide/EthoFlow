"""Y a-t-il de l'information à l'intérieur de la silhouette de la souris ?

Question posée avant de prétraiter des vidéos (inversion, CLAHE…) dans
l'espoir de mieux voir les pattes arrière en vue de dessous.

Deux familles de retouches, deux réponses différentes :

  · **Globales et réversibles** — inversion, gamma, luminosité, contraste
    global. Elles ne créent aucune information : chaque niveau de gris
    est envoyé sur un autre, de façon bijective. Le réseau apprend
    simplement des filtres transformés. Inutile pour le modèle.

  · **Locales** — CLAHE (égalisation adaptative par tuiles). Elles
    redistribuent la dynamique *région par région*, donc peuvent étirer
    les 15 niveaux de gris où vit l'intérieur de la silhouette sur toute
    la plage. Utile SI ces niveaux contiennent quelque chose.

Le script tranche cette condition. Il isole la silhouette de la souris
sur quelques frames, mesure la dispersion des niveaux de gris À
L'INTÉRIEUR (bord exclu, il mélange souris et fond), et produit pour
chaque frame une mosaïque : original, inversé, CLAHE, et un zoom sur la
souris étiré sur sa propre plage — la vue la plus révélatrice.

Verdict, sur le « relief » (p99 − médiane de la silhouette, après un
filtre médian qui efface le bruit capteur) :
  · relief < 8 niveaux → intérieur uniforme, rien à récupérer : aucune
    retouche n'aidera, seul un changement d'éclairage peut faire
    apparaître les pattes ;
  · relief 8-20 → structures faibles, à juger à l'œil sur les mosaïques ;
  · relief > 20 → des structures claires existent (coussinets…), un
    prétraitement CLAHE mérite d'être testé.

Usage :
    python scripts/dlc_model-training/preview_contraste.py \\
        --video D:/EthoFlow/data/20260716_CUS_Ang2/1.mp4

    # Plus de frames, autre dossier de sortie
    python scripts/dlc_model-training/preview_contraste.py \\
        --video <...> --frames 8 --out D:/EthoFlow/preview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Relief = p99 - médiane dans la silhouette, après filtre médian 5x5.
# Calibré sur des vidéos synthétiques : bruit capteur seul ≈ 2-6 niveaux,
# coussinets faiblement contrastés ≈ 10-15, nettement contrastés > 20.
RELIEF_NUL = 8
RELIEF_FRANC = 20


def silhouette(gris: np.ndarray, erosion_px: int = 3) -> np.ndarray | None:
    """Masque de la souris : plus grande composante sombre, bord érodé.

    Otsu sépare la souris (sombre) du plancher (clair). On ne garde que
    la plus grande composante — les coins et marquages de l'arène peuvent
    aussi être sombres — puis on érode pour exclure le liseré de bord,
    dont les pixels mélangent souris et fond et gonfleraient
    artificiellement la dispersion.
    """
    import cv2

    _, masque = cv2.threshold(gris, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, etiquettes, stats, _ = cv2.connectedComponentsWithStats(masque, 8)
    if n <= 1:
        return None
    plus_grande = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (etiquettes == plus_grande).astype(np.uint8)
    if erosion_px > 0:
        noyau = np.ones((2 * erosion_px + 1,) * 2, np.uint8)
        m = cv2.erode(m, noyau)
    return m.astype(bool) if m.any() else None


def stats_interieur(gris: np.ndarray, masque: np.ndarray) -> dict:
    """Distribution des niveaux de gris à l'intérieur de la silhouette.

    La mesure qui compte est le **relief** : p99 − médiane. Les pattes
    sont de petits objets — quelques pourcents des pixels de la
    silhouette — donc une dispersion globale (p95 − p5) passe à côté :
    elle est dominée par le remplissage du corps. Le relief demande au
    contraire « existe-t-il une petite fraction de pixels nettement plus
    claire que le corps ? », ce qui est exactement la signature d'un
    coussinet sous un pelage sombre.

    On filtre d'abord par une médiane 5×5 : sans ça, le bruit capteur
    et la compression vidéo produisent à eux seuls un relief de quelques
    niveaux, indiscernable d'une structure réelle. La médiane efface ce
    bruit pixel à pixel mais conserve des taches de la taille d'une
    patte.
    """
    import cv2

    lisse = cv2.medianBlur(gris, 5)
    v = lisse[masque].astype(float)
    p1, p5, p50, p95, p99 = np.percentile(v, [1, 5, 50, 95, 99])
    return {
        "n_px": int(v.size),
        "p5": p5, "p50": p50, "p95": p95, "p1": p1, "p99": p99,
        "dispersion": p95 - p5,
        "relief": p99 - p50,
        # « Plat au plancher » : part des pixels collés au minimum de la
        # silhouette. Relatif plutôt qu'un seuil absolu, car le noir
        # d'une caméra réelle n'est presque jamais à 0.
        "pct_plat": 100.0 * float((v <= p1 + 2).mean()),
    }


def mosaique(gris: np.ndarray, masque: np.ndarray, st: dict,
              clip: float, tuile: int) -> np.ndarray:
    """Original | inversé | CLAHE, puis zoom souris original / étiré / CLAHE."""
    import cv2

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tuile, tuile))
    g_clahe = clahe.apply(gris)
    g_inv = 255 - gris

    ys, xs = np.where(masque)
    marge = 40
    y0, y1 = max(ys.min() - marge, 0), min(ys.max() + marge, gris.shape[0])
    x0, x1 = max(xs.min() - marge, 0), min(xs.max() + marge, gris.shape[1])

    def zoom(img):
        z = img[y0:y1, x0:x1]
        return cv2.resize(z, None, fx=3, fy=3,
                          interpolation=cv2.INTER_NEAREST)

    # Étirement linéaire sur la plage PROPRE de la souris : tout ce qui
    # est plus clair que son p99 sature en blanc, ce qui libère toute la
    # dynamique pour l'intérieur de la silhouette.
    lo, hi = st["p1"], max(st["p99"], st["p1"] + 1)
    etire = np.clip((gris.astype(float) - lo) / (hi - lo) * 255, 0, 255)
    etire = etire.astype(np.uint8)

    def legende(img, texte):
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(out, texte, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return out

    h = 480
    def red(img):
        return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))

    haut = np.hstack([legende(red(gris), "original"),
                      legende(red(g_inv), "inverse"),
                      legende(red(g_clahe), f"CLAHE clip={clip}")])
    bas = [legende(zoom(gris), "zoom original"),
           legende(zoom(etire), "zoom etire sur la souris"),
           legende(zoom(g_clahe), "zoom CLAHE")]
    bh = max(b.shape[0] for b in bas)
    bas = [cv2.resize(b, (int(b.shape[1] * bh / b.shape[0]), bh)) for b in bas]
    bas = np.hstack(bas)
    # Aligne les largeurs des deux rangées
    w = max(haut.shape[1], bas.shape[1])
    def pad(img):
        return cv2.copyMakeBorder(img, 0, 0, 0, w - img.shape[1],
                                  cv2.BORDER_CONSTANT, value=(40, 40, 40))
    return np.vstack([pad(haut), pad(bas)])


def verdict(series: list[dict]) -> str:
    relief = float(np.median([s["relief"] for s in series]))
    if relief < RELIEF_NUL:
        return (f"❌ Intérieur uniforme (relief médian {relief:.0f} niveaux).\n"
                f"   Il n'y a rien à récupérer dans la silhouette : aucune "
                f"retouche ne fera apparaître\n"
                f"   les pattes arrière. Seul l'éclairage peut changer ça "
                f"(lumière réfléchie côté\n"
                f"   caméra, ou plancher éclairé par la tranche).")
    if relief < RELIEF_FRANC:
        return (f"⚠  Structures faibles (relief médian {relief:.0f} "
                f"niveaux).\n"
                f"   Regarde la colonne « zoom étiré » des mosaïques : si les "
                f"pattes arrière s'y\n"
                f"   détachent du corps, un prétraitement CLAHE vaut un "
                f"essai ; sinon, non.")
    return (f"✅ Des structures claires existent dans la silhouette (relief "
            f"médian {relief:.0f} niveaux).\n"
            f"   Un prétraitement CLAHE mérite d'être testé — à appliquer "
            f"IDENTIQUEMENT\n"
            f"   à l'entraînement et à chaque inférence future.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", type=Path, required=True,
                        help="Vidéo à examiner.")
    parser.add_argument("--frames", type=int, default=5,
                        help="Nombre de frames échantillonnées (défaut 5).")
    parser.add_argument("--clip", type=float, default=3.0,
                        help="clipLimit CLAHE (défaut 3.0).")
    parser.add_argument("--tuile", type=int, default=8,
                        help="Taille de grille CLAHE (défaut 8).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Dossier des mosaïques (défaut : à côté de la "
                             "vidéo, dans preview_contraste/).")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("❌ OpenCV requis : conda activate dlc", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir {args.video}", file=sys.stderr)
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # On évite les 5 % de début et de fin (arène vide, mains de l'expérimentateur)
    indices = np.linspace(total * 0.05, total * 0.95, args.frames).astype(int)

    out = args.out or (args.video.parent / "preview_contraste")
    out.mkdir(parents=True, exist_ok=True)

    series = []
    print(f"Vidéo : {args.video.name}  ({total} frames)\n")
    print(f"{'frame':>8}  {'pixels':>7}  {'p50':>4}  {'p99':>4}  "
          f"{'relief':>6}  {'% plat':>7}")
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        masque = silhouette(gris)
        if masque is None:
            print(f"{i:>8}  (souris introuvable)")
            continue
        st = stats_interieur(gris, masque)
        series.append(st)
        print(f"{i:>8}  {st['n_px']:>7}  {st['p50']:>4.0f}  {st['p99']:>4.0f}  "
              f"{st['relief']:>6.0f}  {st['pct_plat']:>6.1f}%")
        cv2.imwrite(str(out / f"frame_{i:06d}.png"),
                    mosaique(gris, masque, st, args.clip, args.tuile))
    cap.release()

    if not series:
        print("\n❌ Aucune silhouette détectée.", file=sys.stderr)
        sys.exit(1)
    print(f"\nMosaïques : {out}\n")
    print(verdict(series))


if __name__ == "__main__":
    main()
