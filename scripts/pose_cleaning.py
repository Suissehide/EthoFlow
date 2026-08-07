"""Nettoyage des poses DLC avant VAME — au-delà du simple cutoff.

Recommandations Tony (VAME/LIN), mail de juillet 2026 :

    « En post-processing, le cutoff n'est qu'un proxy qui marche en
    moyenne sur un nombre suffisant de frames, et devrait toujours être
    combiné à d'autres méthodes. J'utiliserais un cutoff autour de 70 %
    et j'appliquerais des méthodes qui détectent les frames individuelles
    au tracking cassé et essaient de les corriger. »

Trois méthodes complémentaires implémentées ici :

1. **Cutoff de likelihood** (`mask_low_likelihood`)
   Le filet grossier. Défaut 0.70 au lieu des 0.30 historiques.

2. **Détection de vitesse aberrante** (`detect_velocity_outliers`)
   La méthode que Tony privilégie. On convertit les pixels en mètres
   (via une calibration règle → `px_per_cm`), on calcule la vitesse
   inter-frame de chaque keypoint, et on marque comme cassées les
   frames où un point dépasse une vitesse physiquement impossible
   (défaut 5 m/s). Indépendant de la likelihood : attrape aussi les
   labels *confiants mais faux*.

3. **Détection de points collants** (`detect_sticky_points`)
   « Parfois les labels bruités sautent toujours au même point que
   l'animal ne peut pas atteindre. Tu peux interpoler directement
   toutes les frames qui sautent à ce point. » On repère les
   coordonnées où un keypoint atterrit anormalement souvent (reflet
   IR fixe, coin d'arène, artefact de capteur) et on les marque.

Les frames marquées par 1/2/3 sont ensuite interpolées depuis les
valeurs valides voisines (`interpolate_flagged`) — pas jetées. Objectif
de Tony : « tracer la trajectoire de l'animal sur toute la vidéo ne doit
montrer aucun saut anormal de position, sans avoir à jeter complètement
des points, ce qui pose beaucoup de problèmes en aval. »

`plot_trajectory_qc` produit exactement ce graphe de contrôle.

Note sur le filtre de Kalman : Tony le mentionne comme alternative
(prédire la position depuis les frames précédentes, utiliser la
prédiction quand la mesure s'en écarte trop). Il précise que « ça n'a
un vrai bénéfice qu'après pas mal de bricolage sur ton dataset
spécifique » — donc non implémenté ici pour l'instant. La détection de
vitesse couvre le même besoin avec moins de paramètres à régler.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


# Vitesse max plausible pour un keypoint de souris, en m/s.
# Tony : « disons que 4 ou 5 m/s est la vitesse maximale que tu
# autoriserais ». Le corps d'une souris plafonne vers 1 m/s ; les pattes
# en phase de swing vont plus vite. 5 m/s laisse de la marge tout en
# attrapant les téléportations franches.
DEFAULT_MAX_SPEED_MS = 5.0

# Likelihood par défaut. Tony recommande ~0.70, à combiner avec les
# détections ci-dessous (et NON comme unique filtre).
DEFAULT_LIKELIHOOD = 0.70


def group_columns_by_bodypart(df: pd.DataFrame) -> dict[str, dict[str, tuple]]:
    """{bodypart: {'x': col, 'y': col, 'likelihood': col}} depuis un h5 DLC."""
    by_bp: dict[str, dict[str, tuple]] = defaultdict(dict)
    for col in df.columns:
        if not isinstance(col, tuple) or len(col) < 2:
            continue
        bp, coord = col[-2], col[-1]
        by_bp[bp][coord] = col
    return by_bp


def mask_low_likelihood(x: np.ndarray, y: np.ndarray, likelihood: np.ndarray,
                         threshold: float) -> np.ndarray:
    """Renvoie un masque booléen des frames à likelihood < threshold."""
    return likelihood < threshold


def detect_velocity_outliers(x: np.ndarray, y: np.ndarray, fps: float,
                              px_per_cm: float,
                              max_speed_ms: float = DEFAULT_MAX_SPEED_MS
                              ) -> np.ndarray:
    """Marque les frames où le keypoint se déplace trop vite pour être vrai.

    Convertit le déplacement inter-frame en m/s :

        vitesse = (distance_px / px_per_cm / 100) * fps

    Args:
        x, y: coordonnées du keypoint (NaN autorisés)
        fps: framerate de la vidéo
        px_per_cm: échelle de la caméra (voir calibrate_scale.py)
        max_speed_ms: vitesse au-delà de laquelle on considère le
            tracking cassé

    Returns:
        Masque booléen : True = frame suspecte à corriger.

    Note : quand une frame saute, c'est la frame d'ARRIVÉE qui est
    marquée (le saut se mesure entre i-1 et i). Si le label revient
    ensuite à sa vraie position, on marque aussi la frame suivante —
    l'aller-retour est symétrique et les deux sont fausses.
    """
    n = len(x)
    flagged = np.zeros(n, dtype=bool)
    if n < 2 or px_per_cm <= 0:
        return flagged

    dx = np.diff(x)
    dy = np.diff(y)
    dist_px = np.sqrt(dx ** 2 + dy ** 2)
    # px → cm → m, puis × fps pour des m/s
    speed_ms = (dist_px / px_per_cm / 100.0) * fps

    # NaN → pas de jugement (la frame est déjà traitée par le cutoff)
    too_fast = np.nan_to_num(speed_ms, nan=0.0) > max_speed_ms
    # Le saut i→i+1 incrimine la frame d'arrivée
    flagged[1:] |= too_fast
    return flagged


def detect_sticky_points(x: np.ndarray, y: np.ndarray,
                          tol_px: float = 3.0,
                          min_occurrences: int = 20,
                          max_fraction: float = 0.25
                          ) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Détecte les coordonnées où le keypoint atterrit anormalement souvent.

    Tony : « parfois les labels bruités sautent toujours au même point
    que l'animal ne peut pas atteindre — tu peux interpoler directement
    toutes les frames qui sautent à ce point ».

    Typiquement : un reflet IR fixe, un coin d'arène, un artefact de
    capteur. Le keypoint y revient sans cesse alors que l'animal est
    ailleurs.

    Args:
        tol_px: rayon (px) pour considérer deux positions identiques
        min_occurrences: nb minimum de frames sur le même point pour
            déclencher la détection
        max_fraction: garde-fou — si un « point collant » représente
            plus que cette fraction de la session, c'est probablement
            un vrai comportement stationnaire (souris immobile dans un
            coin), pas un artefact. On l'ignore.

    Returns:
        (masque booléen des frames concernées, liste des points détectés)
    """
    n = len(x)
    flagged = np.zeros(n, dtype=bool)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < min_occurrences:
        return flagged, []

    # Discrétise sur une grille de tol_px pour regrouper les positions
    # quasi identiques sans clustering coûteux.
    gx = np.full(n, np.nan)
    gy = np.full(n, np.nan)
    gx[valid] = np.round(x[valid] / tol_px)
    gy[valid] = np.round(y[valid] / tol_px)

    keys = {}
    for i in np.where(valid)[0]:
        k = (gx[i], gy[i])
        keys.setdefault(k, []).append(i)

    sticky_points: list[tuple[float, float]] = []
    n_valid = int(valid.sum())
    for k, idx in keys.items():
        if len(idx) < min_occurrences:
            continue
        if len(idx) / n_valid > max_fraction:
            # Trop fréquent pour être un artefact — comportement réel
            continue
        # Un vrai point collant est visité de manière DISPERSÉE dans le
        # temps (le label y saute puis revient). Un comportement
        # stationnaire réel occupe des frames contiguës.
        idx_arr = np.array(idx)
        gaps = np.diff(idx_arr)
        if len(gaps) and np.median(gaps) <= 2:
            # Frames quasi contiguës → immobilité réelle, on laisse
            continue
        flagged[idx_arr] = True
        sticky_points.append((float(np.nanmedian(x[idx_arr])),
                               float(np.nanmedian(y[idx_arr]))))

    return flagged, sticky_points


def interpolate_flagged(x: np.ndarray, y: np.ndarray, flagged: np.ndarray,
                         interp_limit: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Met à NaN les frames marquées puis interpole les trous courts.

    Les trous plus longs que `interp_limit` restent NaN — inventer une
    trajectoire sur une seconde d'occlusion ferait plus de mal que de
    bien à VAME.

    Returns:
        (x_clean, y_clean, n_frames_reparees)
    """
    xs = pd.Series(x.astype(float)).copy()
    ys = pd.Series(y.astype(float)).copy()
    xs[flagged] = np.nan
    ys[flagged] = np.nan

    nan_before = int(xs.isna().sum())
    xs = xs.interpolate(method="linear", limit=interp_limit, limit_area="inside")
    ys = ys.interpolate(method="linear", limit=interp_limit, limit_area="inside")
    n_repaired = nan_before - int(xs.isna().sum())
    return xs.to_numpy(), ys.to_numpy(), n_repaired


def clean_dataframe(df: pd.DataFrame, fps: float, px_per_cm: float | None,
                     likelihood_threshold: float = DEFAULT_LIKELIHOOD,
                     max_speed_ms: float = DEFAULT_MAX_SPEED_MS,
                     interp_limit: int = 25,
                     detect_sticky: bool = True,
                     ) -> tuple[pd.DataFrame, dict]:
    """Pipeline complet de nettoyage sur un h5 DLC single-animal.

    Ordre : cutoff likelihood → vitesse aberrante → points collants →
    interpolation des frames marquées.

    Si `px_per_cm` est None, la détection de vitesse est désactivée (on
    ne peut pas convertir en m/s sans échelle) et un avertissement est
    remonté dans les stats.

    Returns:
        (df_nettoye, stats)
    """
    df = df.copy()
    n_frames = len(df)
    by_bp = group_columns_by_bodypart(df)
    valid_bps = [bp for bp, c in by_bp.items()
                 if all(k in c for k in ("x", "y", "likelihood"))]

    stats = {
        "n_frames": n_frames,
        "n_keypoints": len(valid_bps),
        "n_low_likelihood": 0,
        "n_velocity_outliers": 0,
        "n_sticky": 0,
        "n_repaired": 0,
        "n_remaining_nan": 0,
        "sticky_points": {},
        "velocity_enabled": px_per_cm is not None and px_per_cm > 0,
        "likelihood_threshold": likelihood_threshold,
        "max_speed_ms": max_speed_ms,
        "px_per_cm": px_per_cm,
    }

    for bp in valid_bps:
        c = by_bp[bp]
        x = df[c["x"]].to_numpy(dtype=float)
        y = df[c["y"]].to_numpy(dtype=float)
        lik = df[c["likelihood"]].to_numpy(dtype=float)

        # 1) Cutoff de likelihood
        low = mask_low_likelihood(x, y, lik, likelihood_threshold)
        stats["n_low_likelihood"] += int(low.sum())
        x[low] = np.nan
        y[low] = np.nan

        flagged = low.copy()

        # 2) Vitesse aberrante (indépendante de la likelihood)
        if stats["velocity_enabled"]:
            vel = detect_velocity_outliers(x, y, fps, px_per_cm, max_speed_ms)
            n_new = int((vel & ~flagged).sum())
            stats["n_velocity_outliers"] += n_new
            flagged |= vel

        # 3) Points collants
        if detect_sticky:
            sticky, points = detect_sticky_points(x, y)
            n_new = int((sticky & ~flagged).sum())
            stats["n_sticky"] += n_new
            flagged |= sticky
            if points:
                stats["sticky_points"][bp] = points

        # 4) Interpolation des frames marquées
        x_clean, y_clean, n_rep = interpolate_flagged(x, y, flagged,
                                                       interp_limit)
        stats["n_repaired"] += n_rep
        stats["n_remaining_nan"] += int(np.isnan(x_clean).sum())

        df[c["x"]] = x_clean
        df[c["y"]] = y_clean

    return df, stats


def plot_trajectory_qc(df_before: pd.DataFrame, df_after: pd.DataFrame,
                        bodypart: str, out_path, session_id: str = "",
                        stats: dict | None = None) -> bool:
    """Trace la trajectoire avant/après nettoyage — le contrôle de Tony.

    « Ce que tu veux voir au final, c'est que tracer la trajectoire de
    l'animal sur toute la vidéo ne montre aucun saut anormal de position
    dans l'arène, sans avoir à jeter complètement des points. »

    Returns True si la figure a été écrite.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    def _xy(d):
        by_bp = group_columns_by_bodypart(d)
        if bodypart not in by_bp:
            return None, None
        c = by_bp[bodypart]
        if "x" not in c or "y" not in c:
            return None, None
        return d[c["x"]].to_numpy(float), d[c["y"]].to_numpy(float)

    x0, y0 = _xy(df_before)
    x1, y1 = _xy(df_after)
    if x0 is None or x1 is None:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (x, y, title) in zip(axes, [
        (x0, y0, f"Avant nettoyage — {bodypart}"),
        (x1, y1, f"Après nettoyage — {bodypart}"),
    ]):
        ax.plot(x, y, "-", lw=0.4, color="#1f77b4", alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (px)")
        # `datalim` est interdit sur des axes partagés (matplotlib lève une
        # ValueError) : il voudrait étirer les limites de chaque panneau
        # indépendamment, ce que sharex/sharey empêchent par construction.
        # `box` garde les limites communes et ajuste la boîte à la place —
        # c'est ce qu'on veut de toute façon : les deux trajectoires doivent
        # se lire dans le même repère pour être comparables.
        ax.set_aspect("equal", adjustable="box")
    # Repère image (y vers le bas). Les axes étant partagés, une seule
    # inversion suffit — l'appliquer aux deux annulerait la première.
    axes[0].invert_yaxis()
    axes[0].set_ylabel("y (px)")

    suptitle = f"Contrôle trajectoire — {session_id}" if session_id else \
        "Contrôle trajectoire"
    if stats:
        suptitle += (
            f"\ncutoff={stats.get('likelihood_threshold')} · "
            f"vitesse>{stats.get('max_speed_ms')} m/s : "
            f"{stats.get('n_velocity_outliers')} · "
            f"collants : {stats.get('n_sticky')} · "
            f"réparées : {stats.get('n_repaired')}"
        )
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True
