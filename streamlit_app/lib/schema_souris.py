"""Schéma de souris : chaque point du corps coloré selon sa fiabilité (%).

SVG inline pour la page Nettoyage (bilan qualité). Les positions couvrent
les deux jeux de keypoints du projet (bottom-view et top-view, voir
`scripts/dlc_model-training/_config.py`) ; un nom inconnu est listé sous
le schéma plutôt qu'ignoré. Pas de Streamlit ici.
"""
from __future__ import annotations

from html import escape

# Souris vue de dessus, nez en haut, dans un viewBox de -60 à 260 en x
# (place pour les libellés de part et d'autre) et 320 en y.
# Gauche de l'animal à gauche du dessin (convention top-view).
POSITIONS: dict[str, tuple[int, int]] = {
    "nose": (100, 28), "chin": (100, 50), "left_ear": (74, 48), "right_ear": (126, 48),
    "head_center": (100, 72), "chest_center": (100, 112), "center": (100, 150),
    "belly_center": (100, 168), "left_flank": (58, 150),
    "front_paw_left": (60, 96), "front_paw_right": (140, 96),
    "hind_paw_left": (60, 200), "hind_paw_right": (140, 200),
    "tail_base": (100, 218), "tail_mid": (100, 258), "tail_tip": (100, 298),
}
# Points de l'axe central dont le libellé part à gauche, pour que deux
# libellés voisins ne se chevauchent jamais (les autres partent à droite).
_A_GAUCHE = {"chest_center", "belly_center", "tail_mid"}
# Libellés courts, lisibles sur le dessin.
LIBELLES = {
    "nose": "nez", "chin": "menton", "left_ear": "oreille G", "right_ear": "oreille D",
    "head_center": "tête", "chest_center": "poitrail", "center": "centre",
    "belly_center": "ventre", "left_flank": "flanc G",
    "front_paw_left": "patte av. G", "front_paw_right": "patte av. D",
    "hind_paw_left": "patte ar. G", "hind_paw_right": "patte ar. D",
    "tail_base": "base queue", "tail_mid": "mi-queue", "tail_tip": "bout queue",
}

# Dégradé rouge → orange → vert ; version légère pour les fonds de tableau.
_VIF = ((0xe3, 0x49, 0x48), (0xed, 0xa1, 0x00), (0x2f, 0x95, 0x67))
_LEGER = ((0xfb, 0xdc, 0xd9), (0xfd, 0xee, 0xc9), (0xd6, 0xef, 0xdf))


def couleur(pct: float | None, *, leger: bool = False) -> str:
    """Couleur hex pour un pourcentage 0-100 (rouge → vert). None : gris."""
    if pct is None or pct != pct:  # NaN
        return "#e8e6e1" if leger else "#b9b5ac"
    t = min(max(float(pct), 0.0), 100.0) / 100.0
    a, b, c = _LEGER if leger else _VIF
    lo, hi, u = (a, b, t * 2) if t < 0.5 else (b, c, (t - 0.5) * 2)
    return "#" + "".join(f"{round(lo[i] + (hi[i] - lo[i]) * u):02x}" for i in range(3))


def style_cellule(pct) -> str:
    """Pour `DataFrame.style.map` : fond léger du rouge au vert."""
    return f"background-color: {couleur(pct, leger=True)}"


def svg(pct_par_kp: dict[str, float], titre: str, *, largeur: int = 230) -> str:
    """Le schéma, avec le % écrit à côté de chaque point. `titre` au-dessus."""
    hauteur = round(largeur * 320 / 320)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largeur}" height="{hauteur}" '
        'viewBox="-60 0 320 320" font-family="IBM Plex Sans, sans-serif">',
        # Silhouette : tête, corps, queue.
        '<g fill="#f1efe9" stroke="#d8d5cc" stroke-width="1.5">'
        '<ellipse cx="100" cy="58" rx="30" ry="34"/>'
        '<ellipse cx="100" cy="150" rx="46" ry="78"/>'
        '<circle cx="72" cy="40" r="11"/><circle cx="128" cy="40" r="11"/></g>'
        '<path d="M100 218 C 96 245, 104 272, 100 300" fill="none" stroke="#d8d5cc" '
        'stroke-width="5" stroke-linecap="round"/>',
    ]
    inconnus = []
    for kp, pct in pct_par_kp.items():
        if kp not in POSITIONS:
            inconnus.append(f"{kp} {pct:.0f} %")
            continue
        x, y = POSITIONS[kp]
        a_gauche = x < 100 or kp in _A_GAUCHE
        tx, anchor = (x - 12, "end") if a_gauche else (x + 12, "start")
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="7" fill="{couleur(pct)}" stroke="#fff" stroke-width="1.5">'
            f"<title>{escape(kp)} : {pct:.1f} %</title></circle>"
            f'<text x="{tx}" y="{y + 4}" text-anchor="{anchor}" font-size="10" fill="#34322e">'
            f"{escape(LIBELLES.get(kp, kp))} <tspan font-weight=\"600\">{pct:.0f} %</tspan></text>"
        )
    parts.append("</svg>")
    legende = (f'<p style="margin:4px 0 0;font-size:11px;color:#6b6862">Hors schéma : '
               f"{escape(', '.join(inconnus))}</p>" if inconnus else "")
    return (f'<div style="text-align:center"><p style="margin:0 0 2px;font:600 13px '
            f'IBM Plex Sans,sans-serif;color:#34322e">{escape(titre)}</p>'
            f"{''.join(parts)}{legende}</div>")
