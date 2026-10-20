"""Briques de graphiques Altair partagées par les pages (VAME, Qualité DLC).

Règles de la palette et des marques : skill dataviz — teintes de
`lib.icons.SERIES` en ordre fixe, traits de 2 px, grille discrète, survol
avec réticule, libellés directs en encre neutre.
"""
from __future__ import annotations

import math
import re

import altair as alt
import pandas as pd
import streamlit as st

from lib.icons import FOND, SERIES

NEUTRE = "#9ca3af"


def _champs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Copie de `df` aux noms de colonnes sûrs pour Vega-Lite, où un point
    est un accès imbriqué (`metrics/test.rmse` serait lu comme
    `metrics/test` → `rmse`, donc une courbe vide)."""
    noms = {c: re.sub(r"\W", "_", str(c)) for c in df.columns}
    return df.rename(columns=noms), noms


def _libelles_lisibles(dernier: pd.DataFrame, log: bool, bas: float, haut: float) -> bool:
    """Faux si deux fins de courbe sont à moins de 6 % de la hauteur : les
    libellés directs se chevaucheraient, la légende suffit alors."""
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    etendue = (f(haut) - f(bas)) or 1.0
    pos = sorted(f(v) for v in dernier["valeur"])
    return all((b - a) / etendue > 0.06 for a, b in zip(pos, pos[1:]))


def _axe_x(d: pd.DataFrame, x: str, titre: str, ticks: int, entier: bool) -> alt.X:
    """Axe x. `entier` (epochs…) : démarre à la première valeur et non à 0,
    graduations entières seulement (pas de 1.1, 1.2 quand il y a peu de
    points)."""
    if not entier:
        return alt.X(f"{x}:Q", title=titre, axis=alt.Axis(grid=False, tickCount=ticks))
    bas, haut = int(d[x].min()), int(d[x].max())
    return alt.X(f"{x}:Q", title=titre,
                 scale=alt.Scale(domain=[bas, max(haut, bas + 1)], nice=False),
                 axis=alt.Axis(grid=False, tickCount=ticks, tickMinStep=1, format="d"))


def mode() -> str:
    return "dark" if st.context.theme.type == "dark" else "light"


def lignes(df: pd.DataFrame, x: str, colonnes: dict[str, str], *,
           titre_x: str, titre_y: str, log: bool = False,
           couches_fond: list | None = None, couches_dessus: list | None = None,
           hauteur: int = 300, x_entier: bool = False) -> alt.LayerChart:
    """Courbes de `colonnes` ({colonne: libellé}) en fonction de `x`.

    Survol : réticule vertical + points + infobulle de toutes les séries.
    Libellés directs au bout de chaque courbe. `log` n'est appliqué que si
    toutes les valeurs sont > 0 (sinon l'échelle log n'a pas de sens).
    """
    df, noms = _champs(df)
    x, colonnes = noms[x], {noms[c]: lib for c, lib in colonnes.items()}
    longue = (df.melt(x, value_vars=list(colonnes), var_name="serie", value_name="valeur")
              .dropna(subset=["valeur"])
              .assign(serie=lambda d: d["serie"].map(colonnes)))
    log = log and bool((longue["valeur"] > 0).all())
    echelle = alt.Scale(type="log") if log else alt.Scale(zero=False)
    survol = alt.selection_point(fields=[x], nearest=True, on="pointerover",
                                 empty=False, clear="pointerout")
    base = alt.Chart(longue).encode(
        x=_axe_x(longue, x, titre_x, 10, x_entier),
        y=alt.Y("valeur:Q", title=f"{titre_y} (échelle log)" if log else titre_y,
                scale=echelle, axis=alt.Axis(tickCount=4, gridOpacity=0.5)),
        color=alt.Color("serie:N", title=None, scale=alt.Scale(
            domain=list(colonnes.values()), range=SERIES[mode()][:len(colonnes)])),
    )
    dernier = longue.sort_values(x).groupby("serie").tail(1)
    couches = list(couches_fond or []) + [
        base.mark_line(strokeWidth=2),
        base.mark_point(filled=True, size=60).encode(
            opacity=alt.condition(survol, alt.value(1), alt.value(0))),
        alt.Chart(df).mark_rule(color=NEUTRE).encode(x=f"{x}:Q").transform_filter(survol),
        alt.Chart(df).mark_rule(opacity=0, strokeWidth=12).encode(
            x=f"{x}:Q",
            tooltip=[alt.Tooltip(f"{x}:Q", title=titre_x, format="d" if x_entier else "")]
            + [alt.Tooltip(f"{c}:Q", title=lib, format=".4g") for c, lib in colonnes.items()],
        ).add_params(survol),
    ] + list(couches_dessus or [])
    if _libelles_lisibles(dernier, log, longue["valeur"].min(), longue["valeur"].max()):
        # Libellés directs en bout de courbe (encre neutre, la ligne porte la couleur).
        couches.append(alt.Chart(dernier).mark_text(align="left", dx=6, color=NEUTRE).encode(
            x=f"{x}:Q", y=alt.Y("valeur:Q", scale=echelle), text="serie:N"))
    return alt.layer(*couches).properties(height=hauteur)


def petit_multiple(df: pd.DataFrame, x: str, colonne: str, titre: str, *,
                   titre_x: str, hauteur: int = 200, x_entier: bool = False) -> alt.Chart:
    """Une seule série, sans légende (le titre la nomme)."""
    df, noms = _champs(df)
    x, colonne = noms[x], noms[colonne]
    return (alt.Chart(df).mark_line(strokeWidth=2, color=SERIES[mode()][0])
            .encode(x=_axe_x(df, x, titre_x, 6, x_entier),
                    y=alt.Y(f"{colonne}:Q", title=None, scale=alt.Scale(zero=False),
                            axis=alt.Axis(tickCount=3, gridOpacity=0.5)),
                    tooltip=[alt.Tooltip(f"{x}:Q", title=titre_x, format="d" if x_entier else ""),
                             alt.Tooltip(f"{colonne}:Q", title=titre, format=".4g")])
            .properties(title=titre, height=hauteur))


def repere_vertical(valeur: float, texte: str, x: str) -> list:
    """Ligne pointillée verticale + libellé en haut du graphique."""
    d = pd.DataFrame({x: [valeur], "texte": [texte]})
    return [
        alt.Chart(d).mark_rule(color=NEUTRE, strokeDash=[4, 4]).encode(x=f"{x}:Q"),
        alt.Chart(d).mark_text(align="left", baseline="top", dx=4, dy=4,
                               color=NEUTRE, fontSize=11)
        .encode(x=f"{x}:Q", y=alt.value(0), text="texte:N"),
    ]
