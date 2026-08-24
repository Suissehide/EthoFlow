"""Un nom d'icône Lucide absent de ICONS ne doit jamais faire planter la
sidebar (ruling R12.0) — voir `lib/icons._make_svg`."""
from __future__ import annotations

from lib import icons


def test_icone_inconnue_retombe_sur_le_fallback_sans_lever():
    html = icons.lucide_html("ce-nom-n-existe-pas")
    assert "<svg" in html
    assert "</svg>" in html


def test_icone_inconnue_dans_lucide_data_uri_ne_leve_pas():
    uri = icons.lucide_data_uri("ce-nom-n-existe-pas-non-plus")
    assert uri.startswith("data:image/svg+xml;base64,")


def test_icones_du_schema_de_navigation_cible_sont_enregistrees():
    """Les noms listés par la Task 13 (spec §8), sauf ceux notés absents
    dans `lib/icons.py` faute de tracé fiable (ex. brush-cleaning)."""
    attendues = {
        "folder-open", "database", "video", "scan-line", "waypoints",
        "tags", "chart-column", "clapperboard", "settings", "info",
    }
    manquantes = attendues - set(icons.ICONS)
    assert not manquantes, manquantes
