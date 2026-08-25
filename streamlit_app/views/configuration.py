"""Page Configuration — racines de préférence + sondes d'environnement.

Deux sections :

1. **Racines** — `projects_root` / `models_root`, éditables et persistées
   via `lib.config.save_prefs`. Un root configuré qui n'existe pas est
   signalé plutôt que silencieusement ignoré (sur une machine de dev hors
   Windows, c'est l'état normal : les défauts de `scripts/interactive.py`
   sont des chemins Windows).

2. **Environnements conda** — sonde d'import pour `ethoflow`/`dlc`/`vame`
   (voir `lib.envcheck`, spec §15) : `environment-vame.yml` ne déclare que
   `vame-py`, matplotlib/scipy/umap/scikit-learn arrivent en dépendances
   transitives. Une sonde de quelques secondes vaut mieux qu'un
   `analyze_vame.py` qui plante à l'import après un entraînement de
   plusieurs heures. Les sondes shellent (`conda run`) : c'est
   `lib.envcheck`, pas cette vue, qui le fait — jamais de `subprocess`
   direct depuis une page, et ce n'est pas un job de pipeline (synchrone,
   quelques secondes) donc pas `views._job` non plus.
"""
from __future__ import annotations

import lib.envcheck as EC
from lib.config import (
    DEFAULT_MODELS_ROOT,
    DEFAULT_PROJECTS_ROOT,
    current_project,
    load_prefs,
    models_root,
    projects_root,
    save_prefs,
)
from lib.icons import ACCENT, lucide_html, lucide_title

import streamlit as st

from views._widgets import champ_chemin

_CLE_RESULTATS = "_config_probe_results"


# ============================================================
# Section 1 : racines
# ============================================================

def _section_racines() -> None:
    st.markdown(lucide_title("database", "Racines"), unsafe_allow_html=True)
    st.caption(
        "Valeurs par défaut (`scripts/interactive.py`) : projets "
        f"`{DEFAULT_PROJECTS_ROOT}`, modèles `{DEFAULT_MODELS_ROOT}` — des "
        "chemins Windows, donc normalement absents sur une machine de "
        "développement macOS/Linux."
    )

    # Hors formulaire : `champ_chemin` porte un bouton « Parcourir… », et un
    # `st.form` n'accepte que son propre bouton de soumission.
    projets_val = champ_chemin(
        "Racine des projets EthoFlow",
        cle="config_projects_root",
        valeur_defaut=str(projects_root()),
        titre_dialogue="Racine des projets EthoFlow",
    )
    modeles_val = champ_chemin(
        "Racine des modèles DLC",
        cle="config_models_root",
        valeur_defaut=str(models_root()),
        titre_dialogue="Racine des modèles DLC",
    )
    enregistrer = st.button("Enregistrer", key="config_enregistrer")

    if enregistrer:
        prefs = load_prefs()
        prefs["projects_root"] = projets_val
        prefs["models_root"] = modeles_val
        save_prefs(prefs)
        st.toast("Racines enregistrées")
        st.rerun()

    racines = {
        "Projets EthoFlow": projects_root(),
        "Modèles DLC": models_root(),
    }
    for label, chemin in racines.items():
        if chemin.exists():
            icone = lucide_html("circle-check", 14, ACCENT)
            st.markdown(f"{icone} **{label}** — `{chemin}`", unsafe_allow_html=True)
        else:
            icone = lucide_html("circle-x", 14, "#ef4444")
            st.markdown(
                f"{icone} **{label}** — `{chemin}` — introuvable sur cette machine",
                unsafe_allow_html=True,
            )


# ============================================================
# Section 2 : sondes d'environnement
# ============================================================

def _affiche_resultat(env: str, resultat: "EC.ProbeResult") -> None:
    if resultat.ok:
        icone = lucide_html("circle-check", 16, ACCENT)
        st.markdown(f"{icone} **`{env}`** — import OK", unsafe_allow_html=True)
    else:
        icone = lucide_html("circle-x", 16, "#ef4444")
        st.markdown(f"{icone} **`{env}`** — échec de la sonde", unsafe_allow_html=True)
        with st.expander("Détails"):
            st.code(resultat.output or "(pas de sortie)")

    if env == "dlc" and resultat.ok:
        if resultat.cuda is True:
            st.success("CUDA disponible — l'inférence DLC utilisera le GPU.")
        elif resultat.cuda is False:
            st.warning(
                "CUDA indisponible sur cet environnement : l'inférence DLC "
                "tournera sur CPU — compte des heures plutôt que des minutes."
            )
        else:
            st.warning("Statut CUDA indéterminé (sortie inattendue de la sonde).")


def _section_environnements() -> None:
    st.markdown(
        lucide_title("clipboard-list", "Environnements conda"),
        unsafe_allow_html=True,
    )
    st.caption(
        "`environment-vame.yml` ne déclare que `vame-py` : matplotlib, "
        "scipy, umap et scikit-learn arrivent en dépendances transitives. "
        "Si l'une manque, `analyze_vame.py` ou `behavior_structure_gif.py` "
        "échouent à l'import — potentiellement après un entraînement VAME "
        "de plusieurs heures. Cette sonde le détecte en quelques secondes."
    )

    if st.button("Vérifier les environnements", key="config_btn_verif_env"):
        with st.spinner("Sondes en cours (jusqu'à 60 s par environnement)…"):
            st.session_state[_CLE_RESULTATS] = EC.probe_all()

    resultats = st.session_state.get(_CLE_RESULTATS)
    if resultats is None:
        st.info("Aucune vérification effectuée depuis le démarrage de l'app.")
        return

    for env, resultat in resultats.items():
        _affiche_resultat(env, resultat)


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    st.title("Configuration")
    st.caption("Chemins de préférence et santé des environnements conda")

    st.divider()
    _section_racines()

    st.divider()
    _section_environnements()

    projet = current_project()
    if projet:
        st.divider()
        st.markdown(f"**Projet courant :** `{projet}`")
