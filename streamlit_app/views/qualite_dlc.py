"""Page Qualité DLC — le modèle DeepLabCut du projet, en lecture seule.

L'entraînement lui-même reste au terminal (`scripts/dlc_model-training/`,
voir la page À propos) : cette page ne lance rien, elle montre ce que
trois scripts de diagnostic diraient, sans les relancer à la main.

- Entraînement : `learning_stats.csv` (comme `watch_training.py`), en
  direct tant que DLC le réécrit.
- Erreurs par keypoint : évaluation vs frames labellisées
  (`keypoint_errors.py`), avec les confusions gauche/droite.
- Confiance par session : le modèle généralise-t-il aux vidéos du projet ?
  (`inspect_predictions.py`, verdict du script tel quel).
"""
from __future__ import annotations

import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import lib.dlc_qualite as Q
from lib.config import require_project
from views import _graphiques as G
from views._widgets import entete_page


# ============================================================
# Onglet 1 : entraînement
# ============================================================

def _titre_court(colonne: str) -> str:
    return colonne.removeprefix("losses/").removeprefix("metrics/")


def _entrainement(model_dir: Path) -> None:
    sources = Q.stats_entrainement(model_dir)
    if not sources:
        st.info("Aucun `learning_stats.csv` dans ce modèle — l'entraînement "
                "(`02_train.py`) n'a pas encore écrit de premier epoch.")
        return
    source = sources[0]
    if len(sources) > 1:
        chemins = {str(s["chemin"].relative_to(model_dir)): s for s in sources}
        source = chemins[st.selectbox("Fichier", list(chemins), key="qualite_stats")]

    df, prevues = source["df"], source["epochs_prevues"]
    en_cours = time.time() - source["mtime"] < Q.WT.SILENCE_S
    principales = {c: titre for c, titre in (("losses/train.total_loss", "Train"),
                                             ("losses/eval.total_loss", "Eval")) if c in df}
    autres = [c for c in df.columns if c != "step" and c not in principales]

    cols = st.columns(1 + len(principales))
    dernier = int(df["step"].iloc[-1])
    cols[0].metric("Epoch", f"{dernier} / {prevues}" if prevues else dernier,
                   help="En cours — rafraîchi toutes les 60 s." if en_cours else None)
    for col, (c, titre) in zip(cols[1:], principales.items()):
        col.metric(f"Perte {titre.lower()}", f"{df[c].dropna().iloc[-1]:.4g}")

    if principales:
        st.altair_chart(G.lignes(df, "step", principales, titre_x="Epoch",
                                 titre_y="Perte totale", log=True), width="stretch")
    if autres:
        grille = st.columns(2)
        for i, c in enumerate(autres):
            grille[i % 2].altair_chart(
                G.petit_multiple(df, "step", c, _titre_court(c), titre_x="Epoch"),
                width="stretch")
    with st.expander("Comment lire ces courbes", expanded=False):
        st.markdown(
            "- **Train et Eval descendent ensemble** : le modèle apprend. "
            "Si **Eval remonte pendant que Train descend**, il apprend par "
            "cœur tes images au lieu de généraliser : ajoute des images "
            "variées au jeu d'entraînement (README, Parcours B.3).\n"
            "- **Cassures nettes vers les epochs 160 et 190** : le pas "
            "d'apprentissage baisse, la phase de raffinement démarre. "
            "C'est elle qui rend les points précis — d'où les 200 epochs, à "
            "ne pas raccourcir.\n"
            "- **Assez entraîné ?** Si `test.rmse_pcutoff` baisse encore sur "
            "les derniers epochs, le run était court ; si elle plafonne, "
            "c'est bon. Le meilleur état (meilleure mAP de test, vérifiée "
            "tous les 10 epochs) est gardé à part et sert à l'inférence."
        )
    st.caption(f"`{source['chemin']}`")


_entrainement_en_direct = st.fragment(run_every=60)(_entrainement)


# ============================================================
# Onglet 2 : erreurs par keypoint
# ============================================================

def _erreurs(model_dir: Path) -> None:
    pcutoff = st.slider("Seuil de confiance (pcutoff)", 0.05, 0.95,
                        float(Q.DEFAULT_PCUTOFF), 0.05, key="qualite_pcutoff",
                        help="Une prédiction au-dessus de ce seuil compte comme "
                             "« confiante ». Baisse-le (ex : 0,3) pour voir ce "
                             "que donnerait un seuil plus permissif.")
    try:
        tab, confusions, fichier = Q.erreurs_keypoints(model_dir, pcutoff)
    except (FileNotFoundError, ValueError) as e:
        st.info("Pas d'évaluation exploitable — lance `02_train.py --eval-only`.")
        st.code(str(e), language=None)
        return

    total = tab[tab["keypoint"] == "TOTAL"].iloc[0]
    kp = tab[tab["keypoint"] != "TOTAL"]
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (toutes)", f"{total['rmse']:.1f} px")
    c2.metric("RMSE (confiantes)", f"{total['rmse_seuil']:.1f} px")
    c3.metric("Au-dessus du seuil", f"{total['pct_au_dessus_seuil']:.0f} %")
    st.caption(
        "Repère pour la RMSE confiante (vidéo vue de dessous en 2048×2048, "
        "souris d'environ 300 px du nez à la base de la queue) : sous 8 px, "
        "c'est bon ; de 8 à 15 px, utilisable mais améliorable ; au-dessus "
        "de 15 px, il manque des images sur certaines situations ou les "
        "annotations sont incohérentes. À une autre résolution, raisonne "
        "en part du corps (8 px ≈ 2,7 %)."
    )

    st.caption(
        "Par keypoint : RMSE sur toutes les prédictions et sur les seules "
        "confiantes. Un grand écart = le modèle **sait** qu'il ne sait pas, "
        "le cutoff écartera ces points. Une RMSE *confiante* élevée est le "
        "vrai problème : il se trompe en étant sûr de lui."
    )
    noms = {"rmse": "Toutes", "rmse_seuil": "Confiantes"}
    longue = (kp.melt(id_vars=["keypoint", "n", "pct_au_dessus_seuil", "mediane"],
                      value_vars=list(noms), var_name="mesure", value_name="px")
              .assign(mesure=lambda d: d["mesure"].map(noms)))
    ordre = kp.sort_values("rmse_seuil", ascending=False)["keypoint"].tolist()
    y = alt.Y("keypoint:N", sort=ordre, title=None, axis=alt.Axis(labelOverlap=False))
    infobulle = [alt.Tooltip("keypoint:N", title="Keypoint"),
                 alt.Tooltip("mesure:N", title="Prédictions"),
                 alt.Tooltip("px:Q", title="RMSE (px)", format=".1f"),
                 alt.Tooltip("mediane:Q", title="Médiane (px)", format=".1f"),
                 alt.Tooltip("pct_au_dessus_seuil:Q", title="% > seuil", format=".0f"),
                 alt.Tooltip("n:Q", title="Instances")]
    barre = alt.Chart(longue).mark_line(color=G.NEUTRE, strokeWidth=2).encode(
        x="px:Q", y=y, detail="keypoint:N")
    # Anneau couleur du fond : deux mesures égales restent toutes deux visibles.
    points = alt.Chart(longue).mark_point(filled=True, size=90, opacity=1,
                                          stroke=G.FOND[G.mode()], strokeWidth=2).encode(
        x=alt.X("px:Q", title="RMSE (px)", axis=alt.Axis(gridOpacity=0.5, tickCount=8)),
        y=y,
        color=alt.Color("mesure:N", title=None, scale=alt.Scale(
            domain=list(noms.values()), range=G.SERIES[G.mode()][:2])),
        tooltip=infobulle,
    )
    st.altair_chart((barre + points).properties(height=26 * len(kp) + 40),
                    width="stretch")
    with st.expander("Tableau complet", expanded=False):
        st.dataframe(tab.round(2), hide_index=True, width="stretch")
        st.caption(f"Prédictions : `{fichier}`")

    paires = confusions.get("paires") or []
    retour = confusions.get("retournement")
    if paires or retour:
        st.markdown("**Confusions**")
        st.caption(
            "Un point posé sur son jumeau (patte gauche sur la droite, par "
            "exemple) trahit souvent des inversions gauche/droite dans les "
            "annotations : refais l'audit gauche/droite (README, B.3.4) et "
            "regarde les frames listées."
        )
        if paires:
            st.dataframe(pd.DataFrame([{
                "Keypoint": p["keypoint"],
                "% posé sur le jumeau": round(p["pct_sur_jumeau"], 1),
                "Distance médiane au jumeau (px)": round(p["mediane_vers_jumeau"], 1),
                "Frames à regarder": ", ".join(p["frames"][:5]),
            } for p in paires]), hide_index=True, width="stretch")
        if retour:
            st.caption(f"Retournement tête/queue : {retour['n']} frame(s) sur "
                       f"{retour['total']}" + (f" — {', '.join(retour['frames'][:5])}"
                                               if retour["frames"] else ""))


# ============================================================
# Onglet 3 : confiance sur les sessions du projet
# ============================================================

def _confiances(projet: Path) -> None:
    fichiers = Q.h5_sessions(projet)
    if not fichiers:
        st.info("Aucune inférence dans `data/dlc-output/` — page **Pose (DLC)**.")
        return
    session = st.selectbox("Session", list(fichiers), key="qualite_session")
    tab, sup, verdict = Q.confiances(fichiers[session])

    st.caption(
        "Le modèle généralise-t-il à cette vidéo ? La RMSE d'évaluation est "
        "mesurée sur les vidéos d'entraînement ; ici, c'est la confiance "
        "réelle sur une vidéo du projet."
    )
    with st.expander("Comment lire ce graphique", expanded=False):
        st.markdown(
            "- **Médiane au-dessus de 0,6 sur la plupart des points** : le "
            "modèle est à l'aise, la vidéo ressemble à celles de "
            "l'entraînement.\n"
            "- **Médiane haute sauf sur 2 ou 3 points** : normal, ces points "
            "sont cachés une partie du temps (pattes sous le corps).\n"
            "- **Médiane entre 0,05 et 0,3 sur tous les points** : le modèle "
            "ne reconnaît pas cette vidéo (autre éclairage, caméra, zoom). "
            "Baisser le seuil n'y changera rien : annote des images de "
            "cette vidéo (README, B.6) ou vérifie qu'elle vient du même "
            "dispositif.\n"
            "- Une confiance basse n'est pas toujours une erreur : le point "
            "peut être au bon endroit mais flou. Vérifie sur la vidéo "
            "annotée à 0,3."
        )
    # `>0.3` etc. : le point serait lu comme un accès imbriqué par Vega-Lite.
    seuils = {c: f"pct_{i}" for i, c in enumerate(tab.columns) if c.startswith(">")}
    donnees = tab.rename(columns=seuils).reset_index(names="keypoint")
    st.altair_chart(
        alt.Chart(donnees).mark_bar(color=G.SERIES[G.mode()][0], cornerRadiusEnd=4,
                                    height=14).encode(
            x=alt.X("mediane:Q", title="Confiance médiane",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(gridOpacity=0.5, tickCount=10, format=".1f")),
            y=alt.Y("keypoint:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
            tooltip=[alt.Tooltip("keypoint:N", title="Keypoint"),
                     alt.Tooltip("mediane:Q", title="Médiane", format=".2f")]
            + [alt.Tooltip(f"{champ}:Q", title=f"% frames {c}", format=".0f")
               for c, champ in seuils.items()],
        ).properties(height=26 * len(tab) + 40),
        width="stretch",
    )
    st.code(verdict, language=None)
    with st.expander("Tableau et superposition gauche/droite", expanded=False):
        st.dataframe(tab.round(2), width="stretch")
        if not sup.empty:
            st.dataframe(sup.round(1), hide_index=True, width="stretch")
        st.caption(f"`{fichiers[session]}`")


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    projet = require_project()
    entete_page(f"{projet.name} · Outil", "Qualité du modèle DLC",
                "Le modèle DeepLabCut du projet, en lecture seule : l'entraînement se fait "
                "au terminal (scripts/dlc_model-training/), cette page montre ce que ses "
                "scripts de diagnostic en disent.")
    model_dir = Q.modele_dir(projet)
    if model_dir is None:
        st.info("Aucun modèle DLC configuré (ou introuvable) — tableau de bord → "
                "menu du projet (son nom) → Modèle DLC.")
        return
    st.caption(f"Modèle : `{model_dir}`")

    onglet_train, onglet_erreurs, onglet_conf = st.tabs(
        ["Entraînement", "Erreurs par keypoint", "Confiance par session"])
    with onglet_train:
        sources = Q.stats_entrainement(model_dir)
        en_cours = bool(sources) and time.time() - sources[0]["mtime"] < Q.WT.SILENCE_S
        (_entrainement_en_direct if en_cours else _entrainement)(model_dir)
    with onglet_erreurs:
        _erreurs(model_dir)
    with onglet_conf:
        _confiances(projet)
