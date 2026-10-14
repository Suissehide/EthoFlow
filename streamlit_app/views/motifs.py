"""Page Motifs — README étape 8 : visionner les clips VAME et nommer les motifs.

`lib.motif_labels` (Task 6) est l'unique lecteur/écrivain de
`<projet>/data/vame/motif_labels.csv`. Cette page ne fabrique jamais ce
fichier elle-même (c'est `run_vame motif-videos` / `motif-labels` qui le
génère, `usage_pct` et `video` déjà pré-remplis) et ne re-devine jamais
l'emplacement d'un clip par glob — c'était la fragilité de l'ancienne page
(`label_motifs.py`, trois patterns de glob successifs), remplacée ici par
`ML.video_path`, qui lit la colonne `video` déjà résolue par le script.

Deux notions à ne pas confondre (voir README étape 8) :
- `category` : liste fermée de 8 valeurs (`lib.config.categories()`),
  écrite dans le CSV et utilisée par `analyze_vame.py` pour grouper les
  motifs. Présentée en `selectbox`/`SelectboxColumn` fermé — une valeur
  hors liste casse le regroupement.
- `VOCABULAIRE_SUGGERE` : dictionnaire d'exemples pour aider à remplir le
  champ **libre** `label` de façon cohérente (« grooming_face » plutôt que
  « toilettage tête »). Présenté comme exemples repliés dans l'onglet
  « Par motif », jamais comme source de `category`.

Note sur `artifact` : `analyze_vame.is_artifact_motif` accepte la marque
dans `confidence` **ou** dans `category`. `confidence` est l'emplacement
canonique — `category` reste la liste fermée des 8 catégories ETHOGRAM,
qui n'inclut pas `artifact` — mais les deux excluent bien le motif des
stats, parce que le README a longtemps indiqué `category` et que des CSV
annotés ainsi existent. `confidence` est donc un troisième champ libre,
éditable dans l'onglet « Par motif ».
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import lib.motif_labels as ML
import lib.pipeline as PL
from lib.config import VOCABULAIRE_SUGGERE, categories, require_project
from views import _job
from views import _widgets


# ============================================================
# CSV absent : uniquement les deux commandes qui le génèrent
# ============================================================

def _section_generation(projet: Path) -> None:
    st.info(
        "Pas encore de clips ni de fiche des motifs pour ce projet : "
        "génère-les ci-dessous (après le découpage en motifs de l'étape VAME)."
    )
    st.caption(
        "Tu obtiens un court clip par motif et une fiche avec une ligne par "
        "motif, à remplir ici. Relancer plus tard n'efface jamais une fiche "
        "déjà remplie."
    )
    cmd = PL.vame_stage(projet, "motif-videos")
    cmd_fiche = PL.vame_stage(projet, "motif-labels")
    with st.container(horizontal=True, vertical_alignment="center"):
        _job.bouton_lancer(
            projet, "Générer les clips et la fiche", cmd, cle="btn_motifs_videos",
            help="Long : un clip vidéo par motif, et la fiche à remplir.",
        )
        _job.bouton_lancer(
            projet, "Refaire la fiche seule", cmd_fiche, cle="btn_motifs_labels",
            type="secondary", help="Rapide : seulement si les clips existent déjà.",
        )
    _widgets.details_techniques(cmd, notes=[
        "Fiche : `data/vame/motif_labels.csv` ; clips : "
        "`data/vame/results/community_videos/`. « Refaire la fiche seule » : "
        "`run_vame.py motif-labels`."])


# ============================================================
# Reprise de l'ancien format (analysis/motif_labels_<algo>.yaml)
# ============================================================

def _section_legacy(projet: Path) -> None:
    fichiers = ML.legacy_yaml_files(projet)
    if not fichiers:
        return
    with st.expander(
        f"Reprendre des noms saisis avec l'ancienne version ({len(fichiers)} fichier(s))",
        expanded=False,
    ):
        st.caption(
            "L'ancienne version de l'app rangeait les noms de motifs dans "
            "un fichier que les analyses ne lisaient pas. Tu peux les "
            "reprendre ici ; rien n'est écrasé sans ton clic."
        )
        for f in fichiers:
            if st.button(f"Importer « {f.name} »", key=f"btn_import_{f.name}"):
                n = ML.migrate_from_yaml(projet, f)
                st.success(f"{n} nom(s) repris depuis {f.name}.")
                st.rerun()


# ============================================================
# Tri par usage décroissant
# ============================================================

def _tri_par_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Le README conseille de commencer par le haut : un motif sous 1 %
    pèse à peine dans les stats — pas la peine de s'y attarder en premier."""
    df = df.copy()
    df["_usage"] = pd.to_numeric(df["usage_pct"], errors="coerce")
    df = df.sort_values("_usage", ascending=False, na_position="last")
    return df.drop(columns="_usage").reset_index(drop=True)


# ============================================================
# Onglet "Par motif"
# ============================================================

def _libelle_motif(row: pd.Series) -> str:
    usage = row.get("usage_pct") or "?"
    label = row.get("label") or "(sans label)"
    return f"motif {row['motif_id']} — {usage} % — {label}"


def _tab_individual(projet: Path, df: pd.DataFrame) -> None:
    cats = categories()
    n = len(df)
    if n == 0:
        st.info("Le CSV ne contient aucune ligne.")
        return

    if "motifs_idx" not in st.session_state:
        st.session_state["motifs_idx"] = 0
    valeur_initiale = min(st.session_state["motifs_idx"], n - 1)
    idx = st.select_slider(
        "Motif (trié par usage décroissant — commence par le haut)",
        options=list(range(n)), value=valeur_initiale,
        format_func=lambda i: _libelle_motif(df.loc[i]),
        key="motifs_slider",
    )
    st.session_state["motifs_idx"] = idx
    row = df.loc[idx]
    motif_id = int(row["motif_id"])

    col_video, col_form = st.columns([2, 1])
    with col_video:
        clip = ML.video_path(projet, row)
        if clip:
            st.video(str(clip))
        else:
            st.warning(
                "Pas de clip pour ce motif : le fichier vidéo est introuvable."
            )
        usage = row.get("usage_pct") or ""
        st.metric("Usage", f"{usage} %" if usage else "—",
                  help="Part des images attribuées à ce motif, toutes "
                       "sessions confondues.")

    with col_form:
        label_key = f"motifs_label_{motif_id}"
        cat_key = f"motifs_cat_{motif_id}"
        conf_key = f"motifs_conf_{motif_id}"
        seed_key = f"motifs_seed_{motif_id}"

        # `df` est rechargé depuis le disque à chaque rerun (voir
        # render()) : si l'onglet Tableau a sauvegardé une édition pour ce
        # motif entre-temps, `row` la reflète déjà. On ne réamorce les
        # widgets que quand la valeur sur disque a changé depuis le
        # dernier amorçage (premier passage sur ce motif, ou édition
        # faite depuis l'autre onglet) — sinon on écraserait la saisie en
        # cours de l'utilisateur ici même à chaque rerun.
        valeurs_disque = (
            row.get("label", "") or "",
            (row.get("category") or "").strip(),
            row.get("confidence", "") or "",
        )
        if st.session_state.get(seed_key) != valeurs_disque:
            st.session_state[label_key] = valeurs_disque[0]
            st.session_state[cat_key] = valeurs_disque[1] or None
            st.session_state[conf_key] = valeurs_disque[2]
            st.session_state[seed_key] = valeurs_disque

        st.text_input("Label (texte libre)", key=label_key)

        # La valeur déjà dans le CSV peut être hors des 8 catégories
        # fermées (typo, vocabulaire de labo, ou `artifact` comme le
        # suggère par erreur le README — voir le module docstring). Si on
        # ne présentait que `cats`, le selectbox afficherait cette valeur
        # comme "non catégorisé" et un simple clic sur Enregistrer
        # écrirait `category=""`, effaçant l'annotation sans avertir
        # personne. On l'ajoute donc aux options pour qu'elle reste
        # visible et sélectionnée, avec un avertissement explicite.
        valeur_actuelle = st.session_state.get(cat_key)
        hors_liste = bool(valeur_actuelle) and valeur_actuelle not in cats
        options_cat = [*cats, valeur_actuelle] if hors_liste else cats
        if hors_liste:
            st.warning(
                f"Catégorie « {valeur_actuelle} » hors de la liste fermée "
                "des 8 catégories : conservée telle quelle (rien n'est perdu "
                "si tu enregistres sans y toucher), mais les analyses ne "
                "pourront pas l'utiliser pour regrouper les motifs. Choisis "
                "une des 8 catégories pour corriger."
            )
        st.selectbox(
            "Catégorie", options=options_cat, key=cat_key,
            placeholder="— non catégorisé —",
            help="Liste fermée de 8 catégories, utilisée par les analyses "
                 "pour regrouper les motifs.",
        )

        st.text_input(
            "Confiance", key=conf_key,
            help="Champ libre, par exemple `high` quand le comportement est "
                 "net dans le clip. Motif impossible à interpréter (erreur de "
                 "suivi, animal hors champ) : écris `artifact` et les "
                 "analyses l'écarteront au lieu de le compter comme un "
                 "comportement.",
        )

        if st.button("Enregistrer", key=f"motifs_save_{motif_id}", type="primary"):
            ML.set_fields(
                projet, motif_id,
                label=st.session_state[label_key],
                category=st.session_state[cat_key] or "",
                confidence=st.session_state[conf_key],
            )
            st.toast(f"Motif {motif_id} enregistré")
            st.rerun()

        with st.expander(
            "Vocabulaire suggéré", expanded=False,
        ):
            st.caption(
                "Des noms cohérents d'un motif à l'autre. Cliquer un terme "
                "remplit le champ « Label » (pas la catégorie) ; clique "
                "ensuite Enregistrer."
            )
            for groupe, termes in VOCABULAIRE_SUGGERE.items():
                st.caption(f"**{groupe}**")
                cols = st.columns(3)
                for i, terme in enumerate(termes):
                    # Via on_click : modifier la clé d'un widget déjà
                    # instancié dans ce run lève StreamlitAPIException ;
                    # le callback, lui, tourne avant sa recréation.
                    cols[i % 3].button(
                        terme, key=f"motifs_voc_{motif_id}_{groupe}_{terme}",
                        width="stretch",
                        on_click=st.session_state.__setitem__,
                        args=(label_key, terme),
                    )


# ============================================================
# Onglet "Tableau"
# ============================================================

def _column_config(df: pd.DataFrame, cats: list[str]) -> dict:
    """`label` et `category` éditables, tout le reste — colonnes connues
    ET colonnes ajoutées à la main — en lecture seule."""
    config: dict = {
        col: st.column_config.TextColumn(col, disabled=True)
        for col in df.columns
    }
    config["label"] = st.column_config.TextColumn(
        "label", help="Texte libre — voir le vocabulaire suggéré dans "
                      "l'onglet « Par motif ».",
    )
    # Un `SelectboxColumn` rejette silencieusement toute valeur absente de
    # `options` (elle s'affiche comme vide, et la moindre interaction avec
    # le tableau la réécrirait à "" au prochain save — même bug que dans
    # l'onglet « Par motif »). On complète donc les 8 catégories fermées
    # avec les valeurs hors liste déjà présentes dans le fichier, pour ne
    # jamais en perdre une.
    valeurs_hors_liste = sorted({
        v for v in df["category"].dropna().unique() if v and v not in cats
    })
    config["category"] = st.column_config.SelectboxColumn(
        "category", options=[*cats, *valeurs_hors_liste],
        help="Liste fermée de 8 catégories, utilisée par les analyses. Les "
             "valeurs hors liste déjà saisies sont gardées pour ne rien perdre, "
             "mais les analyses ne les reconnaissent pas.",
    )
    return config


def _tab_table(projet: Path, df: pd.DataFrame) -> None:
    cats = categories()
    # SelectboxColumn n'accepte que les valeurs de `options` (ou None) :
    # une cellule "" (motif pas encore catégorisé) doit devenir None pour
    # l'affichage, sans quoi data_editor la refuse.
    df_affiche = df.copy()
    df_affiche["category"] = df_affiche["category"].replace("", None)

    st.caption(
        "Les modifications s'enregistrent toutes seules. Seuls « label » et "
        "« category » se modifient ici ; la confiance se saisit dans "
        "l'onglet « Par motif »."
    )
    edited = st.data_editor(
        df_affiche,
        column_config=_column_config(df_affiche, cats),
        hide_index=True,
        width="stretch",
        key="motifs_table_editor",
    )

    a_sauver = edited.copy()
    a_sauver["category"] = a_sauver["category"].fillna("")
    # Le tableau est affiché trié par usage décroissant (cohérence avec
    # l'onglet « Par motif »), mais le CSV doit rester dans l'ordre
    # `motif_id` écrit par `run_vame` — sauvegarder l'ordre d'affichage
    # réordonnerait pour de bon le fichier du chercheur à chaque édition.
    par_motif_id = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    a_sauver = a_sauver.sort_values("motif_id", key=par_motif_id).reset_index(drop=True)
    df_pour_comparaison = df.sort_values("motif_id", key=par_motif_id).reset_index(drop=True)
    if not a_sauver.equals(df_pour_comparaison):
        ML.save(projet, a_sauver)
        st.toast("Tableau enregistré")


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    projet = require_project()

    _widgets.entete_etape(projet)

    _job.panneau(projet)

    if not ML.exists(projet):
        _section_generation(projet)
        st.divider()
        _job.historique(projet)
        return

    df = ML.load(projet)
    if df is None or df.empty:
        st.warning("La fiche des motifs existe mais est vide ou illisible.")
        return
    df = _tri_par_usage(df)

    _section_legacy(projet)

    with st.expander("Comment bien nommer les motifs", expanded=False):
        st.markdown(
            "- **Regarde le clip, puis nomme ce que fait l'animal.** Le label "
            "est libre mais précis (`grooming_face`, `walking_slow`) ; garde "
            "le même vocabulaire d'un motif à l'autre.\n"
            "- **La catégorie compte plus que le label** : les analyses "
            "regroupent les motifs par catégorie. Deux motifs qui montrent le "
            "même comportement prennent la même catégorie.\n"
            "- **Commence par les motifs les plus fréquents.** Un motif sous "
            "1 % d'usage pèse à peine dans les analyses.\n"
            "- **Bruit de suivi, animal hors champ** : écris `artifact` dans "
            "« Confiance », le motif sera exclu des statistiques.\n"
            "- Un motif sans nom apparaît comme `motif_0`, `motif_1`… dans "
            "les figures."
        )

    onglet_motif, onglet_table = st.tabs(["Par motif", "Tableau"])
    with onglet_motif:
        _tab_individual(projet, df)
    with onglet_table:
        _tab_table(projet, df)

    st.divider()
    _job.historique(projet)
