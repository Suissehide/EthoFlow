"""Page Analyses — README étape 9 : de la segmentation aux statistiques.

C'est ici qu'une interface change le plus la vie du chercheur : à la
console, `analyze_vame.py --list-columns` sort une liste texte qu'il faut
relire, puis retaper dans `--group-by` / `--cross`. Cette page transforme
ce texte en cases à cocher.

`_axes_disponibles` relit le résultat depuis le **log du dernier job
`--list-columns`** (via `lib.runner`), jamais depuis `session_state` : le
job et son log survivent à un rafraîchissement du navigateur, la mémoire
de page non. C'est le même principe que `views._job.panneau` (Task 9-16) :
`lib/runner.py` sous `<projet>/.ethoflow/jobs/` est la seule source de
vérité.

Deux notions à ne pas confondre (voir README étape 9) :
- un axe **simple** (`captopril`, `condition`, …), découvert par
  `--list-columns` ;
- un axe **composite** (`condition_x_captopril`), construit à la volée par
  `--cross <a> <b>` — jamais dans la liste découverte, mais utilisable
  comme `--extended-by` dès qu'il a été ajouté dans le formulaire.

Le README insiste : en dessous de 3 sessions par groupe, le motif est
ignoré (pas assez pour un test) ; un axe à 6 groupes sur 16 sessions donne
des p-values inexploitables. Cette page affiche l'effectif de chaque
groupe avant le lancement, pas seulement dans la sortie du script.

Onglet Résultats : les noms de fichiers réels d'`analyze_vame.py` (voir
`lib.analysis.group_analysis_files`) ne sont plus ceux de l'ancien
`views/results.py` (`heatmap_usage.png`, `mean_by_condition*.png`), qui
cherchait des fichiers qu'aucune version actuelle du script ne produit.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

import lib.motif_labels as ML
import lib.pipeline as PL
import lib.vame as VA
from lib.analysis import (csv_separator, group_analysis_files,
                          parse_list_columns, usage_par_groupe)
from lib.config import require_project
from lib.project import dlc_output_dir, project_kind
from lib.icons import FOND, SERIES
from lib import runner
from views import _job
from views import _widgets

_CLE_CROSS = "analyses_cross_pairs"


# ============================================================
# Axes découverts — relus depuis le log, jamais depuis session_state
# ============================================================

def _axes_disponibles(projet: Path) -> list[dict]:
    """Axes découverts, relus depuis le dernier job --list-columns."""
    for job in runner.history(projet, limit=30):
        if job.script == "analyze_vame.py" and "--list-columns" in job.argv:
            if job.state == "succeeded":
                return parse_list_columns(runner.read_log(projet, job.job_id))
            return []
    return []


def _effectifs(groupes: str) -> list[int]:
    """Effectifs par groupe extraits de `« Captopril (8 sessions), … »`."""
    return [int(n) for n in re.findall(r"\((\d+) sessions?\)", groupes)]


# ============================================================
# Onglet "Lancer une analyse"
# ============================================================

def _section_decouverte(projet: Path, axes: list[dict]) -> None:
    cmd = PL.analyze_vame(projet, list_columns=True)
    _job.bouton_lancer(
        projet, "Découvrir les axes", cmd, cle="btn_analyses_list_columns",
        type="primary" if not axes else "secondary",
        help="Relit les colonnes de ton Excel et retient celles qui ont "
             "entre 2 et 12 valeurs : les seules utilisables pour comparer.",
    )
    # Pas d'invitation à cliquer un bouton verrouillé : sa raison est déjà dessous.
    if not axes and _widgets.verrou_etape() is None:
        st.info(
            "Aucun critère de comparaison pour l'instant : clique sur "
            "**Découvrir les axes** ci-dessus."
        )


def _case_group_by(axe: dict) -> bool:
    effectifs = _effectifs(axe["groupes"])
    libelle = f"**{axe['nom']}** — {axe['n_groupes']} groupes : {axe['groupes']}"
    valeur = st.checkbox(libelle, key=f"analyses_gb_{axe['nom']}")
    if effectifs and min(effectifs) < 3:
        st.warning(
            f"« {axe['nom']} » : au moins un groupe a moins de 3 sessions. "
            "En dessous, les tests statistiques ignorent ce groupe."
        )
    return valeur


def _section_group_by(axes: list[dict]) -> list[str]:
    st.markdown('<p class="ef-section">Comparer selon</p>', unsafe_allow_html=True)
    st.caption(
        "Le test s'adapte au nombre de groupes (Mann-Whitney à 2, "
        "Kruskal-Wallis à 3 ou plus), pas au nombre de sessions : 6 groupes "
        "sur 16 sessions ne donnent pas de résultats fiables. Vérifie les "
        "effectifs avant de conclure. Cocher deux critères donne deux séries "
        "de figures indépendantes, pas un croisement."
    )
    return [axe["nom"] for axe in axes if _case_group_by(axe)]


def _section_cross(noms_axes: list[str]) -> list[tuple[str, str]]:
    st.markdown('<p class="ef-section">Croiser deux critères</p>', unsafe_allow_html=True)
    if _CLE_CROSS not in st.session_state:
        st.session_state[_CLE_CROSS] = []
    paires: list[tuple[str, str]] = st.session_state[_CLE_CROSS]

    st.caption(
        "Croiser crée un critère combiné (ex : génotype × traitement donne "
        "4 groupes) pour voir chaque combinaison. Ce n'est pas un test "
        "d'interaction : à 3 groupes ou plus, le test dit seulement qu'au "
        "moins un groupe diffère. Pour une interaction, fais une ANOVA à deux "
        "facteurs dans R à partir de `analysis_global_long.csv`. Surveille "
        "les effectifs : 16 sessions croisées donnent 4 groupes de 4."
    )
    if len(noms_axes) < 2:
        st.caption("Il faut au moins deux critères pour les croiser.")
    else:
        col_a, col_b, col_btn = st.columns([2, 2, 1], vertical_alignment="bottom")
        with col_a:
            a = st.selectbox("Critère A", options=noms_axes, key="analyses_cross_a")
        with col_b:
            options_b = [n for n in noms_axes if n != a]
            b = st.selectbox("Critère B", options=options_b, key="analyses_cross_b")
        with col_btn:
            if st.button("Ajouter", key="analyses_cross_add", width="stretch"):
                if (a, b) not in paires and (b, a) not in paires:
                    paires.append((a, b))
                    st.rerun()

    for i, (a, b) in enumerate(list(paires)):
        col_txt, col_rm = st.columns([5, 1], vertical_alignment="center")
        col_txt.markdown(f"{a} × {b}")
        if col_rm.button("Retirer", key=f"analyses_cross_rm_{i}"):
            paires.pop(i)
            st.rerun()

    return list(paires)


def _section_extended(noms_axes: list[str],
                      cross_pairs: list[tuple[str, str]]) -> tuple[bool, str | None]:
    st.markdown('<p class="ef-section">Analyses étendues</p>', unsafe_allow_html=True)
    composites = [f"{a}_x_{b}" for a, b in cross_pairs]
    options = noms_axes + composites
    extended = st.checkbox(
        "Ajouter les épisodes, le temps près des bords et l'évolution par quart de session",
        key="analyses_extended",
        help="Compte 5 à 10 min de plus : relit le motif de chaque image. "
             "Ajoute la durée moyenne d'un épisode (« plus souvent » ou "
             "« plus longtemps » ?), l'évolution sur les 4 quarts de la "
             "session (effet réel ou habituation ?) et la part du temps au "
             "centre de l'arène (thigmotaxie, mesure d'anxiété). Tourne une "
             "seule fois, sur le critère choisi ci-dessous.",
    )
    extended_by = None
    if extended:
        if not options:
            st.warning(
                "Aucun critère disponible : découvre les axes ou ajoute un "
                "croisement d'abord."
            )
        else:
            extended_by = st.selectbox(
                "Critère des analyses étendues", options=options, key="analyses_extended_by",
                width=320,
            )
    return extended, extended_by


def _section_labels(projet: Path) -> str | None:
    st.markdown('<p class="ef-section">Noms des motifs</p>', unsafe_allow_html=True)
    defaut = str(ML.path(projet)) if ML.exists(projet) else ""
    valeur = st.text_input(
        "Fiche des motifs", value=defaut, key="analyses_labels",
        help="Sans elle, les figures affichent « motif_0, motif_1… » au "
             "lieu des noms de comportements.",
    )
    if not valeur:
        st.caption(
            "Pas encore de noms de motifs : remplis-les dans la page "
            "**Motifs** pour des figures lisibles."
        )
    return valeur or None


def _section_options_avancees(projet: Path) -> dict:
    with st.expander("Réglages avancés", expanded=False):
        algos_presents = sorted({VA.parse_algo_n(a)[0] for a in VA.list_algos(projet)})
        options_algo = algos_presents or ["hmm", "kmeans"]
        algo = st.selectbox("Méthode de découpage", options=options_algo,
                            key="analyses_algo", width=320)

        preciser_n = st.checkbox(
            "Préciser le nombre de motifs", key="analyses_preciser_n",
            help="Seulement si plusieurs découpages (nombres de motifs "
                 "différents) existent ; sinon, l'unique découpage est pris.",
        )
        n_clusters = None
        if preciser_n:
            n_actuel = VA.n_clusters(projet) or 15
            n_clusters = int(st.number_input(
                "Nombre de motifs", min_value=2, value=int(n_actuel), step=1, width=220,
                key="analyses_n_clusters",
            ))

        # Les arènes vides se repèrent sur les poses par arène
        # (data/dlc-output/<session>/<session>_A1.h5) : plusieurs animaux par
        # vidéo seulement.
        multi = project_kind(projet) == "multi"
        mask_empty, min_edge_frames = False, 25
        if multi:
            mask_empty = st.checkbox(
                "Exclure les images où l'arène est vide, en début et fin "
                "d'enregistrement", key="analyses_mask_empty",
                help="Avant que la souris soit posée dans l'arène et après "
                     "qu'elle en est retirée, il n'y a rien à analyser : ces "
                     "images sont retirées du calcul de l'usage.",
            )
        col1, col2 = st.columns(2)
        if multi:
            with col1:
                min_edge_frames = int(st.number_input(
                    "Arène vide pendant au moins (images)", min_value=1, value=25, step=1,
                    key="analyses_min_edge_frames",
                    help="Un trou plus court en début ou fin est un raté de "
                         "détection ponctuel, pas une arène vide.",
                ))
        with col2:
            fps = float(st.number_input(
                "Images par seconde", min_value=1.0, value=30.0, step=1.0,
                key="analyses_fps",
                help="Cadence réelle de tes vidéos. Sert à convertir les "
                     "durées d'épisodes en secondes.",
            ))
    return {
        "algo": algo, "n_clusters": n_clusters, "mask_empty": mask_empty,
        "min_edge_frames": min_edge_frames, "fps": fps,
    }


def _tab_lancer(projet: Path, axes: list[dict]) -> None:
    _section_decouverte(projet, axes)

    if not axes:
        return

    st.divider()
    group_by = _section_group_by(axes)
    st.divider()
    cross_pairs = _section_cross([axe["nom"] for axe in axes])
    st.divider()
    extended, extended_by = _section_extended(
        [axe["nom"] for axe in axes], cross_pairs,
    )
    st.divider()
    labels = _section_labels(projet)
    st.divider()
    options = _section_options_avancees(projet)

    st.divider()
    cmd = PL.analyze_vame(
        projet,
        algo=options["algo"],
        n_clusters=options["n_clusters"],
        labels=labels,
        group_by=group_by or None,
        cross=cross_pairs or None,
        extended=extended,
        extended_by=extended_by,
        mask_empty=options["mask_empty"],
        validity_source=dlc_output_dir(projet) if options["mask_empty"] else None,
        min_edge_frames=options["min_edge_frames"],
        fps=options["fps"],
    )
    _job.bouton_lancer(
        projet, "Lancer l'analyse", cmd, cle="btn_analyses_lancer",
        help="Les résultats s'affichent dans l'onglet **Résultats**.",
    )
    _widgets.details_techniques(cmd, notes=["Sorties : `data/vame/analysis/`."])


# ============================================================
# Onglet "Résultats"
# ============================================================

def _apercu_csv(path: Path) -> None:
    st.markdown(f"**{path.name}**")
    try:
        df = pd.read_csv(path, sep=csv_separator(path))
    except Exception as e:
        st.error(f"Lecture impossible : {e}")
        return
    st.caption(f"{len(df)} lignes × {len(df.columns)} colonnes")
    st.dataframe(df.head(20), width="stretch", hide_index=True)
    st.download_button(
        f"Télécharger {path.name}", data=path.read_bytes(),
        file_name=path.name, mime="text/csv", key=f"analyses_dl_{path.name}",
    )


def _affiche_fichiers(fichiers: list[Path]) -> None:
    images = [p for p in fichiers if p.suffix.lower() == ".png"]
    csvs = [p for p in fichiers if p.suffix.lower() == ".csv"]
    for p in images:
        st.image(str(p), caption=p.name, width="stretch")
    for p in csvs:
        _apercu_csv(p)


def _affiche_autres(fichiers: list[Path]) -> None:
    """Fichiers qu'aucune règle n'a su rattacher à un axe.

    Généraliste à dessein (contrairement à `_affiche_fichiers`) : un
    fichier qui atterrit ici peut être n'importe quel type — mieux vaut un
    simple lien de téléchargement qu'une disparition silencieuse (ruling
    R17.1).
    """
    for p in fichiers:
        col_nom, col_dl = st.columns([4, 1], vertical_alignment="center")
        col_nom.caption(p.name)
        try:
            contenu = p.read_bytes()
        except OSError:
            continue
        col_dl.download_button(
            "Télécharger", data=contenu, file_name=p.name,
            key=f"analyses_dl_autre_{p.name}",
        )


def _tab_resultats(projet: Path, axes: list[dict]) -> None:
    analysis_dir = VA.analysis_dir(projet)
    if not analysis_dir.is_dir():
        st.info(
            "Pas encore de résultats : lance une analyse depuis l'onglet "
            "**Lancer une analyse**."
        )
        return

    fichiers = sorted(p for p in analysis_dir.iterdir() if p.is_file())
    if not fichiers:
        st.info("Pas encore de résultats : le dossier d'analyse est vide.")
        return

    # Axes réellement découverts (--list-columns) : plus fiables que toute
    # règle inférée du nom de fichier pour lever les ambiguïtés (ruling
    # R17.1) — passés à group_analysis_files quand disponibles.
    axes_connus = [axe["nom"] for axe in axes] or None
    globaux, par_axe, autres = group_analysis_files(fichiers, axes_connus)

    _aide_lecture()

    if globaux:
        st.subheader("Général")
        _affiche_fichiers(globaux)

    for axe, fichiers_axe in par_axe.items():
        with st.expander(f"Axe : {axe} ({len(fichiers_axe)} fichier(s))",
                         expanded=len(par_axe) == 1):
            _affiche_fichiers(fichiers_axe)

    if autres:
        with st.expander(f"Autres ({len(autres)} fichier(s) non rattachés "
                         "à un axe)", expanded=False):
            st.caption(
                "Aucun axe connu ni motif de nom reconnu — présents ici "
                "pour ne pas disparaître, pas forcément à ignorer."
            )
            _affiche_autres(autres)


def _aide_lecture() -> None:
    with st.expander("Comment lire ces résultats", expanded=False):
        st.markdown(
            "- **Par où commencer** : `heatmap_usage_by_<critère>.png`. Une "
            "ligne par session, triée par groupe. Si une bande verticale "
            "change de teinte d'un bloc à l'autre, ce motif distingue tes "
            "groupes. Elle montre aussi les sessions aberrantes.\n"
            "- **Barres** (`mean_by_…`) : l'ampleur de l'effet, motif par "
            "motif. **Boîtes** (`boxplots_top_by_…`) : les 6 motifs aux "
            "moyennes les plus écartées — les plus *différents*, pas "
            "forcément les plus *significatifs*. Elles montrent si l'effet "
            "tient à deux ou trois sessions extrêmes.\n"
            "- **Significativité** (`stats_by_motif_<critère>.csv`) : "
            "Mann-Whitney à 2 groupes, Kruskal-Wallis à 3 ou plus. "
            "`p_value` est brute ; `p_adj_bh` est corrigée "
            "(Benjamini-Hochberg) sur tous les motifs testés. Lis "
            "`p_adj_bh` : un motif est retenu sous 0,05. Un motif dont un "
            "groupe a moins de 3 sessions n'est pas testé et n'apparaît pas.\n"
            "- **Catégories** (`…_by_category_…`) : les mêmes vues par "
            "catégorie de comportement, si tu as rempli les catégories dans "
            "**Motifs**. C'est ce niveau qui se raconte dans un papier.\n"
            "- **Unités** : en suffixe des colonnes. `_prop` = proportion "
            "des images (0–1), `_s` = secondes, `_frames` = nombre "
            "d'images, `_sessions` = effectif du test."
        )
    with st.expander("Emporter les données dans R ou Python", expanded=False):
        st.markdown(
            "`analysis_global_long.csv` empile toutes les mesures, une "
            "ligne par session × mesure, avec l'unité et toutes les "
            "colonnes de ton Excel. Séparateur `;`, décimale `.`. Filtre "
            "toujours sur `level` (`motif`, `category` ou `session`) avant "
            "une moyenne : sinon tu additionnes des motifs avec la "
            "catégorie qui les contient déjà."
        )
        st.code(
            'd <- read.csv("analysis_global_long.csv", sep = ";")\n'
            'usage <- subset(d, level == "motif" & metric == "usage")',
            language="r",
        )
        st.code(
            "import pandas as pd\n"
            'd = pd.read_csv("analysis_global_long.csv", sep=";")\n'
            "usage = d.query(\"level == 'motif' and metric == 'usage'\")",
            language="python",
        )


# ============================================================
# Usage par groupe — graphique interactif calculé dans l'app
# ============================================================



@st.cache_data(ttl=60, show_spinner="Lecture des motif_usage…")
def _usage_cache(projet: str, algo: str):
    return usage_par_groupe(Path(projet), algo)


def _tab_usage(projet: Path) -> None:
    import altair as alt

    st.caption(
        "Fréquence de chaque motif par session, colorée par groupe ; le "
        "trait épais est la moyenne du groupe. Descriptif uniquement — les "
        "tests statistiques sont dans « Lancer une analyse »."
    )
    algos = VA.list_algos(projet)
    if not algos:
        st.info("Aucune segmentation VAME pour l'instant — page **VAME**, étape segment.")
        return

    col_algo, col_axe = st.columns(2)
    algo = col_algo.selectbox("Segmentation", algos, key="analyses_usage_algo")
    df, colonnes = _usage_cache(str(projet), algo)
    colonnes = [c for c in colonnes if c[1] <= len(SERIES["light"])]
    if df.empty or not colonnes:
        st.info(
            "Aucune colonne de l'Excel ne sépare ces sessions en 2 à 8 "
            "groupes. Ajoute-en une dans **Données** (ex : `group`)."
        )
        return
    noms = [c[0] for c in colonnes]
    # `group` = la variable de comparaison principale de l'Excel (template).
    noms.sort(key=lambda n: n not in ("group", "condition"))
    axe = col_axe.selectbox(
        "Comparer par", noms, key="analyses_usage_axe",
        format_func=lambda n: f"{n} ({next(c[1] for c in colonnes if c[0] == n)} groupes)",
    )

    df = df.dropna(subset=[axe]).assign(groupe=lambda d: d[axe].astype(str),
                                        pct=lambda d: 100 * d["frequency"])
    groupes = sorted(df["groupe"].unique())
    ordre = (df.groupby("motif_nom")["pct"].mean()
             .sort_values(ascending=False).index.tolist())
    effectifs = df.groupby("groupe")["session"].nunique()
    mode = "dark" if st.context.theme.type == "dark" else "light"
    couleur = alt.Color("groupe:N", title=axe, scale=alt.Scale(
        domain=groupes, range=SERIES[mode][:len(groupes)]))

    base = alt.Chart(df).encode(
        x=alt.X("motif_nom:N", sort=ordre, title=None,
                axis=alt.Axis(labelAngle=-40, labelLimit=160)),
        xOffset=alt.XOffset("groupe:N", sort=groupes),
        color=couleur,
    )
    points = base.mark_circle(size=70, opacity=0.75, stroke=FOND[mode],
                              strokeWidth=1.5).encode(
        y=alt.Y("pct:Q", title="Usage (% des frames)"),
        tooltip=[alt.Tooltip("session:N", title="Session"),
                 alt.Tooltip("groupe:N", title=axe),
                 alt.Tooltip("motif_nom:N", title="Motif"),
                 alt.Tooltip("pct:Q", title="Usage (%)", format=".1f")],
    )
    moyennes = base.mark_tick(thickness=3, size=18).encode(
        y=alt.Y("mean(pct):Q"),
        tooltip=[alt.Tooltip("groupe:N", title=axe),
                 alt.Tooltip("motif_nom:N", title="Motif"),
                 alt.Tooltip("mean(pct):Q", title="Moyenne (%)", format=".1f"),
                 alt.Tooltip("count(session):Q", title="Sessions")],
    )
    st.altair_chart((points + moyennes).properties(height=380), width="stretch")

    st.caption("Sessions par groupe : " + " · ".join(
        f"{g} : {effectifs[g]}" for g in groupes))
    tableau = (df.pivot_table(index="motif_nom", columns="groupe", values="pct",
                              aggfunc="mean").reindex(ordre).round(2))
    tableau.columns = [f"{g} (n={effectifs[g]})" for g in tableau.columns]
    with st.expander("Tableau des moyennes (%)", expanded=False):
        st.dataframe(tableau, width="stretch")
        st.download_button(
            "Télécharger (CSV)", tableau.to_csv(sep=";").encode("utf-8-sig"),
            file_name=f"usage_{algo}_par_{axe}.csv", key="btn_usage_csv",
        )


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    projet = require_project()

    _widgets.entete_etape(projet)

    _job.panneau(projet)

    # Calculé une seule fois : sert à la fois aux cases --group-by et à
    # désambiguïser le regroupement des fichiers dans l'onglet Résultats
    # (ruling R17.1).
    axes = _axes_disponibles(projet)

    onglet_usage, onglet_lancer, onglet_resultats = st.tabs(
        ["Usage par groupe", "Lancer une analyse", "Résultats"])
    with onglet_usage:
        _tab_usage(projet)
    with onglet_lancer:
        _tab_lancer(projet, axes)
    with onglet_resultats:
        _tab_resultats(projet, axes)

    st.divider()
    _job.historique(projet)
