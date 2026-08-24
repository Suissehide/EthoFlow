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
from lib.analysis import group_analysis_files, parse_list_columns
from lib.config import require_project
from lib import runner
from views import _job

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
        help="Relit les colonnes de ton Excel (via metadata.yaml) et "
             "détecte celles qui ont entre 2 et 12 valeurs distinctes — "
             "les seules exploitables comme axe de comparaison.",
    )
    if not axes:
        st.info(
            "Aucun axe découvert pour l'instant. Clique sur **Découvrir "
            "les axes** ci-dessus — le résultat survit à un "
            "rafraîchissement de la page, il est relu depuis le log du job."
        )


def _case_group_by(axe: dict) -> bool:
    effectifs = _effectifs(axe["groupes"])
    libelle = f"**{axe['nom']}** — {axe['n_groupes']} groupes : {axe['groupes']}"
    valeur = st.checkbox(libelle, key=f"analyses_gb_{axe['nom']}")
    if effectifs and min(effectifs) < 3:
        st.warning(
            f"« {axe['nom']} » : au moins un groupe a moins de 3 sessions "
            "— en dessous de ce seuil, un motif y est ignoré faute de "
            "données suffisantes pour le test statistique."
        )
    return valeur


def _section_group_by(axes: list[dict]) -> list[str]:
    st.markdown("**`--group-by`** — axes simples à comparer")
    st.caption(
        "Le test s'adapte au nombre de groupes (Mann-Whitney à 2, "
        "Kruskal-Wallis à 3+), mais pas au nombre de sessions : un axe à "
        "6 groupes sur 16 sessions donne des p-values inexploitables. "
        "Vérifie les effectifs ci-dessous avant de conclure quoi que ce "
        "soit."
    )
    return [axe["nom"] for axe in axes if _case_group_by(axe)]


def _section_cross(noms_axes: list[str]) -> list[tuple[str, str]]:
    st.markdown("**`--cross`** — croiser deux axes (design factoriel)")
    if _CLE_CROSS not in st.session_state:
        st.session_state[_CLE_CROSS] = []
    paires: list[tuple[str, str]] = st.session_state[_CLE_CROSS]

    if len(noms_axes) < 2:
        st.caption(
            "Il faut au moins 2 axes découverts pour construire un "
            "croisement."
        )
    else:
        col_a, col_b, col_btn = st.columns([2, 2, 1])
        with col_a:
            a = st.selectbox("Axe A", options=noms_axes, key="analyses_cross_a")
        with col_b:
            options_b = [n for n in noms_axes if n != a]
            b = st.selectbox("Axe B", options=options_b, key="analyses_cross_b")
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Ajouter", key="analyses_cross_add"):
                if (a, b) not in paires and (b, a) not in paires:
                    paires.append((a, b))
                    st.rerun()

    for i, (a, b) in enumerate(list(paires)):
        col_txt, col_rm = st.columns([5, 1])
        col_txt.caption(f"`{a}_x_{b}` — {a} × {b}")
        if col_rm.button("Retirer", key=f"analyses_cross_rm_{i}"):
            paires.pop(i)
            st.rerun()

    return list(paires)


def _section_extended(noms_axes: list[str],
                      cross_pairs: list[tuple[str, str]]) -> tuple[bool, str | None]:
    st.markdown("**`--extended`** — bouts, thigmotaxie, quarts temporels")
    composites = [f"{a}_x_{b}" for a, b in cross_pairs]
    options = noms_axes + composites
    extended = st.checkbox(
        "Lancer les analyses étendues", key="analyses_extended",
        help="Nécessite les labels par frame — plus long que l'analyse de base.",
    )
    extended_by = None
    if extended:
        if not options:
            st.warning(
                "Aucun axe disponible pour `--extended-by` — découvre les "
                "axes ou ajoute un croisement d'abord."
            )
        else:
            extended_by = st.selectbox(
                "`--extended-by`", options=options, key="analyses_extended_by",
            )
    return extended, extended_by


def _section_labels(projet: Path) -> str | None:
    st.markdown("**`--labels`**")
    defaut = str(ML.path(projet)) if ML.exists(projet) else ""
    valeur = st.text_input(
        "Fichier de labels de motifs", value=defaut, key="analyses_labels",
        help="Sans lui, les figures affichent `motif_0`, `motif_1`, … au "
             "lieu d'un vrai nom de comportement.",
    )
    if not valeur:
        st.caption(
            "Pas de `motif_labels.csv` pour ce projet — remplis-le dans la "
            "page **Motifs** pour des figures lisibles."
        )
    return valeur or None


def _section_options_avancees(projet: Path) -> dict:
    with st.expander("Options avancées", expanded=False):
        algos_presents = sorted({VA.parse_algo_n(a)[0] for a in VA.list_algos(projet)})
        options_algo = algos_presents or ["hmm", "kmeans"]
        algo = st.selectbox("`--algo`", options=options_algo, key="analyses_algo")

        preciser_n = st.checkbox(
            "Préciser `--n-clusters`", key="analyses_preciser_n",
            help="Laissé décoché : le script prend l'unique segmentation "
                 "trouvée pour cet algo. À cocher seulement si plusieurs "
                 "granularités coexistent.",
        )
        n_clusters = None
        if preciser_n:
            n_actuel = VA.n_clusters(projet) or 15
            n_clusters = int(st.number_input(
                "`--n-clusters`", min_value=2, value=int(n_actuel), step=1,
                key="analyses_n_clusters",
            ))

        mask_empty = st.checkbox(
            "`--mask-empty` (exclure les frames « arène vide » en bord "
            "d'enregistrement)", key="analyses_mask_empty",
        )
        col1, col2 = st.columns(2)
        with col1:
            min_edge_frames = int(st.number_input(
                "`--min-edge-frames`", min_value=1, value=25, step=1,
                key="analyses_min_edge_frames",
            ))
        with col2:
            fps = float(st.number_input(
                "`--fps`", min_value=1.0, value=30.0, step=1.0,
                key="analyses_fps",
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
        min_edge_frames=options["min_edge_frames"],
        fps=options["fps"],
    )
    _job.bouton_lancer(
        projet, "Lancer l'analyse", cmd, cle="btn_analyses_lancer",
        help="Sortie dans `data/vame/analysis/` — onglet **Résultats**.",
    )


# ============================================================
# Onglet "Résultats"
# ============================================================

def _apercu_csv(path: Path) -> None:
    st.markdown(f"**{path.name}**")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Lecture impossible : {e}")
        return
    st.caption(f"{len(df)} lignes × {len(df.columns)} colonnes")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        f"Télécharger {path.name}", data=path.read_bytes(),
        file_name=path.name, mime="text/csv", key=f"analyses_dl_{path.name}",
    )


def _affiche_fichiers(fichiers: list[Path]) -> None:
    images = [p for p in fichiers if p.suffix.lower() == ".png"]
    csvs = [p for p in fichiers if p.suffix.lower() == ".csv"]
    for p in images:
        st.image(str(p), caption=p.name, use_container_width=True)
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
        col_nom, col_dl = st.columns([4, 1])
        col_nom.caption(f"`{p.name}`")
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
            "Pas de dossier `analysis/` pour ce projet — lance une analyse "
            "depuis l'onglet **Lancer une analyse**."
        )
        return

    fichiers = sorted(p for p in analysis_dir.iterdir() if p.is_file())
    if not fichiers:
        st.info("`analysis/` existe mais est vide.")
        return

    # Axes réellement découverts (--list-columns) : plus fiables que toute
    # règle inférée du nom de fichier pour lever les ambiguïtés (ruling
    # R17.1) — passés à group_analysis_files quand disponibles.
    axes_connus = [axe["nom"] for axe in axes] or None
    globaux, par_axe, autres = group_analysis_files(fichiers, axes_connus)

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


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    projet = require_project()

    st.title("Analyses")
    st.caption(
        "Étape 9 du pipeline : transforme la segmentation VAME en "
        "statistiques et figures. Toute colonne de ton Excel — y compris "
        "celles que tu as inventées — peut servir d'axe de comparaison."
    )

    _job.panneau(projet)

    # Calculé une seule fois : sert à la fois aux cases --group-by et à
    # désambiguïser le regroupement des fichiers dans l'onglet Résultats
    # (ruling R17.1).
    axes = _axes_disponibles(projet)

    onglet_lancer, onglet_resultats = st.tabs(["Lancer une analyse", "Résultats"])
    with onglet_lancer:
        _tab_lancer(projet, axes)
    with onglet_resultats:
        _tab_resultats(projet, axes)

    st.divider()
    _job.historique(projet)
