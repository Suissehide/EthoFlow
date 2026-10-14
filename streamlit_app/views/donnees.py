"""Page Données — étapes 2 et 3 du pipeline.

Étape 2 (README) : remplir l'Excel maître généré par `create_project.py`.
Étape 3 : le transformer en `data/raw/<session>/metadata.yaml` via
`sync_from_excel.py`.

L'Excel est la source de vérité : son contenu se lit et se modifie ici
(`lib.excel`, écriture cellule par cellule qui préserve la mise en forme),
puis le sync le propage aux `metadata.yaml`, qui ne sont donc qu'affichés.
Le script reste la seule autorité sur le format : chaque enregistrement
et chaque dépôt est relu par `sync_from_excel.py` avant de remplacer le
fichier.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import lib.excel as XL
import lib.pipeline as PL
from lib.config import excel_path, project_kind, require_project
from lib.icons import lucide_title
from lib.sessions import (
    arenes_dataframe,
    list_sessions,
    load_metadata,
    metadata_fields,
)
from views import _job
from views import _widgets
from views._widgets import champ_chemin


# ============================================================
# Section 1 : Excel maître
# ============================================================

def _section_excel(projet: Path) -> Path | None:
    st.markdown(lucide_title("file-spreadsheet", "Fichier Excel"), unsafe_allow_html=True)
    st.caption(
        "Le classeur du projet : une ligne par vidéo. Modifie-le directement "
        "ci-dessous, ou télécharge-le, remplis-le et redépose-le."
    )

    with st.expander("Colonnes attendues", expanded=False):
        st.markdown(
            "- **`id`** (obligatoire) — le nom du fichier vidéo, sans "
            "extension : `970-M1` pour `970-M1.mp4`. Une ligne = une session.\n"
            "- **`mouse_id`** (recommandée) — l'animal. Il se répète si la "
            "même souris est filmée plusieurs fois.\n"
            "- **`group`** (recommandée) — le groupe à comparer.\n"
            "- **Toutes les autres colonnes sont libres** : chacune devient un "
            "critère de comparaison dans **Analyses**. Une colonne "
            "`regime_alimentaire` ajoutée ici y apparaîtra.\n\n"
            "Projet à plusieurs animaux par vidéo : trois feuilles "
            "(`Subjects`, `Trials_Videos`, `Arena_Mapping`) au lieu de "
            "`Sessions`, reconnues automatiquement."
        )

    chemin_excel = excel_path(projet)

    if chemin_excel is None:
        st.warning(
            f"Aucun fichier Excel dans le dossier du projet. Dépose-en un "
            f"ci-dessous ; il sera enregistré sous `{projet.name}_sessions.xlsx`."
        )
    else:
        st.success(f"Excel trouvé : `{chemin_excel.name}`")
        st.download_button(
            "Télécharger",
            data=chemin_excel.read_bytes(),
            file_name=chemin_excel.name,
            key="btn_telecharger_excel",
        )

    st.markdown('<p class="ef-section">Déposer une version remplie</p>', unsafe_allow_html=True)

    # Astuce de clé versionnée (comme l'ancien navigateur de fichiers) :
    # après confirmation ou annulation, on incrémente la version pour que
    # Streamlit instancie un widget neuf plutôt que de garder le fichier
    # affiché comme « en attente » indéfiniment.
    version_key = "_donnees_upload_version"
    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    upload = st.file_uploader(
        "Remplace le fichier ci-dessus après confirmation — jamais en silence",
        type=["xlsx"],
        key=f"upload_excel_v{st.session_state[version_key]}",
        label_visibility="collapsed",
    )

    if upload is not None:
        cible = chemin_excel or (projet / f"{projet.name}_sessions.xlsx")
        st.warning(
            f"Écraser `{cible.name}` avec `{upload.name}` ? "
            "Cette action est irréversible."
        )
        col_annuler, col_confirmer = st.columns(2)
        with col_annuler:
            if st.button("Annuler", key="btn_annuler_upload_excel"):
                st.session_state[version_key] += 1
                st.rerun()
        with col_confirmer:
            if st.button("Oui, écraser", key="btn_confirmer_upload_excel", type="primary"):
                # Relu par sync_from_excel AVANT d'écraser : un fichier qui
                # n'est pas un classeur EthoFlow ne remplace jamais le bon.
                tmp = XL.dossier_sauvegardes(projet) / "depot.xlsx"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(upload.getvalue())
                try:
                    XL.valider(tmp)
                except ValueError as e:
                    st.error(f"Fichier refusé, rien n'a été remplacé : {e}")
                else:
                    if cible.exists():
                        cible.replace(XL.dossier_sauvegardes(projet) / XL.nom_sauvegarde(cible))
                    tmp.replace(cible)
                    st.session_state[version_key] += 1
                    st.toast(f"`{cible.name}` remplacé")
                    st.rerun()
                finally:
                    tmp.unlink(missing_ok=True)

    return chemin_excel


# ============================================================
# Section 1 bis : contenu de l'Excel, éditable
# ============================================================

def _section_contenu(projet: Path, chemin_excel: Path | None) -> None:
    if chemin_excel is None:
        return
    st.markdown(lucide_title("table", "Contenu de l'Excel"), unsafe_allow_html=True)

    if XL.ouvert_dans_excel(chemin_excel):
        st.warning(
            f"`{chemin_excel.name}` est ouvert dans Excel. Ferme-le avant "
            "d'éditer ici : le prochain enregistrement d'Excel écraserait "
            "les modifications faites dans l'app."
        )
    if XL.modifie_depuis_sync(projet, chemin_excel):
        st.info(
            "L'Excel a changé depuis la création des sessions : relance "
            "**Créer les sessions** ci-dessous avec « Mettre à jour les "
            "sessions déjà créées » coché pour qu'elles suivent."
        )

    try:
        feuilles = XL.feuilles_editables(chemin_excel)
    except Exception as e:  # classeur corrompu, pas un xlsx...
        st.error(f"Lecture impossible : {e}")
        return
    if not feuilles:
        st.error("Ni feuille `Sessions`, ni `Subjects`/`Trials_Videos`/`Arena_Mapping`.")
        return
    feuille = (st.radio("Feuille", feuilles, horizontal=True, key="donnees_excel_feuille")
               if len(feuilles) > 1 else feuilles[0])

    avant = XL.lire_feuille(chemin_excel, feuille)
    # Clé versionnée : après un enregistrement, l'éditeur repart du
    # fichier relu au lieu de rejouer ses modifications par-dessus.
    version = st.session_state.setdefault("_donnees_excel_version", 0)
    apres = st.data_editor(
        avant,
        column_config={c: st.column_config.TextColumn(c) for c in avant.columns},
        num_rows="dynamic", hide_index=True, width="stretch",
        key=f"donnees_excel_{feuille}_{version}",
    )

    col_nom, col_btn = st.columns([2, 1])
    with col_nom:
        nouvelle = st.text_input(
            "Ajouter une colonne", key=f"donnees_excel_nouvelle_{version}",
            placeholder="ex : regime_alimentaire",
            help="Toute colonne devient un axe de comparaison dans les analyses.",
        ).strip()
    if nouvelle in avant.columns:
        st.error(f"La colonne `{nouvelle}` existe déjà.")
        nouvelle = ""
    if nouvelle:
        apres = apres.assign(**{nouvelle: ""})

    with col_btn:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        enregistrer = st.button(
            "Enregistrer dans l'Excel", key="btn_excel_enregistrer", type="primary",
            disabled=not XL.a_change(avant, apres), width="stretch",
        )
    if enregistrer:
        try:
            sauvegarde = XL.enregistrer(projet, chemin_excel, feuille, avant, apres)
        except ValueError as e:
            st.error(f"Refusé, rien n'a été écrit : {e}")
        except XL.ExcelVerrouille as e:
            st.error(str(e))
        else:
            st.session_state["_donnees_excel_version"] = version + 1
            st.toast(f"Enregistré — ancienne version : `{sauvegarde.name}`")
            st.rerun()


# ============================================================
# Section 2 : Sync avec aperçu
# ============================================================

def _section_sync(projet: Path, chemin_excel: Path | None) -> None:
    st.markdown(lucide_title("play", "Créer les sessions"), unsafe_allow_html=True)
    st.caption(
        "Une session par ligne de l'Excel, reliée à sa vidéo. Lance d'abord "
        "l'aperçu : il n'écrit rien et signale les lignes dont la vidéo est "
        "introuvable."
    )

    videos_dir = champ_chemin(
        "Dossier des vidéos",
        cle="donnees_videos_dir",
        valeur_defaut=st.session_state.get("_donnees_videos_dir", ""),
        placeholder="ex : E:/data/bottom_view/08062026",
        titre_dialogue="Dossier des vidéos à synchroniser",
    )
    st.session_state["_donnees_videos_dir"] = videos_dir

    col_ext, col_overwrite = st.columns([1, 2], vertical_alignment="bottom")
    with col_ext:
        ext = st.text_input("Extension des vidéos", value="mp4", key="donnees_video_ext")
    with col_overwrite:
        overwrite = st.checkbox(
            "Mettre à jour les sessions déjà créées",
            value=False, key="donnees_overwrite",
            help="Réécrit leurs informations depuis l'Excel. À cocher après "
                 "une correction dans l'Excel.",
        )

    pas_de_videos_dir = not videos_dir.strip()
    aide = "Renseigne le dossier des vidéos d'abord." if pas_de_videos_dir else None

    cmd_apercu = PL.sync_from_excel(
        projet, videos_dir=videos_dir, excel=chemin_excel,
        video_ext=ext or "mp4", dry_run=True,
    )
    cmd_reel = PL.sync_from_excel(
        projet, videos_dir=videos_dir, excel=chemin_excel,
        video_ext=ext or "mp4", overwrite=overwrite,
    )
    # Boutons côte à côte, raison écrite une fois, à leur droite (maquette).
    with st.container(horizontal=True, vertical_alignment="center"):
        _job.bouton_lancer(
            projet, "Aperçu (n'écrit rien)", cmd_apercu,
            cle="btn_sync_apercu", type="secondary",
            disabled=pas_de_videos_dir, help=aide, afficher_raison=False,
        )
        _job.bouton_lancer(
            projet, "Créer les sessions", cmd_reel,
            cle="btn_sync_reel", type="primary",
            disabled=pas_de_videos_dir, help=aide,
        )

    _job.panneau(projet)
    _widgets.details_techniques(cmd_reel, notes=[
        "Aperçu : même commande avec `--dry-run`. Chaque session est écrite "
        "dans `data/raw/<session>/metadata.yaml`."])
    _job.historique(projet)


# ============================================================
# Section 3 : Sessions synchronisées
# ============================================================

def _section_sessions(projet: Path) -> None:
    st.markdown(lucide_title("layout-dashboard", "Sessions créées"), unsafe_allow_html=True)

    df = list_sessions(projet)
    if df.empty:
        st.info("Aucune session pour l'instant : crée-les ci-dessus.")
        return

    # Pas de matrice sessions × étapes (handoff) : l'avancement se lit par
    # étape sur le tableau de bord ; ici, juste les sessions et leur vidéo.
    df = df[["session_id", "vidéo"]].rename(columns={"session_id": "Session", "vidéo": "Vidéo"})
    n_total = len(df)
    n_video_ok = int((df["Vidéo"] == "OK").sum())

    st.caption(f"{n_total} session{'s' if n_total > 1 else ''} · "
               f"{n_video_ok} vidéo{'s' if n_video_ok > 1 else ''} trouvée{'s' if n_video_ok > 1 else ''}")
    st.dataframe(df.assign(Vidéo=df["Vidéo"].map({"OK": "trouvée", "manque": "introuvable"})),
                 width="stretch", hide_index=True)

    _section_metadata(projet, df["Session"].tolist())

    if n_video_ok < n_total:
        st.caption(
            f"{n_total - n_video_ok} session(s) dont la vidéo est introuvable : "
            "retrouve-les depuis la page **Vidéos & calibration**."
        )


# ============================================================
# Entrée
# ============================================================

def render() -> None:
    projet = require_project()

    _widgets.entete_etape(projet)

    chemin_excel = _section_excel(projet)
    st.divider()
    if chemin_excel is not None:
        _section_contenu(projet, chemin_excel)
        st.divider()
    _section_sync(projet, chemin_excel)
    st.divider()
    _section_sessions(projet)


# ============================================================
# Metadata d'une session — consultation
# ============================================================

def _section_metadata(projet: Path, sessions: list[str]) -> None:
    """Ce que le sync a écrit pour une session, en lecture seule.

    Pas d'édition ici : l'Excel est la source de vérité, et un sync avec
    `--overwrite` réécrit `metadata.yaml` depuis l'Excel — une correction
    faite dans le YAML serait perdue au sync suivant. On corrige dans
    « Contenu de l'Excel », puis on resynchronise.
    """
    if not sessions:
        return

    st.divider()
    st.markdown(lucide_title("file-spreadsheet", "Informations d'une session"),
                unsafe_allow_html=True)
    st.caption(
        "Lecture seule : ce que l'Excel a donné pour cette session. Pour "
        "corriger, modifie le **Contenu de l'Excel** ci-dessus, puis recrée "
        "les sessions avec « Mettre à jour les sessions déjà créées »."
    )

    session = st.selectbox("Session", options=sessions, key="donnees_meta_session")
    meta = load_metadata(projet, session)
    if meta is None:
        st.error(f"Informations introuvables pour la session {session}.")
        return

    champs = metadata_fields(meta)
    if champs:
        st.dataframe(
            pd.DataFrame([{"champ": c, "valeur": "" if v is None else str(v)}
                          for c, v in champs.items()]),
            hide_index=True, width="stretch",
        )
    else:
        st.caption("Cette session n'a aucun champ exploitable.")

    arenes = arenes_dataframe(meta)
    if not arenes.empty:
        st.markdown("**Arènes**")
        st.dataframe(arenes, hide_index=True, width="stretch")
