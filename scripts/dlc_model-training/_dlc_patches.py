"""Correctifs appliqués à DeepLabCut au runtime, avant de l'utiliser.

Pourquoi ce fichier plutôt qu'un patch dans site-packages : DLC est
installé par pip dans le conda env `dlc`, donc une édition manuelle de
site-packages saute au premier `pip install --upgrade` et ne suit pas le
repo d'une machine à l'autre. Ici le correctif voyage avec EthoFlow.

Chaque patch est idempotent et bruyant : il dit ce qu'il fait, et ne
touche rien si la version de DLC installée n'a plus le problème.

--------------------------------------------------------------------
Patch 1 — images annotées vides sur un projet mono-animal
--------------------------------------------------------------------
Symptôme, à la fin de `02_train.py` :

    Warning: DataFrame reshape failed for img12628.png
      Expected: 2 individuals, 12 bodyparts
      Ground truth: 24 elements (expected 48)
      Predictions: 36 elements (expected 72)
      Skipping visualization for this image

...répété pour CHAQUE image, et un dossier `LabeledImages_*` vide.

Cause : `evaluate_network(plotting=True)` fusionne deux DataFrames qui
nomment l'animal différemment. Nos projets sont mono-animal, donc la
vérité terrain n'a pas de niveau de colonnes `individuals` ; DLC lui en
fabrique un au vol en baptisant l'individu **"animal"**
(`apis/utils.py::ensure_multianimal_df_format`). Les prédictions, elles,
sont construites avec le nom d'individu du config.yaml — **"individual_1"**
(`data/dlcloader.py::build_dlc_dataframe_columns`). Le traceur compte
ensuite les individus sur l'UNION des deux scorers : 2 au lieu de 1. Il
attend donc 2x plus de valeurs que le DataFrame n'en contient, le
`reshape` lève ValueError, et l'image est sautée.

Les métriques (RMSE, mAP, mAR) ne sont PAS affectées : elles sont
calculées avant cette étape. Seules les images annotées manquent.

Correctif : avant de tracer, on réaligne les noms d'individus de la
vérité terrain sur ceux des prédictions. No-op quand les noms coïncident
déjà (vrai projet multi-animal) ou quand les deux côtés n'ont pas le même
nombre d'individus — dans ce cas on laisse DLC gérer.
"""
from __future__ import annotations

_APPLIED = False


def apply_patches() -> None:
    """Applique tous les correctifs DLC. Sûr à appeler plusieurs fois."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_plot_evaluation_results()
    _APPLIED = True


def _unique(values) -> list:
    """Valeurs uniques en conservant l'ordre d'apparition."""
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def align_individual_names(df_combined, scorer: str, model_name: str):
    """Renomme les individus de `scorer` pour coller à ceux de `model_name`.

    `scorer` est le labelliseur humain (vérité terrain), `model_name` le
    modèle. Renvoie le DataFrame inchangé si les noms coïncident déjà ou
    si les deux côtés n'ont pas le même nombre d'individus.
    """
    import pandas as pd

    cols = df_combined.columns
    if "individuals" not in (cols.names or []):
        return df_combined

    def idvs(target: str) -> list:
        return _unique(
            c[1] for c in cols if c[0] == target and c[1] != "single"
        )

    gt_idvs, pred_idvs = idvs(scorer), idvs(model_name)
    if gt_idvs == pred_idvs or len(gt_idvs) != len(pred_idvs):
        return df_combined

    # Appariement positionnel : en mono-animal il n'y a qu'un candidat,
    # et en multi-animal DLC construit les deux listes dans l'ordre du
    # config.yaml, donc la position est la bonne clé.
    mapping = dict(zip(gt_idvs, pred_idvs))
    print(f"  [patch] individus de '{scorer}' renommés pour le traçage : "
          f"{mapping}")

    df = df_combined.copy()
    df.columns = pd.MultiIndex.from_tuples(
        [(s, mapping.get(i, i) if s == scorer else i, b, c)
         for s, i, b, c in cols],
        names=cols.names,
    )
    return df


def _patch_plot_evaluation_results() -> None:
    """Enrobe `plot_evaluation_results` d'un réalignement des individus."""
    try:
        from deeplabcut.pose_estimation_pytorch.apis import evaluation
    except ImportError as err:  # DLC absent ou trop ancien
        print(f"⚠️  Patch images annotées non appliqué ({err})")
        return

    original = getattr(evaluation, "plot_evaluation_results", None)
    if original is None or getattr(original, "_ethoflow_patched", False):
        return

    def patched(*args, **kwargs):
        # DLC appelle cette fonction en tout-keyword ; on reste tolérant.
        if "df_combined" in kwargs:
            kwargs["df_combined"] = align_individual_names(
                kwargs["df_combined"], kwargs["scorer"], kwargs["model_name"],
            )
        elif args:
            args = (align_individual_names(
                args[0], kwargs["scorer"], kwargs["model_name"],
            ),) + args[1:]
        return original(*args, **kwargs)

    patched._ethoflow_patched = True
    evaluation.plot_evaluation_results = patched
