#!/usr/bin/env bash
# EthoFlow — lance l'interface Streamlit. Linux et macOS.
# (Sous Windows : scripts\lancer_app.cmd.)
#
# Appelé par EthoFlow-Mac-Linux.command, le lanceur à double-cliquer de la
# racine — c'est lui qu'il faut utiliser. Ce script-ci démarre l'app seule,
# sans vérifier les environnements ni la GPU :
#     scripts/lancer_app.sh                            ouvre l'app
#     scripts/lancer_app.sh --server.address=0.0.0.0   ouverte au réseau local
#
# Tout argument supplémentaire est transmis tel quel à `streamlit run`.
# Pas besoin d'activer l'env conda au préalable : le script trouve le
# binaire de l'env `ethoflow` lui-même.
#
# Le script peut être lié dans un dossier du PATH — les liens symboliques
# sont résolus, donc `ln -s /chemin/vers/scripts/lancer_app.sh ~/.local/bin/ethoflow`
# fonctionne et l'app est trouvée à sa vraie place.
set -euo pipefail

ENV_NAME="ethoflow"

# Résolution portable des liens symboliques : `readlink -f` n'existe pas
# sur les BSD/macOS anciens, donc on déroule la chaîne à la main.
resoudre_dossier() {
    local src="$1" dossier
    while [ -L "$src" ]; do
        dossier="$(cd -P "$(dirname "$src")" && pwd)"
        src="$(readlink "$src")"
        # Un lien relatif se résout par rapport au dossier du lien.
        case "$src" in
            /*) ;;
            *) src="$dossier/$src" ;;
        esac
    done
    cd -P "$(dirname "$src")" && pwd
}

# Ce script vit dans scripts/ : la racine du dépôt est le dossier parent.
ICI="$(cd -P "$(resoudre_dossier "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ICI/streamlit_app/app.py"

if [ ! -f "$APP" ]; then
    echo "Erreur : $APP introuvable." >&2
    echo "Ce script doit rester dans scripts/ du dépôt EthoFlow" >&2
    echo "(un lien symbolique depuis le PATH est possible)." >&2
    exit 1
fi

# On appelle le binaire de l'env directement plutôt que `conda run` : ce
# dernier tamponne la sortie tant que le process tourne, donc l'URL du
# serveur n'apparaîtrait jamais, et il s'interpose sur Ctrl-C.
trouver_streamlit() {
    # 1. L'env est déjà activé dans ce shell.
    if [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ] && command -v streamlit >/dev/null 2>&1; then
        command -v streamlit
        return 0
    fi
    # 2. Sinon, déduire la racine des envs et viser l'env directement.
    #    conda, mamba et micromamba sont tous gérés : sur les machines
    #    Linux de labo, micromamba est fréquent et n'exporte pas CONDA_EXE.
    local racines=() base
    [ -n "${CONDA_EXE:-}" ] && racines+=("$(dirname "$(dirname "$CONDA_EXE")")")
    [ -n "${MAMBA_ROOT_PREFIX:-}" ] && racines+=("$MAMBA_ROOT_PREFIX")
    for outil in conda mamba micromamba; do
        if command -v "$outil" >/dev/null 2>&1; then
            base="$("$outil" info --base 2>/dev/null || true)"
            [ -n "$base" ] && racines+=("$base")
        fi
    done
    for base in "${racines[@]:-}"; do
        if [ -n "$base" ] && [ -x "$base/envs/$ENV_NAME/bin/streamlit" ]; then
            echo "$base/envs/$ENV_NAME/bin/streamlit"
            return 0
        fi
    done
    return 1
}

if ! STREAMLIT="$(trouver_streamlit)"; then
    cat >&2 <<EOF
Erreur : l'environnement conda « $ENV_NAME » est introuvable.

Crée-le depuis la racine du dépôt :

    conda env create -f environment-pipeline.yml

(voir README.md, section « Installation »)
EOF
    exit 1
fi

# Premier lancement : Streamlit demande une adresse e-mail et attend une
# réponse au clavier — serveur bloqué, navigateur jamais ouvert. Un fichier
# d'identifiants vide supprime la question.
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

# Depuis la racine du dépôt : Streamlit y lit .streamlit/config.toml (thème),
# que le lanceur soit appelé d'ici ou via un lien dans le PATH.
cd "$ICI"
exec "$STREAMLIT" run "$APP" "$@"
