@echo off
rem EthoFlow — lance l'interface Streamlit. Windows.
rem (Sous Linux et macOS, utilise ./ethoflow a la place.)
rem
rem Usage :
rem     ethoflow                            ouvre l'app dans le navigateur
rem     ethoflow --server.address=0.0.0.0   accessible depuis le LAN
rem
rem Tout argument supplementaire est transmis tel quel a `streamlit run`.
rem Pas besoin d'activer l'env conda au prealable : le script trouve le
rem binaire de l'env `ethoflow` lui-meme.
setlocal EnableDelayedExpansion

set "ENV_NAME=ethoflow"
set "ICI=%~dp0"
set "APP=%ICI%streamlit_app\app.py"

if not exist "%APP%" (
    echo Erreur : "%APP%" introuvable. 1>&2
    echo Ce script doit rester a la racine du depot EthoFlow. 1>&2
    exit /b 1
)

rem On appelle le binaire de l'env directement plutot que `conda run` : ce
rem dernier tamponne la sortie tant que le process tourne, donc l'URL du
rem serveur n'apparaitrait jamais, et il s'interpose sur Ctrl-C.
set "STREAMLIT="

rem 1. L'env est deja active dans ce shell.
if /I "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" (
    for /f "delims=" %%S in ('where streamlit 2^>nul') do (
        if not defined STREAMLIT set "STREAMLIT=%%S"
    )
)

rem 2. Sinon, deduire la racine des envs et viser l'env directement.
if not defined STREAMLIT (
    set "BASE="
    if defined CONDA_EXE (
        rem %CONDA_EXE% vaut <base>\Scripts\conda.exe : deux niveaux au-dessus.
        for %%A in ("%CONDA_EXE%") do set "DOSSIER_SCRIPTS=%%~dpA"
        for %%B in ("!DOSSIER_SCRIPTS!..") do set "BASE=%%~fB"
    )
    if not defined BASE if defined MAMBA_ROOT_PREFIX set "BASE=%MAMBA_ROOT_PREFIX%"
    if not defined BASE (
        for /f "delims=" %%C in ('conda info --base 2^>nul') do set "BASE=%%C"
    )
    if defined BASE (
        if exist "!BASE!\envs\%ENV_NAME%\Scripts\streamlit.exe" (
            set "STREAMLIT=!BASE!\envs\%ENV_NAME%\Scripts\streamlit.exe"
        )
    )
)

if not defined STREAMLIT (
    echo Erreur : l'environnement conda "%ENV_NAME%" est introuvable. 1>&2
    echo. 1>&2
    echo Cree-le depuis la racine du depot : 1>&2
    echo     conda env create -f environment-pipeline.yml 1>&2
    echo. 1>&2
    echo ^(voir README.md, section "Installation"^) 1>&2
    exit /b 1
)

"%STREAMLIT%" run "%APP%" %*
exit /b %ERRORLEVEL%
