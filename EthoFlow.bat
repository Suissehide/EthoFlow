@echo off
rem EthoFlow - double-clic pour installer (la premiere fois) et lancer l'app.
rem Windows. (macOS / Linux : EthoFlow.command.)
rem
rem A chaque lancement :
rem   1. trouve conda - et propose d'installer Miniforge s'il n'y en a pas ;
rem   2. cree les environnements absents (ethoflow, dlc, vame), met a jour
rem      ceux dont le fichier environment-*.yml a change (git pull), et
rem      complete l'env ethoflow s'il lui manque une dependance ;
rem   3. verifie que torch voit la GPU NVIDIA dans dlc et vame, lance
rem      `diagnose_gpu.py --fix` sinon (une fois par environnement) ;
rem   4. lance l'app (ethoflow.cmd) : elle s'ouvre dans le navigateur.
setlocal EnableDelayedExpansion
chcp 65001 >nul
rem Python ecrit en UTF-8 meme quand sa sortie est redirigee (>nul) : sans
rem ca il prend l'encodage Windows (cp1252), plante sur le premier symbole
rem et la verification GPU croit a une panne qui n'existe pas.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ICI=%~dp0"
cd /d "%ICI%"

rem Modules indispensables a l'app : s'il en manque un, l'env ethoflow est
rem complete depuis environment-pipeline.yml avant le lancement.
set "IMPORTS_APP=streamlit, streamlit_image_coordinates, cv2, pandas, numpy, yaml, tqdm, openpyxl, tables"

title EthoFlow
echo.
echo  EthoFlow                                         %date% %time:~0,5%
echo  ==================================================================

rem ---------------------------------------------------------------- conda
call :etape 1 "Conda"
set "CONDA="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA=%CONDA_EXE%"
if not defined CONDA (
    for %%D in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%USERPROFILE%\miniconda3" ^
                "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" ^
                "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
        if not defined CONDA if exist "%%~D\Scripts\conda.exe" set "CONDA=%%~D\Scripts\conda.exe"
    )
)
if not defined CONDA (
    for /f "delims=" %%C in ('where conda.exe 2^>nul') do if not defined CONDA set "CONDA=%%C"
)
if not defined CONDA (
    call :ligne INFO "conda introuvable - EthoFlow en a besoin pour ses environnements Python"
    set /p "REP=        Installer Miniforge dans %USERPROFILE%\miniforge3 (~100 Mo, sans droits admin) ? [O/n] "
    if /I "!REP!"=="n" goto :erreur_annule
    set "INSTALLEUR=%TEMP%\Miniforge3-Windows-x86_64.exe"
    curl.exe -fsSL -o "!INSTALLEUR!" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
    if errorlevel 1 (set "MSG=telechargement de Miniforge impossible" & goto :erreur)
    call :ligne ... "installation de Miniforge"
    start /wait "" "!INSTALLEUR!" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%USERPROFILE%\miniforge3
    del "!INSTALLEUR!" >nul 2>&1
    set "CONDA=%USERPROFILE%\miniforge3\Scripts\conda.exe"
    if not exist "!CONDA!" (set "MSG=l'installation de Miniforge a echoue" & goto :erreur)
)
rem Le lanceur ethoflow.cmd s'en sert pour trouver l'env.
set "CONDA_EXE=%CONDA%"
for %%A in ("%CONDA%") do set "SCRIPTS=%%~dpA"
for %%B in ("%SCRIPTS%..") do set "BASE=%%~fB"
call :ligne OK "%CONDA%"

rem ------------------------------------------------------- environnements
rem Une copie du .yml est gardee dans l'env : si elle differe (apres un git
rem pull), l'env est mis a jour. Un env dlc/vame cree a la main, sans copie,
rem est laisse tel quel - une mise a jour pourrait y remplacer un torch
rem repare. L'env ethoflow n'a pas ce risque : on y verifie les imports.
call :etape 2 "Environnements"
call :preparer_env ethoflow environment-pipeline.yml 1 || goto :erreur
call :completer_ethoflow || goto :erreur
call :preparer_env dlc environment-dlc.yml 0
call :preparer_env vame environment-vame.yml 0

rem ------------------------------------------------------------------ GPU
call :etape 3 "GPU"
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :ligne INFO "pas de GPU NVIDIA : DLC et VAME tourneront sur CPU (10 a 50x plus lent)"
) else (
    for /f "delims=" %%G in ('nvidia-smi --query-gpu^=name^,driver_version --format^=csv^,noheader 2^>nul') do (
        call :ligne INFO "%%G"
    )
    call :verifier_gpu dlc
    call :verifier_gpu vame
)

rem ----------------------------------------------------------- lancement
call :etape 4 "Application"
call :ligne INFO "http://localhost:8501  (s'ouvre dans le navigateur)"
call :ligne INFO "garde cette fenetre ouverte ; ferme-la pour arreter EthoFlow"
echo.
call "%ICI%ethoflow.cmd"
if errorlevel 1 (set "MSG=l'app s'est arretee sur une erreur (voir ci-dessus)" & goto :erreur)
exit /b 0


rem ======================================================== sous-routines

:etape
rem %1 numero, %2 titre
echo.
echo  [%~1/4] %~2
exit /b 0

:ligne
rem %1 statut (OK, INFO, ATTENTION, ...), %2 message - colonnes alignees
set "STATUT=%~1          "
echo        !STATUT:~0,10!%~2
exit /b 0

:preparer_env
rem %1 nom, %2 fichier .yml, %3 obligatoire (1/0)
set "PREFIXE=%BASE%\envs\%~1"
set "COPIE=%PREFIXE%\.ethoflow-env.yml"
if not exist "%PREFIXE%\" (
    call :ligne ... "%~1 : creation (10 a 20 min, une seule fois)"
    call "%CONDA%" env create -f "%~2" -q
    if errorlevel 1 (
        rem Un env a moitie cree passerait pour installe au prochain lancement.
        call "%CONDA%" env remove -n %~1 -y >nul 2>&1
        if "%~3"=="1" (set "MSG=impossible de creer l'environnement %~1" & exit /b 1)
        call :ligne ATTENTION "%~1 : creation impossible - les etapes qui en dependent seront indisponibles"
        exit /b 0
    )
    copy /y "%~2" "%COPIE%" >nul
    call :ligne OK "%~1 (cree)"
    exit /b 0
)
if not exist "%COPIE%" (
    copy /y "%~2" "%COPIE%" >nul
    call :ligne OK "%~1"
    exit /b 0
)
fc /b "%~2" "%COPIE%" >nul 2>&1
if errorlevel 1 (
    call :ligne ... "%~1 : mise a jour (%~2 a change)"
    call "%CONDA%" env update -n %~1 -f "%~2" --prune -q
    if errorlevel 1 (
        call :ligne ATTENTION "%~1 : mise a jour echouee - l'ancienne version est gardee"
    ) else (
        copy /y "%~2" "%COPIE%" >nul
        rem torch a pu changer : reverifier la GPU.
        del "%PREFIXE%\.ethoflow-gpu-ok" >nul 2>&1
        call :ligne OK "%~1 (mis a jour)"
    )
    exit /b 0
)
call :ligne OK "%~1"
exit /b 0

:completer_ethoflow
rem L'env ethoflow peut dater d'avant l'ajout d'une dependance au .yml :
rem s'il lui manque un module, on le complete (sans --prune : on ne retire
rem rien de ce que l'utilisateur aurait installe).
set "PY=%BASE%\envs\ethoflow\python.exe"
if not exist "%PY%" exit /b 0
"%PY%" -c "import %IMPORTS_APP%" >nul 2>&1
if not errorlevel 1 exit /b 0
call :ligne ... "ethoflow : dependances manquantes, installation"
call "%CONDA%" env update -n ethoflow -f environment-pipeline.yml -q
"%PY%" -c "import %IMPORTS_APP%" >nul 2>&1
if errorlevel 1 (
    set "MSG=l'env ethoflow reste incomplet - detail : %PY% -c import %IMPORTS_APP%"
    exit /b 1
)
call :ligne OK "ethoflow (complete)"
exit /b 0

:verifier_gpu
rem %1 nom de l'env
set "PREFIXE=%BASE%\envs\%~1"
if not exist "%PREFIXE%\python.exe" exit /b 0
if exist "%PREFIXE%\.ethoflow-gpu-ok" (call :ligne OK "%~1 : torch utilise la GPU" & exit /b 0)
"%PREFIXE%\python.exe" scripts\diagnose_gpu.py >nul 2>&1
if not errorlevel 1 goto :gpu_ok
call :ligne ... "%~1 : torch ne voit pas la GPU, reparation"
"%PREFIXE%\python.exe" scripts\diagnose_gpu.py --fix
"%PREFIXE%\python.exe" scripts\diagnose_gpu.py >nul 2>&1
if not errorlevel 1 goto :gpu_ok
call :ligne ATTENTION "%~1 : torch ne voit toujours pas la GPU - detail : %PREFIXE%\python.exe scripts\diagnose_gpu.py"
exit /b 0
:gpu_ok
type nul > "%PREFIXE%\.ethoflow-gpu-ok"
call :ligne OK "%~1 : torch utilise la GPU"
exit /b 0


:erreur_annule
set "MSG=installation annulee"
:erreur
echo.
echo  ERREUR  %MSG%
echo.
pause
exit /b 1
