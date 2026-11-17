@echo off
rem EthoFlow - double-clic pour poser un raccourci "EthoFlow" (avec icone)
rem sur le Bureau. Windows, optionnel : le lanceur reste EthoFlow-Windows.bat.
rem Relancer ce script met le raccourci a jour (depot deplace, par exemple).
rem Le chemin passe par la variable d'environnement ICI : pas de souci de
rem guillemets ou d'apostrophes cote PowerShell.
for %%R in ("%~dp0..") do set "ICI=%%~fR\"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'EthoFlow.lnk'; $r = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk); $r.TargetPath = Join-Path $env:ICI 'EthoFlow-Windows.bat'; $r.WorkingDirectory = $env:ICI; $r.IconLocation = Join-Path $env:ICI 'scripts\ethoflow.ico'; $r.Description = 'EthoFlow - analyse comportementale (DeepLabCut + VAME)'; $r.Save(); Write-Host ('Raccourci cree : ' + $lnk)"
if errorlevel 1 (echo Erreur : le raccourci n'a pas pu etre cree.) 
pause
