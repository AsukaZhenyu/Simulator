@echo off
REM ---------------------------------------------------------------------------
REM Double-click this file to start the viewer and open the browser.
REM It picks the venv python itself, so nothing has to be typed or activated.
REM
REM Extra arguments are forwarded to tools\viewer.py, e.g.
REM     viewer.cmd --scenario examples\matvec-cap160.json
REM     viewer.cmd --export out.html
REM     viewer.cmd --self-check
REM
REM Kept ASCII on purpose: cmd.exe reads .cmd files in the OEM codepage, and a
REM UTF-8 Chinese comment can turn into stray punctuation that breaks parsing.
REM The Chinese explanation lives in README.md, section "Viewer".
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
set "PY=%~dp0..\modeling\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0tools\viewer.py" %*
if errorlevel 1 pause
