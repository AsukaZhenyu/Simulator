@echo off
rem Run the accuracy campaign from a plain console, with nothing else open.
rem Close the editor, the browser, Word and any assistant first: these columns are
rem timing measurements and the numbers belong to whatever state the machine is in.
setlocal
cd /d "%~dp0"

set PY=modeling\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo.
echo   accuracy campaign : model, dense, e2a
echo   output            : build\accuracy-YYYYMMDD\campaign.log
echo.
echo   Anything still running will show up in the log's resident list.
echo   Press Ctrl+C now if a program you meant to close is still open.
echo.
%SystemRoot%\System32\timeout.exe /t 10 /nobreak >nul 2>&1

"%PY%" tools\run_accuracy_campaign.py %*
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
  echo Campaign finished. Read build\accuracy-*\CAMPAIGN_SUMMARY.md
) else (
  echo Campaign stopped with code %RC%. The last phase in build\accuracy-*\campaign.log failed.
)
echo.
pause
