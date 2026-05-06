@echo off
setlocal
pushd "%~dp0"

set "PORT=7860"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "APP_FILE=%~dp0app.py"

echo Checking for listeners on port %PORT%...
for /f %%P in ('powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; if ($conn) { $conn | Select-Object -ExpandProperty OwningProcess -Unique }"') do (
    for /f %%N in ('powershell -NoProfile -Command "$p = Get-Process -Id %%P -ErrorAction SilentlyContinue; if ($p) { $p.ProcessName }"') do (
        if /I "%%N"=="python" (
            echo Stopping Python process %%P using port %PORT%...
            powershell -NoProfile -Command "Stop-Process -Id %%P -Force"
        )
        if /I "%%N"=="pythonw" (
            echo Stopping Python process %%P using port %PORT%...
            powershell -NoProfile -Command "Stop-Process -Id %%P -Force"
        )
    )
)

if not exist "%PYTHON_EXE%" (
    echo Python executable not found: "%PYTHON_EXE%"
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo App file not found: "%APP_FILE%"
    exit /b 1
)

echo Starting app.py...
"%PYTHON_EXE%" "%APP_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
