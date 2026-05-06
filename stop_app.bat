@echo off
setlocal
pushd "%~dp0"

set "PORT=7860"
set "FOUND=0"

echo Checking for Python listeners on port %PORT%...
for /f %%P in ('powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; if ($conn) { $conn | Select-Object -ExpandProperty OwningProcess -Unique }"') do (
    for /f %%N in ('powershell -NoProfile -Command "$p = Get-Process -Id %%P -ErrorAction SilentlyContinue; if ($p) { $p.ProcessName }"') do (
        if /I "%%N"=="python" (
            set "FOUND=1"
            echo Stopping Python process %%P using port %PORT%...
            powershell -NoProfile -Command "Stop-Process -Id %%P -Force"
        )
        if /I "%%N"=="pythonw" (
            set "FOUND=1"
            echo Stopping Python process %%P using port %PORT%...
            powershell -NoProfile -Command "Stop-Process -Id %%P -Force"
        )
    )
)

if "%FOUND%"=="0" (
    echo No Python process is listening on port %PORT%.
) else (
    echo Port %PORT% has been cleared.
)

popd
exit /b 0
