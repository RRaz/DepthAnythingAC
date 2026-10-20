@echo off
setlocal enabledelayedexpansion

rem ===========================================================================
rem  Depth Anything AC - drop-zone runner (CPU only)
rem
rem  EASIEST USE - no arguments at all:
rem      Put images, or whole folders of images, into the "input" folder that
rem      sits next to this file, then double-click this file.
rem
rem  OR - drag folders (or images) onto this file in Explorer.
rem
rem  Either way each input gets two files beside it, in a "depth" subfolder and
rem  named after the source, so they cannot be mixed up with the originals:
rem
rem      D:\photos\IMG_0001.tif   ->  D:\photos\depth\IMG_0001_depth.tif
rem                                  D:\photos\depth\IMG_0001_depth_color.png
rem
rem  The .tif is the 16-bit greyscale depth map -- the one to use as a mask.
rem  The .png is an 8-bit colour preview, only there to be looked at; turn it
rem  off with --no-color-map.
rem
rem  Re-running is safe: results are skipped when they are found as inputs
rem  again, so the folders do not grow every time.
rem
rem  Explicit paths also work - any argument that exists on disk is an input.
rem      run_depth.bat "D:\photos" "E:\more photos"
rem
rem  No options are needed.  Depth is computed at the model's 518 px working
rem  size and interpolated back, so the map always has the input's dimensions.
rem  Nothing is downscaled and nothing is refused for size.
rem
rem  Any argument starting with "-" is passed straight to local_depth.py and
rem  applied to every run:
rem      --dest output          write to the project "output" folder instead
rem      --dest subfolder       subfolder only, no "_depth" suffix
rem      --dest suffix          "_depth" beside the originals, no subfolder
rem      --subfolder masks      rename the subfolder
rem      --suffix _z            change the suffix
rem      --ext png              16-bit map as PNG instead of TIFF
rem      --colormap inferno     preview colours: spectral (default), inferno, gray
rem      --no-color-map         no 8-bit preview
rem      --manifest run.json    JSON record of the run for another program
rem
rem  This wrapper always processes subfolders.  For everything else run
rem      .venv\Scripts\python.exe local_depth.py --help
rem ===========================================================================

set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
set "INROOT=%HERE%input"
set "CKPT=%HERE%checkpoints\depth_anything_AC_vits.pth"

rem Keep the window open when the file was double-clicked, but not when the
rem command was typed into an already-open prompt.  Set DEPTH_NO_PAUSE=1 to
rem suppress the pause unconditionally, e.g. when calling this from a script.
set "HOLD="
if defined DEPTH_NO_PAUSE goto :no_pause
echo %cmdcmdline% | find /i "%~nx0" >nul
if not errorlevel 1 set "HOLD=1"
:no_pause

if not exist "%PY%" (
    echo [ERROR] Python environment not found:
    echo         %PY%
    echo         Run "python -m venv .venv" in this folder and install requirements.txt.
    goto :end
)
if not exist "%CKPT%" (
    echo [ERROR] No pretrained checkpoint found:
    echo         %CKPT%
    echo         Download depth_anything_AC_vits.pth from the links in README.md
    echo         and place it in the checkpoints folder.
    goto :end
)

if not exist "%INROOT%" mkdir "%INROOT%"

rem ---- sort the arguments into options and inputs ---------------------------
rem An argument beginning with "-" is an option.  Whatever follows an option is
rem treated as its value, even if a file of that name happens to exist, so that
rem things like "--dest output" and "--ext png" are not mistaken for inputs.
rem Every remaining argument must exist on disk to count as an input.
set "OPTIONS="
set "COUNT=0"
set "PREVOPT="
:parse
if "%~1"=="" goto :parsed
set "ARG=%~1"
if "!ARG:~0,1!"=="-" goto :opt
if defined PREVOPT goto :opt_value
if exist "!ARG!" goto :inp
echo [WARN] not found on disk, passing through as an option: !ARG!
goto :opt

:opt
set "PREVOPT=1"
set "OPTIONS=!OPTIONS! !ARG!"
shift
goto :parse

:opt_value
set "PREVOPT="
set "OPTIONS=!OPTIONS! !ARG!"
shift
goto :parse

:inp
set "PREVOPT="
set /a COUNT+=1
set "INPUT_!COUNT!=!ARG!"
shift
goto :parse
:parsed

if !COUNT! EQU 0 (
    rem Nothing dropped, so use the drop-zone folder.
    set /a COUNT=1
    set "INPUT_1=%INROOT%"
    echo Nothing was dropped, so processing the "input" folder.
    dir /b /s "%INROOT%\*.jpg" "%INROOT%\*.jpeg" "%INROOT%\*.png" "%INROOT%\*.bmp" "%INROOT%\*.tif" "%INROOT%\*.tiff" "%INROOT%\*.webp" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo No images found in:
        echo     %INROOT%
        echo Put image files, or folders of images, in there and run again.
        goto :end
    )
)

echo.
for /l %%K in (1,1,!COUNT!) do call :run_one "!INPUT_%%K!"

goto :end

rem ---------------------------------------------------------------------------
rem  Process one input folder or image.  %1 = path.
rem ---------------------------------------------------------------------------
:run_one
set "SRC=%~1"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
echo =====================================================================
echo Source : %SRC%
echo =====================================================================
"%PY%" "%HERE%local_depth.py" --input "%SRC%" --recursive !OPTIONS!
if errorlevel 1 echo [ERROR] local_depth.py exited with code %ERRORLEVEL%
echo.
exit /b 0

:end
echo.
if defined HOLD pause
exit /b 0
