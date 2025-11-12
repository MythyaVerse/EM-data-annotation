@echo off
setlocal enabledelayedexpansion

:: SETTINGS
set IMAGE=torch-fa-transformers-snapshot
set CONTAINER_NAME=DataAnnotation

set HOST_PATH=%cd%
set "DOCKER_PATH=%HOST_PATH:\=/%"
set "DOCKER_PATH=/%DOCKER_PATH:~0,1%%DOCKER_PATH:~2%"

echo Starting container: %CONTAINER_NAME% using image: %IMAGE%

:: Does a container with this name already exist?
docker ps -a -q -f name=%CONTAINER_NAME% | findstr . >nul
if %ERRORLEVEL% equ 0 (
    :: It exists. Is it running?
    for /f %%i in ('docker inspect -f "{{.State.Status}}" %CONTAINER_NAME%') do set STATUS=%%i
    if /I "%STATUS%"=="running" (
        echo Container is already running. Attaching shell...
        docker exec -it %CONTAINER_NAME% bash -lc "cd /workspace && bash"
    ) else (
        echo Starting existing container...
        docker start %CONTAINER_NAME% >nul
        timeout /t 2 >nul
        docker exec -it %CONTAINER_NAME% bash -lc "cd /workspace && bash"
    )
) else (
    echo Creating a new container...
    docker run --gpus all --ipc=host ^
        --name %CONTAINER_NAME% ^
        -v "%DOCKER_PATH%:/workspace" ^
        -w /workspace ^
        -p 8895:8895 ^
        -it %IMAGE% bash
)

endlocal
