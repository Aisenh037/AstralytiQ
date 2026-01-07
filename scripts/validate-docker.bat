@echo off
REM Docker Setup Validation Script (Windows)
setlocal enabledelayedexpansion

echo [INFO] Validating Docker containerization setup...

REM Check if all Dockerfiles exist
set "missing_files="
set "dockerfiles=Dockerfile.gateway Dockerfile.user Dockerfile.tenant Dockerfile.data Dockerfile.ml"

for %%f in (%dockerfiles%) do (
    if exist "%%f" (
        echo [SUCCESS] %%f exists
    ) else (
        echo [ERROR] %%f is missing
        set "missing_files=!missing_files! %%f"
    )
)

REM Check if docker-compose files exist
set "compose_files=docker-compose.yml docker-compose.dev.yml docker-compose.prod.yml"

for %%f in (%compose_files%) do (
    if exist "%%f" (
        echo [SUCCESS] %%f exists
    ) else (
        echo [ERROR] %%f is missing
        set "missing_files=!missing_files! %%f"
    )
)

REM Check if requirements file exists
if exist "requirements-enterprise.txt" (
    echo [SUCCESS] requirements-enterprise.txt exists
) else (
    echo [ERROR] requirements-enterprise.txt is missing
    set "missing_files=!missing_files! requirements-enterprise.txt"
)

REM Check if build scripts exist
if exist "scripts\docker-build.sh" (
    echo [SUCCESS] scripts\docker-build.sh exists
) else (
    echo [ERROR] scripts\docker-build.sh is missing
    set "missing_files=!missing_files! scripts\docker-build.sh"
)

if exist "scripts\docker-build.bat" (
    echo [SUCCESS] scripts\docker-build.bat exists
) else (
    echo [ERROR] scripts\docker-build.bat is missing
    set "missing_files=!missing_files! scripts\docker-build.bat"
)

REM Check if .dockerignore exists
if exist ".dockerignore" (
    echo [SUCCESS] .dockerignore exists
) else (
    echo [WARNING] .dockerignore is missing (recommended for build optimization)
)

REM Check if nginx.conf exists
if exist "nginx.conf" (
    echo [SUCCESS] nginx.conf exists
) else (
    echo [WARNING] nginx.conf is missing (needed for production load balancing)
)

REM Validate docker-compose.yml syntax
docker-compose --version >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Validating docker-compose.yml syntax...
    docker-compose config >nul 2>&1
    if not errorlevel 1 (
        echo [SUCCESS] docker-compose.yml syntax is valid
    ) else (
        echo [ERROR] docker-compose.yml has syntax errors
    )
) else (
    echo [WARNING] docker-compose not available for syntax validation
)

REM Summary
echo.
echo [INFO] Validation Summary:

if "%missing_files%"=="" (
    echo [SUCCESS] All required Docker files are present!
    echo [INFO] You can now:
    echo   1. Start Docker Desktop (if not running)
    echo   2. Run: scripts\docker-build.bat build
    echo   3. Run: scripts\docker-build.bat start-dev
) else (
    echo [ERROR] Some required files are missing:
    for %%f in (%missing_files%) do echo   - %%f
)

echo.
echo [INFO] Docker containerization setup validation complete!

endlocal