@echo off
REM Docker deployment script for Windows

echo 🐳 AstralytiQ Backend Docker Deployment Script
echo ==============================================

set IMAGE_NAME=astralytiq-backend
set IMAGE_TAG=latest
set CONTAINER_PORT=8080
set HOST_PORT=8080

:menu
echo.
echo Select deployment option:
echo 1) Build Docker image only
echo 2) Run locally with Docker
echo 3) Deploy to Google Cloud Run (FREE)
echo 4) Build and run locally
echo 5) Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto run_local
if "%choice%"=="3" goto deploy_gcp
if "%choice%"=="4" goto build_and_run
if "%choice%"=="5" goto exit
goto invalid

:build
echo 📋 Building Docker image...
docker build -f Dockerfile.backend -t %IMAGE_NAME%:%IMAGE_TAG% .
if %errorlevel% neq 0 (
    echo ❌ Failed to build Docker image
    pause
    goto menu
)
echo ✅ Docker image built successfully
pause
goto menu

:run_local
call :build
echo 📋 Running container locally...

REM Stop existing container if running
docker stop %IMAGE_NAME% 2>nul
docker rm %IMAGE_NAME% 2>nul

REM Run new container
docker run -d ^
    --name %IMAGE_NAME% ^
    -p %HOST_PORT%:%CONTAINER_PORT% ^
    -e ENVIRONMENT=production ^
    -e JWT_SECRET=AstralytiQ-Production-JWT-Secret-Key-2025-Campus-Placement ^
    -e CORS_ORIGINS=https://astralytiq-platform.streamlit.app,http://localhost:8501 ^
    %IMAGE_NAME%:%IMAGE_TAG%

echo ✅ Container started on http://localhost:%HOST_PORT%
echo 📋 API Documentation: http://localhost:%HOST_PORT%/docs
echo 📋 Health Check: http://localhost:%HOST_PORT%/health
pause
goto menu

:deploy_gcp
call :build
echo 📋 Deploying to Google Cloud Run...

REM Check if gcloud is installed
gcloud version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ gcloud CLI not found. Please install Google Cloud SDK.
    pause
    goto menu
)

REM Build and submit to Cloud Build
gcloud builds submit --config cloudbuild.yaml .

echo ✅ Deployed to Google Cloud Run
echo 📋 Getting service URL...

for /f "tokens=*" %%i in ('gcloud run services describe astralytiq-backend --region=us-central1 --format="value(status.url)"') do set SERVICE_URL=%%i
echo ✅ Service URL: %SERVICE_URL%
echo 📋 API Documentation: %SERVICE_URL%/docs
pause
goto menu

:build_and_run
call :build
call :run_local
goto menu

:invalid
echo ❌ Invalid option. Please try again.
pause
goto menu

:exit
echo ✅ Goodbye!
pause
exit /b 0