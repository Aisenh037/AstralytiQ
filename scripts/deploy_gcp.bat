@echo off
setlocal enabledelayedexpansion

:: AstralytiQ GCP Deployment Script (Windows) - v2
:: ----------------------------------------------

echo 🚀 Starting Windows deployment...

:: 1. Verify Project
for /f "tokens=*" %%i in ('gcloud config get-value project') do set CURRENT_PROJECT=%%i

echo 🎯 Current gcloud project: %CURRENT_PROJECT%

if not "%CURRENT_PROJECT%"=="astralytiq-483615" (
    echo ⚠️ WARNING: Your current project is NOT astralytiq-483615.
    echo 🔄 Setting project to astralytiq-483615...
    call gcloud config set project astralytiq-483615
    if %ERRORLEVEL% neq 0 (
        echo ❌ FAILED to set project. Please run: gcloud auth login
        pause
        exit /b 1
    )
    set PROJECT_ID=astralytiq-483615
) else (
    set PROJECT_ID=%CURRENT_PROJECT%
)

set REGION=us-central1
set SERVICE_NAME=astralytiq-backend

:: 2. Enable Required APIs (and Check for Billing)
echo 🔗 Enabling required APIs...
echo (If this fails here, check if billing is linked to %PROJECT_ID%)
call gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ ERROR: Failed to enable APIs. 
    echo 👉 Please link your billing account here: https://console.cloud.google.com/billing/projects
    echo.
    pause
    exit /b %ERRORLEVEL%
)

:: 3. Build & Deploy using Cloud Build
echo 📦 Building and deploying in the cloud...
echo ⏳ This WILL take 5-10 minutes (installing Prophet dependencies)...
call gcloud builds submit --config cloudbuild.forecasting.yaml .
if %ERRORLEVEL% neq 0 (
    echo ❌ ERROR: Cloud Build failed. Check the logs above.
    pause
    exit /b %ERRORLEVEL%
)

:: 4. Get the Service URL
echo 🔍 Fetching your new backend URL...
for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format "value(status.url)"') do set SERVICE_URL=%%i

echo.
echo ------------------------------------------------------------
echo ✅ DEPLOYMENT SUCCESSFUL!
echo ------------------------------------------------------------
echo 🔗 Backend URL: %SERVICE_URL%
echo.
echo 👉 NEXT STEPS:
echo 1. Go to your Streamlit Cloud dashboard
echo 2. In Settings ^> Secrets, add the following line:
echo.
echo API_BASE_URL = "%SERVICE_URL%"
echo.
echo 3. Save and refresh your Streamlit app!
echo ------------------------------------------------------------

pause
