@echo off
REM Google Cloud Platform Setup Script for Windows
REM Run this script to set up your GCP environment

echo 🚀 Setting up Google Cloud Platform for Enterprise SaaS Platform
echo ================================================================

REM Check if gcloud is installed
where gcloud >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Google Cloud SDK not found. Please install it first:
    echo    Visit: https://cloud.google.com/sdk/docs/install-windows
    echo    Download and run GoogleCloudSDKInstaller.exe
    pause
    exit /b 1
)

echo ✅ Google Cloud SDK found

REM Set project variables
set PROJECT_ID=enterprise-saas-%RANDOM%
set REGION=us-central1
set ZONE=us-central1-a

echo 📝 Project Configuration:
echo    Project ID: %PROJECT_ID%
echo    Region: %REGION%
echo    Zone: %ZONE%

REM Authenticate with Google Cloud
echo 🔐 Authenticating with Google Cloud...
gcloud auth login

REM Create new project
echo 🏗️ Creating new project...
gcloud projects create %PROJECT_ID% --name="Enterprise SaaS Platform"

REM Set current project
gcloud config set project %PROJECT_ID%

REM Set default region and zone
gcloud config set compute/region %REGION%
gcloud config set compute/zone %ZONE%

echo 💳 Setting up billing...
echo ⚠️  IMPORTANT: You need to enable billing for this project
echo    Visit: https://console.cloud.google.com/billing/linkedaccount?project=%PROJECT_ID%
echo    Link your billing account to continue
pause

REM Enable required APIs
echo 🔧 Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable secretmanager.googleapis.com

REM Create service account
echo 👤 Creating service account...
gcloud iam service-accounts create enterprise-saas-app --display-name="Enterprise SaaS Application" --description="Service account for Enterprise SaaS Platform"

REM Grant necessary permissions
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:enterprise-saas-app@%PROJECT_ID%.iam.gserviceaccount.com" --role="roles/cloudsql.client"
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:enterprise-saas-app@%PROJECT_ID%.iam.gserviceaccount.com" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:enterprise-saas-app@%PROJECT_ID%.iam.gserviceaccount.com" --role="roles/run.invoker"

REM Save configuration
echo 💾 Saving configuration...
echo PROJECT_ID=%PROJECT_ID% > .gcp-config.bat
echo REGION=%REGION% >> .gcp-config.bat
echo ZONE=%ZONE% >> .gcp-config.bat
echo SERVICE_ACCOUNT=enterprise-saas-app@%PROJECT_ID%.iam.gserviceaccount.com >> .gcp-config.bat

echo ✅ Google Cloud setup complete!
echo.
echo 📋 Next Steps:
echo    1. Run: scripts\deploy-streamlit.bat
echo    2. Then: scripts\deploy-backend.bat
echo.
echo 🎯 Your project is ready for deployment!
pause