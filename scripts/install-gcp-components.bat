@echo off
REM Install essential GCP components for Python Enterprise SaaS Platform

echo 🚀 Installing Google Cloud Components for Python Platform
echo =========================================================

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

REM Update gcloud components
echo 🔄 Updating gcloud components...
gcloud components update --quiet

REM Install essential components for Python platform
echo 📦 Installing essential components...

echo   • Cloud Run (for microservices deployment)
gcloud components install cloud-run-proxy --quiet

echo   • Cloud SQL Proxy (for secure database connections)
gcloud components install cloud-sql-proxy --quiet

echo   • kubectl (for Kubernetes management)
gcloud components install kubectl --quiet

echo   • Alpha/Beta commands (for latest ML features)
gcloud components install alpha beta --quiet

echo   • Docker credential helper
gcloud components install docker-credential-gcr --quiet

REM Configure Docker for GCP
echo 🐳 Configuring Docker for Google Cloud...
gcloud auth configure-docker --quiet

echo ✅ Installation complete!
echo.
echo 📋 Installed Components:
gcloud components list --filter="state.name=Installed" --format="table(id,size.size(units=MB))"
echo.
echo 🎯 Next Steps:
echo    1. Run: scripts\gcp-setup.bat
echo    2. Then: scripts\deploy-streamlit.bat
echo.
echo 🚀 You're ready to deploy your enterprise platform!
pause