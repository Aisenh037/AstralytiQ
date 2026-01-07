@echo off
REM Deploy Enterprise SaaS Platform to existing astralytiq project
REM Fast track deployment using your existing GCP project

echo 🚀 Deploying to Google Cloud Project: astralytiq
echo ===============================================

REM Set your project configuration
set PROJECT_ID=astralytiq
set PROJECT_NUMBER=855298666240
set REGION=us-central1
set ZONE=us-central1-a

echo 📝 Using existing project:
echo    Project ID: %PROJECT_ID%
echo    Project Number: %PROJECT_NUMBER%
echo    Region: %REGION%

REM Set current project
echo 🔧 Configuring gcloud for your project...
gcloud config set project %PROJECT_ID%
gcloud config set compute/region %REGION%
gcloud config set compute/zone %ZONE%

REM Check if billing is enabled
echo 💳 Checking billing status...
gcloud billing projects describe %PROJECT_ID% >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  IMPORTANT: Billing may not be enabled for this project
    echo    Visit: https://console.cloud.google.com/billing/linkedaccount?project=%PROJECT_ID%
    echo    Enable billing to continue with deployment
    pause
)

REM Enable required APIs
echo 🔧 Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable secretmanager.googleapis.com

REM Create service account if it doesn't exist
echo 👤 Setting up service account...
gcloud iam service-accounts create astralytiq-app --display-name="AstralytiQ Enterprise SaaS" --description="Service account for AstralytiQ platform" 2>nul || echo "Service account may already exist"

REM Set service account variable
set SERVICE_ACCOUNT=astralytiq-app@%PROJECT_ID%.iam.gserviceaccount.com

REM Grant necessary permissions
echo 🔐 Configuring permissions...
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SERVICE_ACCOUNT%" --role="roles/cloudsql.client"
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SERVICE_ACCOUNT%" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SERVICE_ACCOUNT%" --role="roles/run.invoker"
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SERVICE_ACCOUNT%" --role="roles/aiplatform.user"

REM Create Dockerfile for Streamlit if it doesn't exist
if not exist "Dockerfile.streamlit" (
    echo 📦 Creating Streamlit Dockerfile...
    (
    echo FROM python:3.11-slim
    echo.
    echo WORKDIR /app
    echo.
    echo # Install system dependencies
    echo RUN apt-get update ^&^& apt-get install -y \
    echo     gcc curl \
    echo     ^&^& rm -rf /var/lib/apt/lists/*
    echo.
    echo # Copy requirements
    echo COPY requirements.txt .
    echo RUN pip install --no-cache-dir -r requirements.txt
    echo.
    echo # Copy Streamlit app and config
    echo COPY streamlit_app.py .
    echo COPY .streamlit/ .streamlit/
    echo.
    echo # Expose port
    echo EXPOSE 8080
    echo.
    echo # Health check
    echo HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health
    echo.
    echo # Run Streamlit
    echo CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
    ) > Dockerfile.streamlit
)

REM Create requirements.txt for Streamlit if it doesn't exist
if not exist "requirements.txt" (
    echo 📋 Creating requirements.txt...
    (
    echo streamlit^>=1.28.0
    echo pandas^>=1.5.0
    echo plotly^>=5.15.0
    echo requests^>=2.31.0
    echo networkx^>=3.1
    echo numpy^>=1.24.0
    echo scikit-learn^>=1.3.0
    echo fastapi^>=0.104.0
    echo uvicorn^>=0.24.0
    echo pydantic^>=2.5.0
    ) > requirements.txt
)

REM Create .streamlit config if it doesn't exist
if not exist ".streamlit" (
    echo ⚙️ Creating Streamlit configuration...
    mkdir .streamlit
    (
    echo [server]
    echo port = 8080
    echo address = "0.0.0.0"
    echo maxUploadSize = 200
    echo enableCORS = false
    echo enableXsrfProtection = false
    echo.
    echo [theme]
    echo primaryColor = "#1f77b4"
    echo backgroundColor = "#ffffff"
    echo secondaryBackgroundColor = "#f0f2f6"
    echo textColor = "#262730"
    echo.
    echo [browser]
    echo gatherUsageStats = false
    ) > .streamlit\config.toml
)

REM Deploy to Cloud Run
echo 🏗️ Deploying AstralytiQ to Google Cloud Run...
echo    This may take 3-5 minutes...

gcloud run deploy astralytiq-platform ^
    --source . ^
    --dockerfile Dockerfile.streamlit ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --port 8080 ^
    --memory 2Gi ^
    --cpu 2 ^
    --max-instances 10 ^
    --min-instances 0 ^
    --service-account=%SERVICE_ACCOUNT% ^
    --set-env-vars="DEMO_MODE=true,PROJECT_ID=%PROJECT_ID%"

REM Get the service URL
for /f "tokens=*" %%i in ('gcloud run services describe astralytiq-platform --region=%REGION% --format="value(status.url)"') do set SERVICE_URL=%%i

echo.
echo 🎉 AstralytiQ Platform Deployed Successfully!
echo ============================================
echo.
echo 🌐 Your platform is live at:
echo    %SERVICE_URL%
echo.
echo 📊 Project Details:
echo    • Project ID: %PROJECT_ID%
echo    • Project Number: %PROJECT_NUMBER%
echo    • Region: %REGION%
echo    • Service Account: %SERVICE_ACCOUNT%
echo.
echo 🎯 Platform Features:
echo    ✅ Interactive Analytics Dashboard
echo    ✅ Data Upload ^& Processing
echo    ✅ ML Model Training Interface
echo    ✅ Data Transformation Pipeline
echo    ✅ Data Lineage Visualization
echo    ✅ System Monitoring
echo.
echo 💰 Cost Estimate: ~$0.10-2.00/day (depending on usage)
echo.
echo 🚀 Next Steps:
echo    1. Visit your platform: %SERVICE_URL%
echo    2. Test all features and take screenshots
echo    3. Share URL for demos and portfolio
echo    4. Monitor usage: https://console.cloud.google.com/run?project=%PROJECT_ID%
echo.
echo 🏆 Congratulations! You now have a production-ready
echo    enterprise SaaS platform running on Google Cloud!

REM Save deployment info
(
echo # AstralytiQ Platform - Google Cloud Deployment
echo DEPLOYMENT_DATE=%DATE% %TIME%
echo PROJECT_ID=%PROJECT_ID%
echo PROJECT_NUMBER=%PROJECT_NUMBER%
echo REGION=%REGION%
echo SERVICE_URL=%SERVICE_URL%
echo SERVICE_ACCOUNT=%SERVICE_ACCOUNT%
echo.
echo # Quick Access Links
echo PLATFORM_URL=%SERVICE_URL%
echo CONSOLE_URL=https://console.cloud.google.com/run?project=%PROJECT_ID%
echo BILLING_URL=https://console.cloud.google.com/billing?project=%PROJECT_ID%
echo LOGS_URL=https://console.cloud.google.com/logs/query?project=%PROJECT_ID%
) > .astralytiq-deployment.txt

echo ✅ Deployment info saved to .astralytiq-deployment.txt
echo.
echo 🎊 Ready to showcase your enterprise platform!
pause