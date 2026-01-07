@echo off
REM Deploy Streamlit app to Google Cloud Run
REM This is the fastest way to get your platform live!

REM Load configuration
if exist .gcp-config.bat (
    call .gcp-config.bat
) else (
    echo ❌ GCP configuration not found. Run scripts\gcp-setup.bat first
    pause
    exit /b 1
)

echo 🚀 Deploying Streamlit App to Google Cloud Run
echo ==============================================
echo Project: %PROJECT_ID%
echo Region: %REGION%

REM Create Dockerfile for Streamlit
echo 📦 Creating Streamlit Dockerfile...
(
echo FROM python:3.11-slim
echo.
echo WORKDIR /app
echo.
echo # Install system dependencies
echo RUN apt-get update ^&^& apt-get install -y \
echo     gcc \
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

REM Create requirements.txt for Streamlit
echo 📋 Creating Streamlit requirements...
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

REM Build and deploy to Cloud Run
echo 🏗️ Building and deploying to Cloud Run...
gcloud run deploy enterprise-saas-streamlit --source . --dockerfile Dockerfile.streamlit --platform managed --region %REGION% --allow-unauthenticated --port 8080 --memory 1Gi --cpu 1 --max-instances 10 --set-env-vars="DEMO_MODE=true" --service-account=%SERVICE_ACCOUNT%

REM Get the service URL
for /f "tokens=*" %%i in ('gcloud run services describe enterprise-saas-streamlit --region=%REGION% --format="value(status.url)"') do set SERVICE_URL=%%i

echo.
echo 🎉 Deployment Complete!
echo ======================
echo 🌐 Your Streamlit app is live at:
echo    %SERVICE_URL%
echo.
echo 📱 Features available:
echo    • 📊 Interactive Dashboard
echo    • 📤 Data Upload Interface
echo    • 🔄 Data Transformations
echo    • 🔗 Data Lineage Visualization
echo    • 🤖 ML Model Training Interface
echo    • ⚙️ System Status Monitoring
echo.
echo 🎯 Next Steps:
echo    1. Visit your app and explore the features
echo    2. Share the URL with stakeholders for demo
echo    3. Run scripts\deploy-backend.bat for full microservices
echo.
echo 💡 Pro Tip: This deployment costs ~$0.10/day when idle!

REM Save deployment info
(
echo STREAMLIT_URL=%SERVICE_URL%
echo DEPLOYMENT_DATE=%DATE% %TIME%
echo PROJECT_ID=%PROJECT_ID%
echo REGION=%REGION%
) > .deployment-info.bat

echo ✅ Deployment information saved to .deployment-info.bat
pause