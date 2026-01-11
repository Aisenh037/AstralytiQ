#!/bin/bash

# AstralytiQ GCP Deployment Script
# -------------------------------
# This script automates the deployment of the AstralytiQ backend to Google Cloud Run.

# 1. Configuration - CHANGE THESE
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="astralytiq-backend"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Starting deployment for project: $PROJECT_ID"

# 2. Enable Required APIs
echo "🔗 Enabling required APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# 3. Build the Docker Image using Cloud Build
echo "📦 Building Docker image in the cloud..."
gcloud builds submit --tag $IMAGE_NAME -f backend/Dockerfile.cloud .

# 4. Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars ENVIRONMENT=production,JWT_SECRET=campus-placement-secret-2026

# 5. Get the Service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo "------------------------------------------------------------"
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "------------------------------------------------------------"
echo "🔗 Backend URL: $SERVICE_URL"
echo ""
echo "👉 NEXT STEPS:"
echo "1. Go to your Streamlit Cloud dashboard"
echo "2. In Settings > Secrets, add the following:"
echo ""
echo "API_BASE_URL = \"$SERVICE_URL\""
echo ""
echo "3. Refresh your Streamlit app. It should now show 'Backend Connected'!"
echo "------------------------------------------------------------"
