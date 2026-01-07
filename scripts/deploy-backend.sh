#!/bin/bash
# Deploy full backend microservices to Google Cloud
# This creates the complete enterprise architecture

set -e

# Load configuration
if [ -f .gcp-config ]; then
    source .gcp-config
else
    echo "❌ GCP configuration not found. Run ./scripts/gcp-setup.sh first"
    exit 1
fi

echo "🚀 Deploying Backend Microservices to Google Cloud"
echo "================================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Create Cloud SQL instance
echo "🗄️ Creating Cloud SQL PostgreSQL instance..."
gcloud sql instances create enterprise-saas-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=$REGION \
    --storage-type=SSD \
    --storage-size=20GB \
    --storage-auto-increase \
    --backup-start-time=03:00 \
    --enable-bin-log \
    --maintenance-window-day=SUN \
    --maintenance-window-hour=04 \
    --maintenance-release-channel=production || echo "Database instance may already exist"

# Create database and user
echo "👤 Setting up database and user..."
gcloud sql databases create enterprise_saas --instance=enterprise-saas-db || echo "Database may already exist"
gcloud sql users create saas_user --instance=enterprise-saas-db --password=SecurePassword123! || echo "User may already exist"

# Get database connection info
DB_CONNECTION_NAME=$(gcloud sql instances describe enterprise-saas-db --format="value(connectionName)")
DB_IP=$(gcloud sql instances describe enterprise-saas-db --format="value(ipAddresses[0].ipAddress)")

echo "📊 Database Info:"
echo "   Connection Name: $DB_CONNECTION_NAME"
echo "   IP Address: $DB_IP"

# Create Redis instance
echo "🔴 Creating Redis instance..."
gcloud redis instances create enterprise-saas-redis \
    --size=1 \
    --region=$REGION \
    --redis-version=redis_6_x \
    --network=default || echo "Redis instance may already exist"

# Get Redis connection info
REDIS_IP=$(gcloud redis instances describe enterprise-saas-redis --region=$REGION --format="value(host)")
REDIS_PORT=$(gcloud redis instances describe enterprise-saas-redis --region=$REGION --format="value(port)")

echo "🔴 Redis Info:"
echo "   IP Address: $REDIS_IP"
echo "   Port: $REDIS_PORT"

# Create storage buckets
echo "🪣 Creating Cloud Storage buckets..."
gsutil mb gs://$PROJECT_ID-data-storage || echo "Bucket may already exist"
gsutil mb gs://$PROJECT_ID-ml-models || echo "Bucket may already exist"
gsutil mb gs://$PROJECT_ID-static-assets || echo "Bucket may already exist"

# Set bucket permissions
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectAdmin gs://$PROJECT_ID-data-storage
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectAdmin gs://$PROJECT_ID-ml-models
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectAdmin gs://$PROJECT_ID-static-assets

# Create secrets in Secret Manager
echo "🔐 Creating secrets..."
echo -n "postgresql://saas_user:SecurePassword123!@$DB_IP:5432/enterprise_saas" | \
    gcloud secrets create database-url --data-file=- || echo "Secret may already exist"

echo -n "redis://$REDIS_IP:$REDIS_PORT" | \
    gcloud secrets create redis-url --data-file=- || echo "Secret may already exist"

echo -n "super-secret-jwt-key-$(date +%s)" | \
    gcloud secrets create jwt-secret-key --data-file=- || echo "Secret may already exist"

# Grant secret access to service account
gcloud secrets add-iam-policy-binding database-url \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding redis-url \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding jwt-secret-key \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

# Deploy API Gateway
echo "🌐 Deploying API Gateway..."
gcloud run deploy api-gateway \
    --source . \
    --dockerfile Dockerfile.gateway \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8000 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
    --set-secrets="DATABASE_URL=database-url:latest,REDIS_URL=redis-url:latest,JWT_SECRET_KEY=jwt-secret-key:latest"

# Deploy User Service
echo "👤 Deploying User Service..."
gcloud run deploy user-service \
    --source . \
    --dockerfile Dockerfile.user \
    --platform managed \
    --region $REGION \
    --no-allow-unauthenticated \
    --port 8001 \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 5 \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
    --set-secrets="DATABASE_URL=database-url:latest,REDIS_URL=redis-url:latest,JWT_SECRET_KEY=jwt-secret-key:latest"

# Deploy Tenant Service
echo "🏢 Deploying Tenant Service..."
gcloud run deploy tenant-service \
    --source . \
    --dockerfile Dockerfile.tenant \
    --platform managed \
    --region $REGION \
    --no-allow-unauthenticated \
    --port 8002 \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 5 \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
    --set-secrets="DATABASE_URL=database-url:latest,REDIS_URL=redis-url:latest,JWT_SECRET_KEY=jwt-secret-key:latest"

# Deploy Data Service
echo "📊 Deploying Data Service..."
gcloud run deploy data-service \
    --source . \
    --dockerfile Dockerfile.data \
    --platform managed \
    --region $REGION \
    --no-allow-unauthenticated \
    --port 8003 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10 \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,DATA_STORAGE_BUCKET=$PROJECT_ID-data-storage" \
    --set-secrets="DATABASE_URL=database-url:latest,REDIS_URL=redis-url:latest"

# Deploy ML Service
echo "🤖 Deploying ML Service..."
gcloud run deploy ml-service \
    --source . \
    --dockerfile Dockerfile.ml \
    --platform managed \
    --region $REGION \
    --no-allow-unauthenticated \
    --port 8004 \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 5 \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,ML_MODELS_BUCKET=$PROJECT_ID-ml-models" \
    --set-secrets="DATABASE_URL=database-url:latest,REDIS_URL=redis-url:latest"

# Get service URLs
API_GATEWAY_URL=$(gcloud run services describe api-gateway --region=$REGION --format="value(status.url)")
USER_SERVICE_URL=$(gcloud run services describe user-service --region=$REGION --format="value(status.url)")
TENANT_SERVICE_URL=$(gcloud run services describe tenant-service --region=$REGION --format="value(status.url)")
DATA_SERVICE_URL=$(gcloud run services describe data-service --region=$REGION --format="value(status.url)")
ML_SERVICE_URL=$(gcloud run services describe ml-service --region=$REGION --format="value(status.url)")

# Update Streamlit app with backend URLs
echo "🔄 Updating Streamlit app with backend URLs..."
gcloud run deploy enterprise-saas-streamlit \
    --source . \
    --dockerfile Dockerfile.streamlit \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars="API_BASE_URL=$API_GATEWAY_URL,DATA_SERVICE_URL=$DATA_SERVICE_URL,ML_SERVICE_URL=$ML_SERVICE_URL"

STREAMLIT_URL=$(gcloud run services describe enterprise-saas-streamlit --region=$REGION --format="value(status.url)")

echo ""
echo "🎉 Full Deployment Complete!"
echo "============================"
echo ""
echo "🌐 Service URLs:"
echo "   Frontend (Streamlit): $STREAMLIT_URL"
echo "   API Gateway:          $API_GATEWAY_URL"
echo "   User Service:         $USER_SERVICE_URL"
echo "   Tenant Service:       $TENANT_SERVICE_URL"
echo "   Data Service:         $DATA_SERVICE_URL"
echo "   ML Service:           $ML_SERVICE_URL"
echo ""
echo "🗄️ Infrastructure:"
echo "   Database:             $DB_CONNECTION_NAME"
echo "   Redis:                $REDIS_IP:$REDIS_PORT"
echo "   Storage:              gs://$PROJECT_ID-data-storage"
echo "   ML Models:            gs://$PROJECT_ID-ml-models"
echo ""
echo "💰 Estimated Monthly Cost: $50-150 (depending on usage)"
echo ""
echo "🎯 Next Steps:"
echo "   1. Visit $STREAMLIT_URL to see your platform"
echo "   2. Test API endpoints at $API_GATEWAY_URL/docs"
echo "   3. Monitor costs at: https://console.cloud.google.com/billing"
echo "   4. Set up monitoring: ./scripts/setup-monitoring.sh"
echo ""
echo "🏆 Congratulations! You now have a production-ready"
echo "    enterprise SaaS platform running on Google Cloud!"

# Save all deployment info
cat > .deployment-info << EOF
# Enterprise SaaS Platform - Google Cloud Deployment
DEPLOYMENT_DATE=$(date)
PROJECT_ID=$PROJECT_ID
REGION=$REGION

# Service URLs
STREAMLIT_URL=$STREAMLIT_URL
API_GATEWAY_URL=$API_GATEWAY_URL
USER_SERVICE_URL=$USER_SERVICE_URL
TENANT_SERVICE_URL=$TENANT_SERVICE_URL
DATA_SERVICE_URL=$DATA_SERVICE_URL
ML_SERVICE_URL=$ML_SERVICE_URL

# Infrastructure
DB_CONNECTION_NAME=$DB_CONNECTION_NAME
DB_IP=$DB_IP
REDIS_IP=$REDIS_IP
REDIS_PORT=$REDIS_PORT
DATA_BUCKET=$PROJECT_ID-data-storage
ML_BUCKET=$PROJECT_ID-ml-models

# Service Account
SERVICE_ACCOUNT=$SERVICE_ACCOUNT
EOF

echo "✅ All deployment information saved to .deployment-info"