#!/bin/bash
# Google Cloud Platform Setup Script
# Run this script to set up your GCP environment

set -e

echo "🚀 Setting up Google Cloud Platform for Enterprise SaaS Platform"
echo "================================================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK not found. Installing..."
    
    # Install Google Cloud SDK
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        curl https://sdk.cloud.google.com | bash
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl https://sdk.cloud.google.com | bash
    else
        echo "Please install Google Cloud SDK manually: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Restart shell to load gcloud
    exec -l $SHELL
fi

echo "✅ Google Cloud SDK found"

# Set project variables
PROJECT_ID="enterprise-saas-$(date +%s)"
REGION="us-central1"
ZONE="us-central1-a"

echo "📝 Project Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Zone: $ZONE"

# Authenticate with Google Cloud
echo "🔐 Authenticating with Google Cloud..."
gcloud auth login

# Create new project
echo "🏗️ Creating new project..."
gcloud projects create $PROJECT_ID --name="Enterprise SaaS Platform"

# Set current project
gcloud config set project $PROJECT_ID

# Set default region and zone
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

echo "💳 Setting up billing..."
echo "⚠️  IMPORTANT: You need to enable billing for this project"
echo "   Visit: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
echo "   Link your billing account to continue"

read -p "Press Enter after you've enabled billing..."

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create service account
echo "👤 Creating service account..."
gcloud iam service-accounts create enterprise-saas-app \
    --display-name="Enterprise SaaS Application" \
    --description="Service account for Enterprise SaaS Platform"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:enterprise-saas-app@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:enterprise-saas-app@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:enterprise-saas-app@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker"

# Save configuration
echo "💾 Saving configuration..."
cat > .gcp-config << EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
ZONE=$ZONE
SERVICE_ACCOUNT=enterprise-saas-app@$PROJECT_ID.iam.gserviceaccount.com
EOF

echo "✅ Google Cloud setup complete!"
echo ""
echo "📋 Next Steps:"
echo "   1. Run: source .gcp-config"
echo "   2. Run: ./scripts/deploy-streamlit.sh"
echo ""
echo "🎯 Your project is ready for deployment!"