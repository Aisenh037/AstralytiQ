#!/bin/bash
# Deploy Streamlit app to Google Cloud Run
# This is the fastest way to get your platform live!

set -e

# Load configuration
if [ -f .gcp-config ]; then
    source .gcp-config
else
    echo "❌ GCP configuration not found. Run ./scripts/gcp-setup.sh first"
    exit 1
fi

echo "🚀 Deploying Streamlit App to Google Cloud Run"
echo "=============================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Create Dockerfile for Streamlit
echo "📦 Creating Streamlit Dockerfile..."
cat > Dockerfile.streamlit << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit app and config
COPY streamlit_app.py .
COPY .streamlit/ .streamlit/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
EOF

# Create requirements.txt for Streamlit
echo "📋 Creating Streamlit requirements..."
cat > requirements.txt << 'EOF'
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
requests>=2.31.0
networkx>=3.1
numpy>=1.24.0
scikit-learn>=1.3.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
EOF

# Create .streamlit config if it doesn't exist
if [ ! -d ".streamlit" ]; then
    echo "⚙️ Creating Streamlit configuration..."
    mkdir -p .streamlit
    cat > .streamlit/config.toml << 'EOF'
[server]
port = 8080
address = "0.0.0.0"
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
EOF
fi

# Build and deploy to Cloud Run
echo "🏗️ Building and deploying to Cloud Run..."
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
    --set-env-vars="DEMO_MODE=true" \
    --service-account=$SERVICE_ACCOUNT

# Get the service URL
SERVICE_URL=$(gcloud run services describe enterprise-saas-streamlit --region=$REGION --format="value(status.url)")

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo "🌐 Your Streamlit app is live at:"
echo "   $SERVICE_URL"
echo ""
echo "📱 Features available:"
echo "   • 📊 Interactive Dashboard"
echo "   • 📤 Data Upload Interface"
echo "   • 🔄 Data Transformations"
echo "   • 🔗 Data Lineage Visualization"
echo "   • 🤖 ML Model Training Interface"
echo "   • ⚙️ System Status Monitoring"
echo ""
echo "🎯 Next Steps:"
echo "   1. Visit your app and explore the features"
echo "   2. Share the URL with stakeholders for demo"
echo "   3. Run ./scripts/deploy-backend.sh for full microservices"
echo ""
echo "💡 Pro Tip: This deployment costs ~$0.10/day when idle!"

# Save deployment info
cat > .deployment-info << EOF
STREAMLIT_URL=$SERVICE_URL
DEPLOYMENT_DATE=$(date)
PROJECT_ID=$PROJECT_ID
REGION=$REGION
EOF

echo "✅ Deployment information saved to .deployment-info"