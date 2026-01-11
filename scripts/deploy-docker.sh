#!/bin/bash
# Docker deployment script for all cloud providers

set -e

echo "🐳 AstralytiQ Backend Docker Deployment Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="astralytiq-backend"
IMAGE_TAG="latest"
CONTAINER_PORT=8080
HOST_PORT=8080

print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to build Docker image
build_image() {
    print_step "Building Docker image..."
    
    if docker build -f Dockerfile.backend -t $IMAGE_NAME:$IMAGE_TAG .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run container locally
run_local() {
    print_step "Running container locally..."
    
    # Stop existing container if running
    docker stop $IMAGE_NAME 2>/dev/null || true
    docker rm $IMAGE_NAME 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name $IMAGE_NAME \
        -p $HOST_PORT:$CONTAINER_PORT \
        -e ENVIRONMENT=production \
        -e JWT_SECRET=AstralytiQ-Production-JWT-Secret-Key-2025-Campus-Placement \
        -e CORS_ORIGINS=https://astralytiq-platform.streamlit.app,http://localhost:8501 \
        $IMAGE_NAME:$IMAGE_TAG
    
    print_success "Container started on http://localhost:$HOST_PORT"
    print_step "API Documentation: http://localhost:$HOST_PORT/docs"
    print_step "Health Check: http://localhost:$HOST_PORT/health"
}

# Function to deploy to Google Cloud Run
deploy_gcp() {
    print_step "Deploying to Google Cloud Run..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Please install Google Cloud SDK."
        return 1
    fi
    
    # Get project ID
    PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$PROJECT_ID" ]; then
        print_error "No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
        return 1
    fi
    
    print_step "Using GCP Project: $PROJECT_ID"
    
    # Build and submit to Cloud Build
    gcloud builds submit --config cloudbuild.yaml .
    
    print_success "Deployed to Google Cloud Run"
    print_step "Getting service URL..."
    
    SERVICE_URL=$(gcloud run services describe astralytiq-backend --region=us-central1 --format="value(status.url)")
    print_success "Service URL: $SERVICE_URL"
    print_step "API Documentation: $SERVICE_URL/docs"
}

# Function to deploy to AWS ECS Fargate
deploy_aws() {
    print_step "Deploying to AWS ECS Fargate..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Please install AWS CLI."
        return 1
    fi
    
    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION="us-east-1"
    
    print_step "Using AWS Account: $ACCOUNT_ID"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names $IMAGE_NAME --region $REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $IMAGE_NAME --region $REGION
    
    # Get ECR login token
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
    
    # Tag and push image
    docker tag $IMAGE_NAME:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:$IMAGE_TAG
    docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:$IMAGE_TAG
    
    print_success "Image pushed to ECR"
    print_warning "Please configure ECS cluster and service manually or use AWS CDK/CloudFormation"
}

# Function to deploy to Azure Container Instances
deploy_azure() {
    print_step "Deploying to Azure Container Instances..."
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI not found. Please install Azure CLI."
        return 1
    fi
    
    # Check if logged in
    if ! az account show &> /dev/null; then
        print_error "Not logged into Azure. Run: az login"
        return 1
    fi
    
    RESOURCE_GROUP="astralytiq-rg"
    CONTAINER_NAME="astralytiq-backend"
    REGISTRY_NAME="astralytiqacr"
    
    # Create resource group if it doesn't exist
    az group create --name $RESOURCE_GROUP --location eastus
    
    # Create Azure Container Registry if it doesn't exist
    az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic --admin-enabled true
    
    # Build and push to ACR
    az acr build --registry $REGISTRY_NAME --image $IMAGE_NAME:$IMAGE_TAG -f Dockerfile.backend .
    
    # Deploy to Container Instances
    az container create \
        --resource-group $RESOURCE_GROUP \
        --name $CONTAINER_NAME \
        --image $REGISTRY_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG \
        --cpu 1 \
        --memory 1 \
        --registry-login-server $REGISTRY_NAME.azurecr.io \
        --registry-username $REGISTRY_NAME \
        --registry-password $(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" -o tsv) \
        --dns-name-label astralytiq-backend-$(date +%s) \
        --ports 8080 \
        --environment-variables \
            ENVIRONMENT=production \
            JWT_SECRET=AstralytiQ-Production-JWT-Secret-Key-2025-Campus-Placement \
            CORS_ORIGINS=https://astralytiq-platform.streamlit.app
    
    print_success "Deployed to Azure Container Instances"
    
    # Get the FQDN
    FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query ipAddress.fqdn -o tsv)
    print_success "Service URL: http://$FQDN:8080"
    print_step "API Documentation: http://$FQDN:8080/docs"
}

# Main menu
show_menu() {
    echo ""
    echo "Select deployment option:"
    echo "1) Build Docker image only"
    echo "2) Run locally with Docker"
    echo "3) Deploy to Google Cloud Run (FREE)"
    echo "4) Deploy to AWS ECS Fargate (FREE tier)"
    echo "5) Deploy to Azure Container Instances (FREE trial)"
    echo "6) Build and run locally"
    echo "7) Exit"
    echo ""
}

# Main execution
main() {
    while true; do
        show_menu
        read -p "Enter your choice (1-7): " choice
        
        case $choice in
            1)
                build_image
                ;;
            2)
                build_image
                run_local
                ;;
            3)
                build_image
                deploy_gcp
                ;;
            4)
                build_image
                deploy_aws
                ;;
            5)
                build_image
                deploy_azure
                ;;
            6)
                build_image
                run_local
                ;;
            7)
                print_success "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main