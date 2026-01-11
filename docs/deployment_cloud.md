# 🚀 Cloud Deployment Guide - Docker + Free Tiers

## 🎯 **Campus Placement Impact Ranking**

### 🥇 **Google Cloud Run (HIGHEST IMPACT)**
- **Free Tier**: 2M requests/month, 400,000 GB-seconds
- **Perfect for**: Enterprise interviews, scalability demos
- **Time**: 15 minutes
- **Skills Shown**: GCP, Docker, serverless, auto-scaling

### 🥈 **AWS ECS Fargate (HIGH IMPACT)**  
- **Free Tier**: 12 months free tier
- **Perfect for**: AWS-focused companies
- **Time**: 20 minutes
- **Skills Shown**: AWS, containers, microservices

### 🥉 **Azure Container Instances (GOOD IMPACT)**
- **Free Tier**: $200 credit for 30 days
- **Perfect for**: Microsoft-focused companies
- **Time**: 15 minutes
- **Skills Shown**: Azure, containers, cloud deployment

---

## 🐳 **Docker Setup (All Platforms)**

### **Quick Start**
```bash
# Make script executable (Linux/Mac)
chmod +x scripts/deploy-docker.sh
./scripts/deploy-docker.sh

# Windows
scripts\deploy-docker.bat
```

### **Manual Docker Commands**
```bash
# Build image
docker build -f Dockerfile.backend -t astralytiq-backend .

# Run locally
docker run -d \
  --name astralytiq-backend \
  -p 8080:8080 \
  -e ENVIRONMENT=production \
  -e JWT_SECRET=AstralytiQ-Production-JWT-Secret-Key-2025 \
  -e CORS_ORIGINS=https://astralytiq-platform.streamlit.app \
  astralytiq-backend

# Test locally
curl http://localhost:8080/health
```

---

## ☁️ **Cloud Deployment Options**

### 🟦 **Google Cloud Run (RECOMMENDED)**

#### **Prerequisites**
```bash
# Install Google Cloud SDK
# Windows: Download from https://cloud.google.com/sdk/docs/install
# Mac: brew install google-cloud-sdk
# Linux: curl https://sdk.cloud.google.com | bash

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

#### **Deploy**
```bash
# Option 1: Using our script
./scripts/deploy-docker.sh
# Select option 3

# Option 2: Manual deployment
gcloud builds submit --config cloudbuild.yaml .

# Option 3: Direct deployment
gcloud run deploy astralytiq-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --set-env-vars ENVIRONMENT=production,JWT_SECRET=AstralytiQ-Production-JWT-Secret-Key-2025,CORS_ORIGINS=https://astralytiq-platform.streamlit.app
```

#### **Benefits**
- ✅ **Serverless**: Auto-scaling from 0 to 1000+ instances
- ✅ **Pay-per-use**: Only pay when requests are processed
- ✅ **HTTPS**: Automatic SSL certificates
- ✅ **Global**: Deploy to multiple regions
- ✅ **Monitoring**: Built-in logging and metrics

---

### 🟧 **AWS ECS Fargate**

#### **Prerequisites**
```bash
# Install AWS CLI
# Windows: Download from https://aws.amazon.com/cli/
# Mac: brew install awscli
# Linux: pip install awscli

# Configure AWS
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Output format (json)

# Enable services
aws ecs create-cluster --cluster-name astralytiq-cluster
```

#### **Deploy**
```bash
# Option 1: Using our script
./scripts/deploy-docker.sh
# Select option 4

# Option 2: Manual deployment
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Create ECR repository
aws ecr create-repository --repository-name astralytiq-backend --region $REGION

# Build and push
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker tag astralytiq-backend:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/astralytiq-backend:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/astralytiq-backend:latest

# Create ECS service (requires additional setup)
```

#### **Benefits**
- ✅ **Serverless containers**: No EC2 management
- ✅ **AWS integration**: Works with all AWS services
- ✅ **Security**: VPC, IAM, security groups
- ✅ **Monitoring**: CloudWatch integration

---

### 🟦 **Azure Container Instances**

#### **Prerequisites**
```bash
# Install Azure CLI
# Windows: Download from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
# Mac: brew install azure-cli
# Linux: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login
az account set --subscription "Your Subscription Name"
```

#### **Deploy**
```bash
# Option 1: Using our script
./scripts/deploy-docker.sh
# Select option 5

# Option 2: Manual deployment
RESOURCE_GROUP=astralytiq-rg
REGISTRY_NAME=astralytiqacr

# Create resource group
az group create --name $RESOURCE_GROUP --location eastus

# Create container registry
az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic --admin-enabled true

# Build and push
az acr build --registry $REGISTRY_NAME --image astralytiq-backend:latest -f Dockerfile.backend .

# Deploy container
az container create \
  --resource-group $RESOURCE_GROUP \
  --name astralytiq-backend \
  --image $REGISTRY_NAME.azurecr.io/astralytiq-backend:latest \
  --cpu 1 --memory 1 \
  --registry-login-server $REGISTRY_NAME.azurecr.io \
  --registry-username $REGISTRY_NAME \
  --registry-password $(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" -o tsv) \
  --dns-name-label astralytiq-backend-$(date +%s) \
  --ports 8080 \
  --environment-variables ENVIRONMENT=production JWT_SECRET=AstralytiQ-Production-JWT-Secret-Key-2025 CORS_ORIGINS=https://astralytiq-platform.streamlit.app
```

#### **Benefits**
- ✅ **Simple**: Easiest container deployment
- ✅ **Fast**: Quick startup times
- ✅ **Integrated**: Works with Azure services
- ✅ **Flexible**: Custom networking options

---

## 🔧 **Post-Deployment Configuration**

### **Update Streamlit Secrets**
After deployment, update your Streamlit Cloud secrets:

```toml
# Replace with your actual deployment URL
API_BASE_URL = "https://your-service-url.com"
USER_SERVICE_URL = "https://your-service-url.com/api/v1/users"
DATA_SERVICE_URL = "https://your-service-url.com/api/v1/data"
ML_SERVICE_URL = "https://your-service-url.com/api/v1/ml"
DASHBOARD_SERVICE_URL = "https://your-service-url.com/api/v1/dashboard"
```

### **Test Deployment**
```bash
# Health check
curl https://your-service-url.com/health

# API documentation
open https://your-service-url.com/docs

# Test authentication
curl -X POST https://your-service-url.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@astralytiq.com","password":"admin123"}'
```

---

## 💰 **Free Tier Limits**

### **Google Cloud Run**
- **Requests**: 2M per month
- **CPU**: 400,000 GB-seconds per month
- **Memory**: 400,000 GB-seconds per month
- **Build time**: 120 minutes per day
- **Perfect for**: Demos and small applications

### **AWS ECS Fargate**
- **Duration**: 12 months from signup
- **Compute**: 400,000 GB-seconds per month
- **Storage**: 20 GB per month
- **Data transfer**: 15 GB per month
- **Perfect for**: Learning AWS ecosystem

### **Azure Container Instances**
- **Credit**: $200 for first 30 days
- **After trial**: Pay-as-you-go pricing
- **Perfect for**: Short-term projects and demos

---

## 🎯 **Campus Placement Benefits**

### **What This Demonstrates**
✅ **Containerization**: Docker expertise
✅ **Cloud Platforms**: Multi-cloud deployment experience
✅ **DevOps**: CI/CD pipelines and automation
✅ **Scalability**: Auto-scaling serverless architecture
✅ **Security**: Environment variables, secrets management
✅ **Monitoring**: Health checks and logging
✅ **Cost Optimization**: Free tier utilization

### **Interview Talking Points**
- "Containerized FastAPI backend using Docker multi-stage builds"
- "Deployed to multiple cloud platforms: GCP, AWS, Azure"
- "Implemented serverless architecture with auto-scaling"
- "Used infrastructure as code with YAML configurations"
- "Optimized for cost using free tier resources"
- "Configured CI/CD pipelines for automated deployments"

---

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Docker Build Fails**
   ```bash
   # Check Docker is running
   docker --version
   
   # Clear Docker cache
   docker system prune -a
   ```

2. **Cloud Authentication**
   ```bash
   # GCP
   gcloud auth list
   gcloud config list
   
   # AWS
   aws sts get-caller-identity
   
   # Azure
   az account show
   ```

3. **Memory/CPU Limits**
   - Increase memory allocation in cloud configuration
   - Use smaller base images (alpine, slim)
   - Optimize Python dependencies

4. **CORS Issues**
   - Verify CORS_ORIGINS environment variable
   - Check Streamlit app URL is included
   - Test with browser developer tools

### **Debug Commands**
```bash
# Check container logs
docker logs astralytiq-backend

# Connect to running container
docker exec -it astralytiq-backend /bin/bash

# Test API endpoints
curl -v http://localhost:8080/health
curl -v http://localhost:8080/docs
```

---

## 🎉 **Success Checklist**

- [ ] Docker image builds successfully
- [ ] Container runs locally on port 8080
- [ ] Health check endpoint responds
- [ ] API documentation accessible at /docs
- [ ] Cloud deployment successful
- [ ] HTTPS endpoint accessible
- [ ] Streamlit app connects to backend
- [ ] Authentication works end-to-end
- [ ] Auto-scaling configured (for serverless)

**Your enterprise-grade, cloud-deployed backend is now ready for campus placement demonstrations!** 🚀

## 🔄 **Next Steps**
1. Choose your preferred cloud platform
2. Run the deployment script
3. Update Streamlit secrets with new URL
4. Test full-stack integration
5. Prepare demo for interviews

**Recommended**: Start with **Google Cloud Run** for the best balance of features, ease of use, and interview impact!