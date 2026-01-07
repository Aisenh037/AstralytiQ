# Enterprise SaaS Platform - Development Guide

## 🚀 Project Overview

This is a comprehensive enterprise-grade SaaS analytics platform built with modern microservices architecture. The platform transforms from a monolithic Streamlit application into a scalable, multi-tenant system with advanced data processing, ML capabilities, and comprehensive lineage tracking.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Completed Features](#completed-features)
3. [Development Setup](#development-setup)
4. [Local Testing](#local-testing)
5. [API Documentation](#api-documentation)
6. [Frontend Options](#frontend-options)
7. [Cloud Deployment](#cloud-deployment)
8. [Team Development Guidelines](#team-development-guidelines)
9. [Testing Strategy](#testing-strategy)
10. [Troubleshooting](#troubleshooting)

## 🏗️ Architecture Overview

### Microservices Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  User Service   │    │ Tenant Service  │
│   Port: 8000    │    │   Port: 8001    │    │   Port: 8002    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │  Data Service   │    │   ML Service    │    │Dashboard Service│
         │   Port: 8003    │    │   Port: 8004    │    │   Port: 8005    │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Backend**: FastAPI, Python 3.11+
- **Databases**: PostgreSQL, MongoDB, Redis
- **Authentication**: JWT with RBAC
- **Data Processing**: Pandas, NumPy
- **ML**: Scikit-learn, TensorFlow/PyTorch
- **Containerization**: Docker, Docker Compose
- **Testing**: Pytest, Property-based testing
- **Documentation**: OpenAPI/Swagger

## ✅ Completed Features

### Core Infrastructure (Tasks 1-5)
- ✅ **Microservices Architecture**: Complete separation of concerns
- ✅ **User Management**: JWT auth, RBAC, password reset
- ✅ **Multi-Tenant System**: Tenant isolation, quotas, context middleware
- ✅ **API Gateway**: Request routing, rate limiting, versioning
- ✅ **Security**: Authentication, authorization, input validation

### Data Processing Service (Task 6)
- ✅ **Data Upload & Validation** (6.1): Multi-format file processing
- ✅ **Multi-Format Processing** (6.3): CSV, Excel, JSON, XML, TSV, Parquet
- ✅ **Transformation Engine** (6.5): 13 transformation types, pipeline orchestration
- ✅ **Data Lineage Tracking** (6.7): Complete provenance, impact analysis

### Key Capabilities
- **13 Data Transformations**: Cleaning, normalization, aggregation, filtering
- **Comprehensive Lineage**: Graph-based tracking, visualization support
- **Enterprise Security**: Multi-tenant isolation, RBAC, audit trails
- **Performance Optimization**: Caching, efficient algorithms, scalable architecture
- **API-First Design**: RESTful APIs with comprehensive documentation
## 🛠️ Development Setup

### Prerequisites
```bash
# Required software
- Python 3.11+
- Docker & Docker Compose
- Git
- Node.js 18+ (for frontend development)
```

### Environment Setup
```bash
# 1. Clone the repository
git clone <repository-url>
cd forecasting_bi_app

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-enterprise.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Initialize databases
docker-compose up -d postgres mongodb redis

# 6. Run database migrations
python src/shared/infrastructure/migrations.py
```

### Project Structure
```
forecasting_bi_app/
├── src/
│   ├── services/
│   │   ├── api_gateway/          # API Gateway service
│   │   ├── user_service/         # User management
│   │   ├── tenant_service/       # Multi-tenancy
│   │   ├── data_service/         # Data processing
│   │   ├── ml_service/           # ML/Analytics (TODO)
│   │   └── dashboard_service/    # Dashboards (TODO)
│   └── shared/                   # Shared components
├── tests/                        # Test suites
├── scripts/                      # Utility scripts
├── docker-compose.yml           # Local development
└── requirements-enterprise.txt  # Dependencies
```

## 🧪 Local Testing

### Running Individual Services
```bash
# Start all services with Docker Compose
docker-compose up -d

# Or run services individually:

# API Gateway (Port 8000)
cd src/services/api_gateway
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# User Service (Port 8001)
cd src/services/user_service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Tenant Service (Port 8002)
cd src/services/tenant_service
uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# Data Service (Port 8003)
cd src/services/data_service
uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

### Quick Start Script
```bash
# Use the provided startup script
python scripts/start_services.py

# This will:
# 1. Start all required databases
# 2. Run database migrations
# 3. Start all microservices
# 4. Display service URLs and health checks
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_basic_structure.py -v

# Run lineage tracking tests
python test_lineage_tracking.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Health Checks
```bash
# Check service health
curl http://localhost:8000/health  # API Gateway
curl http://localhost:8001/health  # User Service
curl http://localhost:8002/health  # Tenant Service
curl http://localhost:8003/health  # Data Service
```

## 📚 API Documentation

### Interactive Documentation
- **API Gateway**: http://localhost:8000/docs
- **User Service**: http://localhost:8001/docs
- **Tenant Service**: http://localhost:8002/docs
- **Data Service**: http://localhost:8003/docs

### Key API Endpoints

#### Authentication
```bash
# Register user
POST /api/v1/auth/register
{
  "email": "user@example.com",
  "password": "securepassword",
  "full_name": "John Doe"
}

# Login
POST /api/v1/auth/login
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

#### Data Processing
```bash
# Upload dataset
POST /api/v1/data/upload
Content-Type: multipart/form-data
- file: <file>
- name: "Sales Data"
- description: "Monthly sales data"

# Apply transformations
POST /api/v1/data/datasets/{id}/transform
{
  "transformations": [
    {
      "step": "remove_duplicates",
      "parameters": {"keep": "first"}
    }
  ]
}

# Get lineage
GET /api/v1/data/datasets/{id}/lineage?direction=both&max_depth=5
```
## 🎨 Frontend Options

### Option 1: React Frontend (Recommended)
```bash
# Create React app with TypeScript
npx create-react-app frontend --template typescript
cd frontend

# Install additional dependencies
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material @mui/lab
npm install axios react-router-dom @types/react-router-dom
npm install recharts react-flow-renderer  # For data visualization

# Start development server
npm start  # Runs on http://localhost:3000
```

#### React Project Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── auth/           # Authentication components
│   │   ├── dashboard/      # Dashboard components
│   │   ├── data/          # Data management
│   │   └── lineage/       # Lineage visualization
│   ├── services/          # API service calls
│   ├── hooks/             # Custom React hooks
│   ├── utils/             # Utility functions
│   └── types/             # TypeScript types
├── public/
└── package.json
```

#### Key React Components to Build
```typescript
// src/components/auth/LoginForm.tsx
// src/components/data/DataUpload.tsx
// src/components/data/TransformationPipeline.tsx
// src/components/lineage/LineageGraph.tsx
// src/components/dashboard/DataDashboard.tsx
```

### Option 2: Streamlit Frontend (Quick Prototyping)
```bash
# Create Streamlit app
pip install streamlit plotly

# Create streamlit_app.py
streamlit run streamlit_app.py  # Runs on http://localhost:8501
```

### Option 3: FastAPI + Jinja2 Templates (Simple)
```bash
# Add to existing FastAPI services
pip install jinja2 python-multipart

# Create templates directory in each service
# Add HTML templates with Bootstrap/Tailwind CSS
```

## ☁️ Cloud Deployment

### Option 1: AWS Deployment (Recommended for Production)

#### Prerequisites
```bash
# Install AWS CLI
pip install awscli
aws configure

# Install Terraform
# Download from https://terraform.io
```

#### AWS Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudFront    │    │   Application   │    │   RDS/Aurora    │
│   (CDN/WAF)     │    │  Load Balancer  │    │  (PostgreSQL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │   ECS Fargate   │    │   ElastiCache   │    │   DocumentDB    │
         │  (Containers)   │    │    (Redis)      │    │   (MongoDB)     │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### Deployment Steps
```bash
# 1. Build and push Docker images
docker build -t your-registry/api-gateway:latest src/services/api_gateway/
docker push your-registry/api-gateway:latest

# 2. Deploy with Terraform
cd infrastructure/terraform
terraform init
terraform plan
terraform apply

# 3. Set up CI/CD with GitHub Actions
# See .github/workflows/deploy.yml
```

### Option 2: Google Cloud Platform
```bash
# Use Cloud Run for serverless containers
gcloud run deploy api-gateway \
  --image gcr.io/your-project/api-gateway:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 3: Streamlit Cloud (Quick Demo)
```bash
# 1. Create streamlit_app.py in root directory
# 2. Push to GitHub
# 3. Connect repository to Streamlit Cloud
# 4. Deploy automatically

# Streamlit Cloud URL: https://share.streamlit.io/
```

### Option 4: Railway/Render (Simple Deployment)
```bash
# Railway
railway login
railway init
railway up

# Render
# Connect GitHub repository
# Configure build and start commands
# Deploy automatically on push
```

## 👥 Team Development Guidelines

### Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/new-feature
git commit -m "feat: add new feature"
git push origin feature/new-feature
# Create pull request

# Branch naming convention
feature/feature-name
bugfix/bug-description
hotfix/critical-fix
```

### Code Standards
```python
# Use Black for formatting
pip install black
black src/

# Use isort for imports
pip install isort
isort src/

# Use mypy for type checking
pip install mypy
mypy src/

# Use flake8 for linting
pip install flake8
flake8 src/
```

### Development Workflow
1. **Create Feature Branch**: `git checkout -b feature/your-feature`
2. **Write Tests First**: Follow TDD approach
3. **Implement Feature**: Write minimal code to pass tests
4. **Run All Tests**: Ensure nothing breaks
5. **Update Documentation**: Update relevant .md files
6. **Create Pull Request**: Include description and test results
7. **Code Review**: Team review and approval
8. **Merge to Main**: Deploy to staging/production

### Task Management
- Use the task list in `.kiro/specs/enterprise-saas-platform/tasks.md`
- Mark tasks as completed: `[x]`
- Update completion summaries: `TASK_X_COMPLETION_SUMMARY.md`
- Document any architectural decisions

## 🧪 Testing Strategy

### Test Types
```bash
# Unit Tests
python -m pytest tests/test_basic_structure.py -v

# Integration Tests
python -m pytest tests/test_integration.py -v

# Property-Based Tests (when implemented)
python -m pytest tests/test_properties.py -v

# End-to-End Tests
python -m pytest tests/test_e2e.py -v

# Load Tests
pip install locust
locust -f tests/load_tests.py
```

### Test Coverage
```bash
# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Manual Testing Checklist
- [ ] User registration and login
- [ ] File upload (CSV, Excel, JSON)
- [ ] Data transformations
- [ ] Lineage visualization
- [ ] Multi-tenant isolation
- [ ] API rate limiting
- [ ] Error handling
## 🔧 Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check if databases are running
docker ps | grep postgres
docker ps | grep mongo
docker ps | grep redis

# Restart databases
docker-compose restart postgres mongodb redis

# Check database logs
docker logs <container-id>
```

#### Service Startup Issues
```bash
# Check service logs
docker logs <service-container>

# Verify environment variables
cat .env

# Check port conflicts
netstat -tulpn | grep :8000
```

#### Import/Module Issues
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use relative imports
python -m src.services.api_gateway.main
```

#### Test Failures
```bash
# Run tests with verbose output
python -m pytest tests/ -v -s

# Run specific test
python -m pytest tests/test_integration.py::TestClass::test_method -v

# Debug with pdb
python -m pytest tests/ --pdb
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check database performance
# Connect to PostgreSQL
docker exec -it postgres_container psql -U username -d database

# Check slow queries
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

### Security Checklist
- [ ] Environment variables not committed
- [ ] JWT secrets are secure
- [ ] Database credentials are encrypted
- [ ] API endpoints have proper authentication
- [ ] Input validation is implemented
- [ ] Rate limiting is configured

## 📈 Monitoring and Observability

### Health Monitoring
```bash
# Service health endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

### Logging
```python
# Structured logging example
import logging
import json

logger = logging.getLogger(__name__)

def log_api_request(request, response):
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "response_time": response.elapsed.total_seconds()
    }
    logger.info(json.dumps(log_data))
```

### Metrics Collection
```python
# Prometheus metrics example
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
```

## 🚀 Next Steps for Development

### Immediate Tasks (Ready to Implement)
1. **Frontend Development**: Choose React or Streamlit
2. **ML Service** (Task 7): Model training and deployment
3. **Dashboard Service** (Task 8): Visualization and reporting
4. **Billing Service** (Task 9): Subscription management

### Recommended Development Order
1. **Start with Streamlit** for quick prototyping and testing
2. **Build React frontend** for production-ready UI
3. **Implement ML service** for analytics capabilities
4. **Add dashboard service** for data visualization
5. **Deploy to cloud** for production access

### Team Assignments
- **Backend Developer**: ML service, billing service
- **Frontend Developer**: React UI, dashboard components
- **DevOps Engineer**: Cloud deployment, CI/CD
- **Data Engineer**: Advanced transformations, ML pipelines

## 📞 Support and Resources

### Documentation
- **API Docs**: Available at `/docs` endpoint for each service
- **Architecture**: See `README-Enterprise.md`
- **Task Progress**: Check `.kiro/specs/enterprise-saas-platform/tasks.md`

### Useful Commands
```bash
# Quick service restart
docker-compose restart

# View all logs
docker-compose logs -f

# Clean up containers
docker-compose down -v
docker system prune -a

# Database backup
docker exec postgres_container pg_dump -U username database > backup.sql

# Restore database
docker exec -i postgres_container psql -U username database < backup.sql
```

### External Resources
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://reactjs.org/docs/
- **Docker Documentation**: https://docs.docker.com/
- **AWS Documentation**: https://docs.aws.amazon.com/
- **Streamlit Documentation**: https://docs.streamlit.io/

---

## 🎯 Quick Start for New Developers

1. **Clone and Setup**: Follow [Development Setup](#development-setup)
2. **Run Tests**: `python -m pytest tests/ -v`
3. **Start Services**: `python scripts/start_services.py`
4. **Test APIs**: Visit http://localhost:8000/docs
5. **Choose Frontend**: React (production) or Streamlit (prototype)
6. **Deploy**: Start with Streamlit Cloud for quick demo

**Happy Coding! 🚀**