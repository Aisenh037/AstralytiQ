# 🚀 Deployment Guide - Enterprise SaaS Platform

## 📋 Deployment Options Comparison

| Option | Complexity | Cost | Scalability | Best For |
|--------|------------|------|-------------|----------|
| **Streamlit Cloud** | ⭐ | Free | ⭐⭐ | Quick demos, prototypes |
| **Railway/Render** | ⭐⭐ | $5-20/month | ⭐⭐⭐ | Small teams, MVPs |
| **AWS/GCP** | ⭐⭐⭐⭐ | $50-500/month | ⭐⭐⭐⭐⭐ | Production, enterprise |
| **Docker + VPS** | ⭐⭐⭐ | $10-50/month | ⭐⭐⭐ | Custom deployments |

## 🎯 Recommended Deployment Path

### Phase 1: Quick Demo (Streamlit Cloud)
**Timeline: 30 minutes**
- Perfect for showcasing to stakeholders
- Zero infrastructure management
- Free hosting

### Phase 2: MVP Deployment (Railway/Render)
**Timeline: 2-4 hours**
- Production-ready with databases
- Easy CI/CD integration
- Affordable for small teams

### Phase 3: Enterprise Production (AWS/GCP)
**Timeline: 1-2 days**
- Full scalability and reliability
- Advanced monitoring and security
- Enterprise-grade infrastructure

---

## 🌟 Option 1: Streamlit Cloud (Recommended for Demo)

### Prerequisites
- GitHub account
- Streamlit Cloud account (free)

### Steps

#### 1. Prepare Repository
```bash
# Create streamlit-specific requirements
echo "streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
requests>=2.31.0
networkx>=3.1" > requirements.txt

# Create Streamlit config
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
EOF
```

#### 2. Deploy to Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `streamlit_app.py` as main file
5. Deploy automatically

#### 3. Configure for Demo
```python
# Add to streamlit_app.py for demo mode
DEMO_MODE = True
API_BASE_URL = "https://your-api-demo.herokuapp.com"  # If you have backend deployed
```

**✅ Result**: Live demo at `https://share.streamlit.io/yourusername/yourrepo/main/streamlit_app.py`

---

## 🚂 Option 2: Railway Deployment (Recommended for MVP)

### Prerequisites
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
```

### Backend Deployment

#### 1. Prepare for Railway
```bash
# Create railway.json
cat > railway.json << EOF
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "python scripts/start_services.py",
    "healthcheckPath": "/health"
  }
}
EOF

# Create Dockerfile for production
cat > Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-enterprise.txt .
RUN pip install --no-cache-dir -r requirements-enterprise.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "src.services.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
```

#### 2. Deploy Services
```bash
# Initialize Railway project
railway init

# Add databases
railway add postgresql
railway add redis

# Set environment variables
railway variables set DATABASE_URL=${{RAILWAY_POSTGRESQL_URL}}
railway variables set REDIS_URL=${{RAILWAY_REDIS_URL}}
railway variables set JWT_SECRET_KEY="your-secret-key"

# Deploy
railway up
```

#### 3. Deploy Frontend
```bash
# Create separate Railway service for Streamlit
railway init streamlit-frontend

# Create Dockerfile for Streamlit
cat > Dockerfile.streamlit << EOF
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY streamlit_app.py .
COPY .streamlit/ .streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

railway up
```

**✅ Result**: 
- Backend API: `https://your-app.railway.app`
- Frontend: `https://your-frontend.railway.app`

---

## ☁️ Option 3: AWS Production Deployment

### Prerequisites
```bash
# Install AWS CLI and Terraform
pip install awscli
aws configure

# Install Terraform
# Download from https://terraform.io
```

### Architecture Overview
```
Internet → CloudFront → ALB → ECS Fargate Services
                              ↓
                         RDS + ElastiCache + DocumentDB
```

### Deployment Steps

#### 1. Create Terraform Configuration
```bash
mkdir -p infrastructure/terraform
cd infrastructure/terraform

# Create main.tf
cat > main.tf << 'EOF'
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = {
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier = "${var.project_name}-postgres"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  
  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project_name}-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
  
  enable_deletion_protection = false
  
  tags = {
    Environment = var.environment
  }
}
EOF

# Create variables.tf
cat > variables.tf << 'EOF'
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "enterprise-saas"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "enterprise_saas"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "admin"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}
EOF
```

#### 2. Deploy Infrastructure
```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Apply infrastructure
terraform apply
```

#### 3. Deploy Services to ECS
```bash
# Build and push Docker images
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build images
docker build -t enterprise-saas/api-gateway src/services/api_gateway/
docker build -t enterprise-saas/user-service src/services/user_service/
docker build -t enterprise-saas/tenant-service src/services/tenant_service/
docker build -t enterprise-saas/data-service src/services/data_service/

# Tag and push
docker tag enterprise-saas/api-gateway:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/enterprise-saas/api-gateway:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/enterprise-saas/api-gateway:latest
```

**✅ Result**: Production-ready deployment with auto-scaling, monitoring, and high availability

---

## 🐳 Option 4: Docker Containerization (Recommended for Production)

### Overview
The platform now includes comprehensive Docker containerization with:
- **Multi-stage builds** for optimized image sizes
- **Multi-service architecture** with dedicated containers
- **Development and production** configurations
- **Load balancing** with Nginx
- **Monitoring** with Prometheus and Grafana
- **Security hardening** with non-root users

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- 20GB+ disk space

### Quick Start

#### Development Environment
```bash
# Validate Docker setup
scripts/validate-docker.bat  # Windows
# or
./scripts/validate-docker.sh  # Linux/macOS

# Start development environment
scripts/docker-build.bat start-dev  # Windows
# or
./scripts/docker-build.sh start-dev  # Linux/macOS

# Check service health
scripts/docker-build.bat health
```

#### Production Environment
```bash
# Build all services
scripts/docker-build.bat build

# Start production environment
scripts/docker-build.bat start-prod

# Check logs
scripts/docker-build.bat logs
```

### Service Architecture

| Service | Port | Purpose | Dependencies |
|---------|------|---------|--------------|
| **API Gateway** | 8000 | Request routing, authentication | All services |
| **User Service** | 8001 | Authentication, user management | PostgreSQL, Redis |
| **Tenant Service** | 8002 | Multi-tenancy, organizations | PostgreSQL, Redis |
| **Data Service** | 8003 | Data processing, ETL | PostgreSQL, MongoDB, Redis |
| **ML Service** | 8004 | Machine learning, analytics | PostgreSQL, MongoDB, Redis |
| **PostgreSQL** | 5432 | Primary database | - |
| **MongoDB** | 27017 | Document storage | - |
| **Redis** | 6379 | Caching, sessions | - |
| **Nginx** | 80/443 | Load balancer, SSL termination | API Gateway |

### Docker Compose Configurations

#### Base Configuration (`docker-compose.yml`)
- Core service definitions
- Health checks and dependencies
- Network configuration
- Volume mounts

#### Development Override (`docker-compose.dev.yml`)
- Hot reload enabled
- Debug logging
- Development tools (pgAdmin, mongo-express)
- Exposed database ports

#### Production Override (`docker-compose.prod.yml`)
- Multi-worker processes with Gunicorn
- Resource limits and reservations
- Security hardening
- Monitoring enabled

### Environment Variables

#### Required for Production
```bash
# Database credentials
POSTGRES_PASSWORD=your-secure-password
MONGO_PASSWORD=your-secure-password

# JWT configuration
JWT_SECRET_KEY=your-jwt-secret-key

# Optional registry configuration
DOCKER_REGISTRY=your-registry.com
VERSION=v1.0.0
```

### Build Scripts

#### Windows (`scripts/docker-build.bat`)
```cmd
scripts\docker-build.bat build          # Build all services
scripts\docker-build.bat build gateway  # Build specific service
scripts\docker-build.bat start-dev      # Start development
scripts\docker-build.bat start-prod     # Start production
scripts\docker-build.bat logs           # View logs
scripts\docker-build.bat health         # Health check
scripts\docker-build.bat cleanup        # Cleanup resources
```

#### Linux/macOS (`scripts/docker-build.sh`)
```bash
./scripts/docker-build.sh build         # Build all services
./scripts/docker-build.sh start-dev     # Start development
./scripts/docker-build.sh health        # Health check
./scripts/docker-build.sh push          # Push to registry
```

### Service Access

#### Development Environment
- **API Gateway**: http://localhost:8000
- **User Service**: http://localhost:8001
- **Tenant Service**: http://localhost:8002
- **Data Service**: http://localhost:8003
- **ML Service**: http://localhost:8004
- **pgAdmin**: http://localhost:5050
- **Mongo Express**: http://localhost:8081
- **Redis Commander**: http://localhost:8082

#### Production Environment
- **Load Balancer**: http://localhost:80
- **HTTPS**: https://localhost:443
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Security Features

#### Container Security
- Non-root user execution
- Multi-stage builds for minimal attack surface
- Read-only file systems where possible
- Health checks for all services

#### Network Security
- Custom bridge network with service discovery
- Rate limiting (10 req/s API, 2 req/s uploads)
- Security headers and CORS configuration
- SSL/TLS termination at load balancer

### Monitoring and Logging

#### Health Checks
All services include health check endpoints:
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3

#### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Structured logging**: JSON format with centralized collection

### Storage and Persistence

#### Volumes
- `postgres_data`: PostgreSQL database files
- `mongodb_data`: MongoDB database files
- `redis_data`: Redis persistence files
- `data_storage`: Uploaded files and datasets
- `ml_storage`: ML models and artifacts

#### Backup Strategy
```bash
# Database backup
docker-compose exec postgres pg_dump -U postgres enterprise_saas > backup.sql

# Volume backup
docker run --rm -v data_storage:/data -v $(pwd):/backup alpine tar czf /backup/data_backup.tar.gz /data
```

### Scaling and Production Deployment

#### Horizontal Scaling
```bash
# Scale specific services
docker-compose up -d --scale api-gateway=3 --scale user-service=2

# Use Docker Swarm for orchestration
docker swarm init
docker stack deploy -c docker-compose.yml -c docker-compose.prod.yml enterprise-saas
```

#### Production Checklist
- [ ] SSL certificates configured
- [ ] Production environment variables set
- [ ] Database credentials secured
- [ ] Monitoring and alerting configured
- [ ] Backup strategy implemented
- [ ] Resource limits configured
- [ ] Security scanning completed

### Troubleshooting

#### Common Issues
1. **Port conflicts**: Ensure ports 8000-8004, 5432, 27017, 6379 are available
2. **Memory issues**: Increase Docker memory limit to 8GB+
3. **Permission errors**: Check file permissions and user ownership
4. **Network issues**: Verify Docker network configuration

#### Debug Commands
```bash
# Check container status
docker-compose ps

# View service logs
docker-compose logs -f [service]

# Execute commands in container
docker-compose exec [service] bash

# Check resource usage
docker stats
```

**✅ Result**: Production-ready containerized deployment with microservices architecture, monitoring, and security hardening.

For detailed Docker documentation, see `DOCKER_README.md`.

---

## 🌐 Option 5: VPS Deployment with Docker

### Prerequisites
- VPS with Docker installed (DigitalOcean, Linode, etc.)
- Domain name (optional)

### Steps

#### 1. Prepare Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: enterprise_saas
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  api-gateway:
    build:
      context: .
      dockerfile: src/services/api_gateway/Dockerfile
    ports:
      - "80:8000"
    environment:
      - DATABASE_URL=postgresql://admin:${DB_PASSWORD}@postgres:5432/enterprise_saas
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  user-service:
    build:
      context: .
      dockerfile: src/services/user_service/Dockerfile
    environment:
      - DATABASE_URL=postgresql://admin:${DB_PASSWORD}@postgres:5432/enterprise_saas
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  data-service:
    build:
      context: .
      dockerfile: src/services/data_service/Dockerfile
    environment:
      - DATABASE_URL=postgresql://admin:${DB_PASSWORD}@postgres:5432/enterprise_saas
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api-gateway:8000
    depends_on:
      - api-gateway
    restart: unless-stopped

volumes:
  postgres_data:
```

#### 2. Deploy to VPS
```bash
# Copy files to VPS
scp -r . user@your-vps-ip:/home/user/enterprise-saas/

# SSH to VPS
ssh user@your-vps-ip

# Set environment variables
cat > .env << EOF
DB_PASSWORD=your-secure-password
JWT_SECRET_KEY=your-jwt-secret-key
EOF

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Set up reverse proxy with Nginx (optional)
sudo apt install nginx
sudo nano /etc/nginx/sites-available/enterprise-saas
```

**✅ Result**: Self-hosted production deployment with full control

---

## 🔧 Post-Deployment Configuration

### SSL Certificate (Let's Encrypt)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Monitoring Setup
```bash
# Add monitoring to docker-compose
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Backup Strategy
```bash
# Database backup script
#!/bin/bash
docker exec postgres_container pg_dump -U admin enterprise_saas > backup_$(date +%Y%m%d_%H%M%S).sql

# Add to crontab for daily backups
0 2 * * * /path/to/backup_script.sh
```

---

## 🎯 Deployment Checklist

### Pre-Deployment
- [ ] All tests passing locally
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] SSL certificates obtained
- [ ] Domain DNS configured

### Post-Deployment
- [ ] Health checks passing
- [ ] Database connected
- [ ] Authentication working
- [ ] File uploads functional
- [ ] API documentation accessible
- [ ] Monitoring configured
- [ ] Backups scheduled
- [ ] SSL certificate valid

### Security Checklist
- [ ] JWT secrets are secure
- [ ] Database passwords are strong
- [ ] API rate limiting enabled
- [ ] Input validation implemented
- [ ] HTTPS enforced
- [ ] Security headers configured

---

## 📞 Support and Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker logs <container-name>

# Check resource usage
docker stats

# Restart service
docker-compose restart <service-name>
```

#### Database Connection Issues
```bash
# Test database connection
docker exec -it postgres_container psql -U admin -d enterprise_saas

# Check database logs
docker logs postgres_container
```

#### Performance Issues
```bash
# Monitor resource usage
htop
docker stats

# Check database performance
# Connect to PostgreSQL and run:
SELECT * FROM pg_stat_activity;
```

### Getting Help
- **Documentation**: Check `DEVELOPMENT_GUIDE.md`
- **Logs**: Always check service logs first
- **Health Checks**: Use `/health` endpoints
- **Community**: Create GitHub issues for bugs

---

**🚀 Choose your deployment option and get your Enterprise SaaS Platform live!**