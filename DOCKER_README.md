# Docker Containerization Guide

This guide covers the Docker containerization setup for the Enterprise SaaS Platform, including development and production deployment configurations.

## Overview

The platform uses a microservices architecture with the following containerized services:

- **API Gateway** (Port 8000): Main entry point and request routing
- **User Service** (Port 8001): Authentication and user management
- **Tenant Service** (Port 8002): Multi-tenancy and organization management
- **Data Service** (Port 8003): Data processing and ETL pipelines
- **ML Service** (Port 8004): Machine learning and analytics
- **PostgreSQL** (Port 5432): Primary database
- **MongoDB** (Port 27017): Document storage
- **Redis** (Port 6379): Caching and session storage
- **Nginx** (Port 80/443): Load balancer and reverse proxy

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- 20GB+ disk space

### Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd enterprise-saas-platform

# Start development environment
./scripts/docker-build.sh start-dev
# or on Windows:
scripts\docker-build.bat start-dev

# Check service health
./scripts/docker-build.sh health
```

### Production Environment

```bash
# Build all services
./scripts/docker-build.sh build

# Start production environment
./scripts/docker-build.sh start-prod

# Check logs
./scripts/docker-build.sh logs
```

## Docker Files

### Multi-Stage Build Architecture

All services use multi-stage Docker builds for optimization:

1. **Builder Stage**: Installs dependencies and compiles packages
2. **Production Stage**: Copies only necessary files and runs as non-root user

### Security Features

- Non-root user execution
- Minimal base images (Python 3.11-slim)
- Read-only file system where possible
- Health checks for all services
- Resource limits and reservations

## Docker Compose Configurations

### Base Configuration (`docker-compose.yml`)

Contains the core service definitions with:
- Service dependencies and health checks
- Network configuration
- Volume mounts
- Basic environment variables

### Development Override (`docker-compose.dev.yml`)

Development-specific configurations:
- Hot reload enabled
- Debug logging
- Exposed database ports
- Development tools (pgAdmin, mongo-express, redis-commander)

### Production Override (`docker-compose.prod.yml`)

Production-optimized configurations:
- Multi-worker processes with Gunicorn
- Resource limits and reservations
- Security hardening
- Monitoring and logging

## Environment Variables

### Required for Production

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

### Service-Specific Variables

```bash
# API Gateway
ENVIRONMENT=production
LOG_LEVEL=info
WORKERS=4

# Data Service
MAX_FILE_SIZE_MB=500
UPLOAD_PATH=/app/storage/uploads

# ML Service
MODEL_STORAGE_PATH=/app/storage/models
```

## Build Scripts

### Linux/macOS (`scripts/docker-build.sh`)

```bash
# Build all services
./scripts/docker-build.sh build

# Build specific service
./scripts/docker-build.sh build gateway

# Start development environment
./scripts/docker-build.sh start-dev

# Start production environment
./scripts/docker-build.sh start-prod

# View logs
./scripts/docker-build.sh logs [service]

# Health check
./scripts/docker-build.sh health

# Run tests
./scripts/docker-build.sh test

# Cleanup resources
./scripts/docker-build.sh cleanup

# Push to registry
./scripts/docker-build.sh push
```

### Windows (`scripts/docker-build.bat`)

Same commands but using `.bat` extension:

```cmd
scripts\docker-build.bat build
scripts\docker-build.bat start-dev
scripts\docker-build.bat health
```

## Service Access

### Development Environment

- API Gateway: http://localhost:8000
- User Service: http://localhost:8001
- Tenant Service: http://localhost:8002
- Data Service: http://localhost:8003
- ML Service: http://localhost:8004
- pgAdmin: http://localhost:5050
- Mongo Express: http://localhost:8081
- Redis Commander: http://localhost:8082

### Production Environment

- Load Balancer: http://localhost:80
- HTTPS: https://localhost:443
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Networking

### Custom Network

All services run on a custom bridge network (`enterprise-network`) with:
- Subnet: 172.20.0.0/16
- Service discovery via service names
- Isolated from host network

### Load Balancing

Nginx provides:
- Request routing to appropriate services
- Rate limiting (10 req/s for API, 2 req/s for uploads)
- SSL termination (when configured)
- Security headers
- Health check endpoints

## Storage and Persistence

### Volumes

- `postgres_data`: PostgreSQL database files
- `mongodb_data`: MongoDB database files
- `redis_data`: Redis persistence files
- `data_storage`: Uploaded files and datasets
- `ml_storage`: ML models and artifacts
- `prometheus_data`: Monitoring metrics
- `grafana_data`: Dashboard configurations

### Backup Strategy

```bash
# Database backup
docker-compose exec postgres pg_dump -U postgres enterprise_saas > backup.sql

# MongoDB backup
docker-compose exec mongodb mongodump --db enterprise_saas --out /backups

# Volume backup
docker run --rm -v data_storage:/data -v $(pwd):/backup alpine tar czf /backup/data_backup.tar.gz /data
```

## Monitoring and Logging

### Health Checks

All services include health check endpoints:
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3

### Logging

- Structured logging with JSON format
- Centralized log collection
- Log rotation and retention policies
- Different log levels for dev/prod

### Metrics

- Prometheus metrics collection
- Grafana dashboards
- Custom application metrics
- Infrastructure monitoring

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000-8004, 5432, 27017, 6379 are available
2. **Memory issues**: Increase Docker memory limit to 8GB+
3. **Permission errors**: Check file permissions and user ownership
4. **Network issues**: Verify Docker network configuration

### Debug Commands

```bash
# Check container status
docker-compose ps

# View service logs
docker-compose logs -f [service]

# Execute commands in container
docker-compose exec [service] bash

# Check resource usage
docker stats

# Inspect networks
docker network ls
docker network inspect enterprise-saas_enterprise-network
```

### Performance Tuning

1. **Database optimization**: Tune PostgreSQL and MongoDB settings
2. **Cache configuration**: Optimize Redis memory usage
3. **Worker processes**: Adjust Gunicorn worker count based on CPU cores
4. **Resource limits**: Set appropriate CPU and memory limits

## Security Considerations

### Container Security

- Non-root user execution
- Read-only root filesystem where possible
- Minimal attack surface with slim images
- Regular security updates

### Network Security

- Internal service communication only
- Rate limiting and request validation
- SSL/TLS encryption for external traffic
- Security headers and CORS configuration

### Data Security

- Encrypted data at rest
- Secure credential management
- Audit logging for compliance
- Regular security scanning

## Production Deployment

### Prerequisites

1. SSL certificates for HTTPS
2. Production database credentials
3. Container registry access
4. Monitoring and alerting setup

### Deployment Steps

1. Build and tag images
2. Push to container registry
3. Configure production environment variables
4. Deploy with production compose file
5. Verify health checks and monitoring
6. Set up backup and disaster recovery

### Scaling

```bash
# Scale specific services
docker-compose up -d --scale api-gateway=3 --scale user-service=2

# Use Docker Swarm for orchestration
docker swarm init
docker stack deploy -c docker-compose.yml -c docker-compose.prod.yml enterprise-saas
```

## Development Workflow

### Local Development

1. Start development environment
2. Make code changes (hot reload enabled)
3. Run tests
4. Check logs and health
5. Stop environment when done

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Build and Deploy
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build images
        run: ./scripts/docker-build.sh build
      - name: Run tests
        run: ./scripts/docker-build.sh test
      - name: Push to registry
        run: ./scripts/docker-build.sh push
```

## Support and Maintenance

### Regular Tasks

- Update base images monthly
- Review and rotate secrets quarterly
- Monitor resource usage and scaling needs
- Backup verification and disaster recovery testing

### Monitoring Alerts

Set up alerts for:
- Service health check failures
- High resource usage
- Database connection issues
- Security events and anomalies

For additional support, refer to the main project documentation or create an issue in the repository.