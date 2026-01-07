# Enterprise SaaS Analytics Platform

A modular, enterprise-grade SaaS analytics platform built with microservices architecture, demonstrating industry-standard software engineering practices.

## 🏗️ Architecture Overview

This platform transforms the original monolithic Streamlit application into a scalable, production-ready SaaS solution with:

- **Microservices Architecture**: Separate services for different business domains
- **Clean Architecture**: Domain-driven design with clear separation of concerns
- **Multi-tenancy**: Complete tenant isolation and resource management
- **Enterprise Security**: JWT authentication, RBAC, audit logging
- **Scalable Infrastructure**: Async/await, connection pooling, caching
- **Modern DevOps**: Docker, CI/CD, monitoring, observability

## 🚀 Services

### Core Services
- **API Gateway** (Port 8000): Request routing, authentication, rate limiting
- **User Service** (Port 8001): Authentication, authorization, user management
- **Tenant Service** (Port 8002): Multi-tenancy, organization management
- **Data Service** (Port 8003): Data ingestion, validation, ETL pipelines
- **ML Service** (Port 8004): Model training, deployment, predictions

### Infrastructure
- **PostgreSQL**: Primary database for structured data
- **MongoDB**: Document storage for unstructured data
- **Redis**: Caching, sessions, rate limiting

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework with automatic API documentation
- **SQLAlchemy**: Async ORM with PostgreSQL
- **Motor**: Async MongoDB driver
- **Redis**: In-memory data structure store
- **Pydantic**: Data validation and serialization
- **Dependency Injection**: Custom container for loose coupling

### Development
- **Docker & Docker Compose**: Containerization and orchestration
- **Pytest**: Testing framework with async support
- **Black, isort, flake8**: Code formatting and linting
- **MyPy**: Static type checking

## 📁 Project Structure

```
src/
├── shared/                     # Shared components
│   ├── domain/                # Domain models and entities
│   │   ├── base.py           # Base classes (Entity, Repository, etc.)
│   │   └── models.py         # Core domain models
│   ├── infrastructure/        # Infrastructure layer
│   │   ├── database.py       # Database configuration
│   │   ├── repositories.py   # Repository implementations
│   │   ├── models.py         # SQLAlchemy models
│   │   ├── container.py      # Dependency injection
│   │   └── migrations.py     # Database migrations
│   └── application/          # Application services
└── services/                  # Microservices
    ├── api_gateway/          # API Gateway service
    ├── user_service/         # User management
    ├── tenant_service/       # Tenant management
    ├── data_service/         # Data processing
    └── ml_service/           # ML/Analytics
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- PostgreSQL, MongoDB, Redis (or use Docker Compose)

### 1. Clone and Setup
```bash
git clone <repository>
cd enterprise-saas-platform
cp .env.example .env
```

### 2. Install Dependencies
```bash
pip install -r requirements-enterprise.txt
```

### 3. Start Infrastructure (Docker)
```bash
docker-compose up -d postgres mongodb redis
```

### 4. Start Services
```bash
python scripts/start_services.py
```

### 5. Access Services
- **API Gateway**: http://localhost:8000/docs
- **User Service**: http://localhost:8001/docs
- **Tenant Service**: http://localhost:8002/docs
- **Data Service**: http://localhost:8003/docs
- **ML Service**: http://localhost:8004/docs

## 🏗️ Development Workflow

### 1. Domain-Driven Design
- Start with domain models in `src/shared/domain/models.py`
- Define repository interfaces in domain layer
- Implement repositories in infrastructure layer

### 2. Service Development
- Each service has its own FastAPI application
- Services communicate via HTTP APIs
- Use dependency injection for loose coupling

### 3. Database Management
```bash
# Create tables
python -m src.shared.infrastructure.migrations

# Reset database
python -c "from src.shared.infrastructure.migrations import drop_tables; import asyncio; asyncio.run(drop_tables())"
```

### 4. Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## 🔧 Configuration

### Environment Variables
See `.env.example` for all configuration options:
- Database connections
- Service ports
- Security settings
- External service credentials

### Service Configuration
Each service can be configured independently:
- Database connections
- Authentication settings
- Feature flags
- Resource limits

## 🏢 Enterprise Features

### Multi-Tenancy
- Complete data isolation between tenants
- Tenant-specific configurations and branding
- Resource quotas and usage tracking

### Security
- JWT-based authentication
- Role-based access control (RBAC)
- Input validation and sanitization
- Audit logging for compliance

### Scalability
- Async/await throughout the stack
- Connection pooling and caching
- Horizontal scaling support
- Load balancing ready

### Observability
- Structured logging
- Health check endpoints
- Metrics collection ready
- Distributed tracing support

## 📊 Migration from Original App

The original `app.py` functionality is being migrated to:

1. **Data Upload & Processing** → Data Service
2. **ML Model Training** → ML Service
3. **User Management** → User Service
4. **Dashboard & Visualization** → Dashboard Service (future)
5. **Authentication** → User Service + API Gateway

## 🚀 Deployment

### Docker Deployment
```bash
# Build all services
docker-compose build

# Start full stack
docker-compose up -d
```

### Production Considerations
- Use environment-specific configurations
- Set up proper secrets management
- Configure load balancers
- Set up monitoring and alerting
- Implement backup strategies

## 🧪 Testing Strategy

### Unit Tests
- Domain model validation
- Repository implementations
- Service logic

### Integration Tests
- Database operations
- Service-to-service communication
- API endpoints

### Property-Based Tests
- Security properties (authentication, authorization)
- Data consistency and integrity
- API contract compliance

## 📈 Performance

### Optimization Features
- Async database operations
- Connection pooling
- Redis caching
- Lazy loading
- Pagination support

### Monitoring
- Health check endpoints
- Performance metrics
- Error tracking
- Usage analytics

## 🤝 Contributing

1. Follow the established architecture patterns
2. Write tests for new functionality
3. Use type hints throughout
4. Follow code formatting standards (Black, isort)
5. Update documentation

## 📚 Learning Objectives

This project demonstrates:
- **Microservices Architecture**: Service decomposition and communication
- **Clean Architecture**: Separation of concerns and dependency inversion
- **Domain-Driven Design**: Rich domain models and bounded contexts
- **Enterprise Patterns**: Repository, Unit of Work, Dependency Injection
- **Modern Python**: Async/await, type hints, Pydantic models
- **DevOps Practices**: Containerization, CI/CD, monitoring
- **Security**: Authentication, authorization, input validation
- **Scalability**: Caching, connection pooling, horizontal scaling

Perfect for demonstrating enterprise software development skills in job interviews and real-world projects!