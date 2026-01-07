"""
OpenAPI configuration and documentation enhancements.
"""
from typing import Dict, Any, List
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI


def custom_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with enhanced documentation."""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Enterprise SaaS Platform API",
        version="1.0.0",
        description="""
# Enterprise SaaS Analytics Platform API

A comprehensive API Gateway for the Enterprise SaaS Analytics Platform providing:

## Features
- **Multi-tenant Architecture**: Complete tenant isolation and management
- **User Management**: Authentication, authorization, and role-based access control
- **Data Processing**: Advanced ETL pipelines and data transformation
- **Machine Learning**: Model training, deployment, and monitoring
- **Rate Limiting**: Intelligent rate limiting based on user roles and endpoints
- **API Versioning**: Backward-compatible API versioning support

## Authentication
All protected endpoints require a valid JWT token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

## Rate Limits
API endpoints have different rate limits based on:
- **User Role**: Admin (5x), Analyst (2x), Viewer (1x), Free (0.5x)
- **Endpoint Type**: Authentication (stricter), Data Upload (moderate), General (standard)

## API Versioning
The API supports versioning through multiple methods:
- **URL Path**: `/api/v1/users` (recommended)
- **Header**: `Accept: application/vnd.api.v1+json`
- **Query Parameter**: `/api/users?version=1`

## Error Handling
All errors follow a consistent format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "request_id": "unique-request-id"
  }
}
```

## Response Headers
All responses include:
- `X-Request-ID`: Unique request identifier for tracing
- `X-RateLimit-*`: Rate limiting information
- `API-Version`: Current API version used
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /api/v1/auth/login"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service authentication"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []}
    ]
    
    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png",
        "altText": "Enterprise SaaS Platform"
    }
    
    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "api-support@example.com"
    }
    
    # Add license information
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "https://api.example.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.example.com",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]
    
    # Add common response schemas
    openapi_schema["components"]["schemas"].update({
        "Error": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Machine-readable error code"
                        },
                        "message": {
                            "type": "string",
                            "description": "Human-readable error message"
                        },
                        "request_id": {
                            "type": "string",
                            "description": "Unique request identifier"
                        }
                    },
                    "required": ["code", "message"]
                }
            },
            "required": ["error"]
        },
        "RateLimitError": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "example": "Rate limit exceeded"
                },
                "code": {
                    "type": "string",
                    "example": "RATE_LIMIT_EXCEEDED"
                },
                "limit": {
                    "type": "integer",
                    "description": "Rate limit threshold"
                },
                "retry_after": {
                    "type": "integer",
                    "description": "Seconds to wait before retrying"
                }
            }
        },
        "HealthCheck": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["healthy", "degraded", "unhealthy"]
                },
                "service": {
                    "type": "string"
                },
                "version": {
                    "type": "string"
                }
            }
        },
        "APIInfo": {
            "type": "object",
            "properties": {
                "api_version": {
                    "type": "string"
                },
                "supported_versions": {
                    "type": "object"
                },
                "services": {
                    "type": "object"
                },
                "authentication": {
                    "type": "object"
                }
            }
        }
    })
    
    # Add common response examples
    openapi_schema["components"]["examples"] = {
        "UnauthorizedError": {
            "summary": "Unauthorized access",
            "value": {
                "error": {
                    "code": "AUTHENTICATION_REQUIRED",
                    "message": "Authentication required",
                    "request_id": "req_123456789"
                }
            }
        },
        "ForbiddenError": {
            "summary": "Insufficient permissions",
            "value": {
                "error": {
                    "code": "INSUFFICIENT_PERMISSIONS",
                    "message": "Insufficient permissions",
                    "request_id": "req_123456789"
                }
            }
        },
        "RateLimitError": {
            "summary": "Rate limit exceeded",
            "value": {
                "error": "Rate limit exceeded",
                "code": "RATE_LIMIT_EXCEEDED",
                "limit": 100,
                "retry_after": 3600
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def add_response_examples(app: FastAPI):
    """Add response examples to all endpoints."""
    
    # Common responses that apply to most endpoints
    common_responses = {
        401: {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "error": {
                            "code": "AUTHENTICATION_REQUIRED",
                            "message": "Authentication required"
                        }
                    }
                }
            }
        },
        403: {
            "description": "Insufficient permissions",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "error": {
                            "code": "INSUFFICIENT_PERMISSIONS", 
                            "message": "Insufficient permissions"
                        }
                    }
                }
            }
        },
        429: {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/RateLimitError"}
                }
            },
            "headers": {
                "X-RateLimit-Limit": {
                    "description": "Rate limit threshold",
                    "schema": {"type": "integer"}
                },
                "X-RateLimit-Remaining": {
                    "description": "Remaining requests in window",
                    "schema": {"type": "integer"}
                },
                "X-RateLimit-Reset": {
                    "description": "Window reset time (Unix timestamp)",
                    "schema": {"type": "integer"}
                },
                "Retry-After": {
                    "description": "Seconds to wait before retrying",
                    "schema": {"type": "integer"}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Error"},
                    "example": {
                        "error": {
                            "code": "INTERNAL_SERVER_ERROR",
                            "message": "An unexpected error occurred"
                        }
                    }
                }
            }
        }
    }
    
    # Add common responses to all routes
    for route in app.routes:
        if hasattr(route, 'responses'):
            route.responses.update(common_responses)


def setup_openapi_documentation(app: FastAPI):
    """Setup comprehensive OpenAPI documentation."""
    
    # Set custom OpenAPI schema generator
    app.openapi = lambda: custom_openapi_schema(app)
    
    # Add response examples
    add_response_examples(app)
    
    return app