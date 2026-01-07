# Multi-stage build for ML Service
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-enterprise.txt .
RUN pip install --no-cache-dir --user -r requirements-enterprise.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy source code
COPY src/ ./src/
COPY .env.example .env

# Create storage directory for models with proper permissions
RUN mkdir -p /app/storage/models && \
    chown -R appuser:appuser /app

# Set Python path and user PATH
ENV PYTHONPATH=/app
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8004/health')" || exit 1

# Run the application
CMD ["uvicorn", "src.services.ml_service.main:app", "--host", "0.0.0.0", "--port", "8004"]