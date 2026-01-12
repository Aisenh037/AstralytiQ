# 🚀 Deployment Guide Summary

AstralytiQ supports multiple deployment strategies for different environments.

## Quick Deployment Options

### 🐳 Docker (Recommended for Local/Staging)
```bash
# Full stack with backend
docker-compose --profile full-stack up -d

# Frontend only
docker-compose up -d
```

### ☁️ Google Cloud Platform
```bash
# Deploy backend to Cloud Run
./scripts/deploy_gcp.sh

# Or use Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

### 🎈 Streamlit Cloud (Frontend Only)
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from your fork
4. Add secrets in Streamlit dashboard

### 🔧 Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Environment Variables

Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

Required variables:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_KEY` - Supabase anon/public key
- `JWT_SECRET_KEY` - Secret for JWT token signing

## Detailed Guides

For comprehensive deployment instructions, see:
- [GCP Full Deployment](docs/deployment_gcp_full.md)
- [Backend Deployment](docs/deployment_backend.md)
- [Streamlit Cloud](docs/deployment_streamlit.md)
