# 🚀 Backend Deployment Guide

## Quick Deployment Options (Ranked by Campus Placement Impact)

### 🥇 **Option 1: Render.com (RECOMMENDED)**
**Perfect for:** Campus placements, demos, interviews
**Time:** 10 minutes
**Cost:** Free
**Benefits:** Automatic HTTPS, easy setup, great for demos

#### Step-by-Step Render Deployment:

1. **Prepare Repository**
   ```bash
   git add render.yaml Procfile requirements-backend.txt
   git commit -m "Add backend deployment configuration"
   git push origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub
   - Click **"New +"** → **"Web Service"**
   - Connect your repository
   - Configure:
     - **Name**: `astralytiq-backend`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements-backend.txt`
     - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Free

3. **Environment Variables**
   Add these in Render dashboard:
   ```
   ENVIRONMENT=production
   JWT_SECRET=AstralytiQ-Production-JWT-Secret-Key-2025-Campus-Placement
   CORS_ORIGINS=https://astralytiq-platform.streamlit.app,http://localhost:8501
   ```

4. **Update Streamlit Secrets**
   Replace `astralytiq-backend` with your actual Render service name:
   ```toml
   API_BASE_URL = "https://your-service-name.onrender.com"
   ```

5. **Test Deployment**
   - Visit: `https://your-service-name.onrender.com/docs`
   - Should see FastAPI Swagger documentation

---

### 🥈 **Option 2: Railway.app**
**Perfect for:** Advanced deployments, PostgreSQL integration
**Time:** 15 minutes
**Cost:** Free tier available

#### Railway Deployment:
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically detects FastAPI
4. Add environment variables
5. Get deployment URL

---

### 🥉 **Option 3: Google Cloud Run**
**Perfect for:** Enterprise demonstrations, scalability showcase
**Time:** 20 minutes
**Cost:** Free tier (1M requests/month)

#### GCP Deployment:
```bash
# Build and deploy
gcloud run deploy astralytiq-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🔧 **Post-Deployment Steps**

### 1. Update Frontend Configuration
Update your Streamlit Cloud secrets with the deployed backend URL:

```toml
# In Streamlit Cloud Dashboard → Settings → Secrets
API_BASE_URL = "https://your-backend-url.com"
USER_SERVICE_URL = "https://your-backend-url.com/api/v1/users"
DATA_SERVICE_URL = "https://your-backend-url.com/api/v1/data"
ML_SERVICE_URL = "https://your-backend-url.com/api/v1/ml"
DASHBOARD_SERVICE_URL = "https://your-backend-url.com/api/v1/dashboard"
```

### 2. Test Full-Stack Integration
1. Visit your Streamlit app
2. Login with demo credentials
3. Look for "🟢 Backend Connected" indicator
4. Test API integration features

### 3. Verify API Documentation
Visit: `https://your-backend-url.com/docs`
Should show interactive FastAPI documentation

---

## 🎯 **Campus Placement Benefits**

### What This Demonstrates:
✅ **Full-Stack Development**: Frontend + Backend integration
✅ **Cloud Deployment**: Production deployment experience  
✅ **API Design**: RESTful API with OpenAPI documentation
✅ **Authentication**: JWT-based security implementation
✅ **Database Integration**: SQLite with SQLAlchemy ORM
✅ **DevOps Practices**: CI/CD, environment configuration
✅ **Real-time Features**: WebSocket connections
✅ **Production Readiness**: CORS, logging, error handling

### Interview Talking Points:
- "Deployed FastAPI backend to cloud with automatic scaling"
- "Implemented JWT authentication with secure token management"
- "Built RESTful APIs with auto-generated OpenAPI documentation"
- "Configured CORS for secure cross-origin requests"
- "Used environment variables for production configuration"
- "Integrated real-time WebSocket connections"

---

## 🚨 **Troubleshooting**

### Common Issues:

1. **CORS Errors**
   - Ensure your Streamlit app URL is in CORS_ORIGINS
   - Check browser console for specific CORS messages

2. **Authentication Failures**
   - Verify JWT_SECRET matches between frontend and backend
   - Check token expiration settings

3. **Database Issues**
   - SQLite file permissions in deployment environment
   - Database initialization on first startup

4. **Deployment Failures**
   - Check build logs in deployment platform
   - Verify requirements-backend.txt includes all dependencies
   - Ensure Python version compatibility

### Debug Commands:
```bash
# Test backend locally
uvicorn backend.main:app --reload --port 8081

# Check API health
curl https://your-backend-url.com/health

# Test authentication
curl -X POST https://your-backend-url.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@astralytiq.com","password":"admin123"}'
```

---

## 🎉 **Success Checklist**

- [ ] Backend deployed and accessible
- [ ] API documentation available at `/docs`
- [ ] Health check endpoint responding
- [ ] CORS configured for Streamlit app
- [ ] JWT authentication working
- [ ] Database initialized with demo data
- [ ] Frontend shows "Backend Connected" status
- [ ] Full-stack integration functional

**Your enterprise-grade MLOps platform is now fully deployed and ready for campus placement demonstrations!** 🚀