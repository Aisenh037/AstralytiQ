# ⚡ AstralytiQ - Enterprise MLOps Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://astralytiq.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **🎯 Industry-grade MLOps platform showcasing enterprise development practices**  
> **Perfect for SDE/DE Campus Placements** - Demonstrates Clean Architecture, Production-Ready Code Quality, Enterprise Security & Authentication, Scalable System Design, and Modern DevOps Practices.

## 🚀 Live Demo

**🌐 [Try AstralytiQ Live](https://sales-forecast-app-aisenh037.streamlit.app)**

### 🔐 Demo Credentials
```
🔑 Admin Access: admin@astralytiq.com / admin123
👨‍💻 Data Scientist: data.scientist@astralytiq.com / ds123
📊 Business Analyst: analyst@astralytiq.com / analyst123
```

---

## 🏗️ Architecture

AstralytiQ follows a decoupled, microservices-ready architecture:

- **Frontend**: Streamlit-based Enterprise UI with custom CSS and real-time Plotly charts.
- **Backend**: FastAPI Forecasting Engine (Deployed via Cloud Run).
- **Security**: JWT-based Authentication & Authorization.
- **Data**: Support for CSV, PostgreSQL, and scalable MLOps pipelines.

---

## 🚀 Quick Start

### Local Development
```bash
# Clone and setup
git clone https://github.com/yourusername/astralytiq.git
cd astralytiq
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install and Run
pip install -r requirements.txt
streamlit run app.py
```

### Docker Deployment
```bash
docker-compose up -d
```

---

## 📚 Documentation

Detailed guides are located in the [docs/](docs/) directory:

- [🚀 GCP Quick Start](docs/quickstart_gcp.md)
- [🏗️ Full GCP Deployment Guide](docs/deployment_gcp_full.md)
- [☁️ Cloud Run Backend Guide](docs/deployment_backend.md)
- [🎈 Streamlit Cloud Deployment](docs/deployment_streamlit.md)
- [🛠️ Development Guide](docs/development.md)
- [🤝 Contributing](docs/contributing.md)

---

## 🧪 Testing

```bash
pytest tests/
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">
**⚡ Built with passion for enterprise-grade MLOps**
</div>
