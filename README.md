# 📦 Olist Order Delivery Status Prediction
## Production-Grade MLOps & FastAPI Microservice

[![CI/CD Pipeline](https://github.com/rrody8619/Olist_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/rrody8619/Olist_Project/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end, production-ready MLOps microservice designed to predict e-commerce order delivery delays using the Brazilian Olist dataset. This system integrates automated data validation, robust feature engineering, machine learning modeling, DVC artifact tracking, and a containerized FastAPI backend.

---

## 🏗️ Project Architecture

```text
Olist_Project/
├── .github/workflows/    # CI/CD automation pipelines (GitHub Actions)
├── app/                  # FastAPI Application layer
│   └── main.py           # API Endpoints, prediction routing & logging
├── config/               # System configurations and parameters
├── data/                 # Raw and processed datasets (versioned via DVC)
├── logs/                 # Inference & execution logs
├── models/               # Trained ML models and preprocessing artifacts
├── src/                  # Core source modules
│   ├── data_creation.py
│   ├── feature_engineering.py
│   ├── labeling.py
│   ├── logger.py
│   ├── predict.py
│   └── validation.py     # Data quality checks
├── tests/                # Pytest automation suite
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container orchestration
├── requirements.txt      # Project dependencies
└── pytest.ini            # Pytest configuration
🚀 Key Features
Asynchronous FastAPI Service: High-performance RESTful API endpoints supporting single (/predict) and batch inference.

Strict Data Validation: Runtime payload verification to ensure data integrity before entering the pipeline.

Structured Logging: Centralized tracking for prediction requests, latency, and system execution.

Automated CI/CD Workflows: GitHub Actions pipeline validating the complete test suite and building container images on every push.

Containerized Deployment: Fully isolated, reproducible runtime environments configured via Docker and Docker Compose.

🏁 Quick Start Guide
1. Local Execution
Bash
# Install dependencies
pip install -r requirements.txt

# Run test suite
python -m pytest tests/ -v

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
2. Run via Docker
Bash
# Build the image
docker build -t olist-inference-api:latest .

# Run the container
docker run -p 8000:8000 olist-inference-api:latest
3. Run via Docker Compose
Bash
docker compose up --build
🌐 API Documentation: Access the interactive Swagger UI locally at http://localhost:8000/docs

🛠️ Technologies Used
Language: Python 3.10+

Machine Learning & Data: Scikit-learn, Pandas, NumPy, Joblib

Data Versioning: DVC (Data Version Control)

Backend API: FastAPI, Uvicorn

Testing & Quality: Pytest

Containerization: Docker, Docker Compose

CI/CD: GitHub Actions



```powershell
git add README.md
git commit -m "docs: upgrade README.md with professional MLOps structure"
git push origin main