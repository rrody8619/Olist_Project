# Olist Order Delivery Prediction Service

An end-to-end, production-ready MLOps microservice designed to predict Olist order delivery delays. Built with FastAPI, containerized using Docker, validated with Great Expectations, and fully tested with automated CI/CD workflows.

---

## 🛠️ Project Architecture & Structure

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
│   └── validation.py     # Great Expectations data quality checks
├── tests/                # Pytest automation suite
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container orchestration
├── requirements.txt      # Project dependencies
└── pytest.ini            # Pytest configuration


🚀 Key Features & Best Practices
FastAPI Web Service: Asynchronous, RESTful API endpoints for single (/predict) and batch (/predict-batch) inference.

Data Validation: Runtime payload verification via Great Expectations to prevent invalid features from entering the inference pipeline.

Structured Logging: Granular prediction, latency, and system performance logging.

Automated Testing: 100% passing pytest suite covering feature engineering, model inference, and API endpoints.

CI/CD Integration: Automated GitHub Actions pipeline verifying test suite execution and Docker container builds on every commit.

Containerization: Isolated, reproducible runtime using Docker and docker-compose.

🏁 Quick Start Guide
1. Run Locally via Python
Bash
# Install dependencies
pip install -r requirements.txt

# Execute test suite
python -m pytest tests/

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
2. Run via Docker
Bash
# Build Docker image
docker build -t olist-inference-api:latest .

# Run Container
docker run -p 8000:8000 olist-inference-api:latest
3. Run via Docker Compose
Bash
docker compose up --build
Access Swagger UI documentation at: http://localhost:8000/docs

🧪 Running Validation & Tests
To execute the full test suite manually:

Bash
python -m pytest tests/ -v

---

#### **2. تنفيذ أسباب الإضافة والـ Commit مجدداً**
بعد إنشائه وحفظه، شغلي الأوامر في الـ Terminal:

```powershell
git add .
git commit -m "feat: complete Task 3 with testing, validation, CI/CD, and documentation"
git push origin main