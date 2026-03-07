# 📈 Rossmann Demand Intelligence (RDI): Retail Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Model-Time%20Series%20Forecasting-8A2BE2)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-316192)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Docker Compose](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

## 💼 Business Problem

In the retail sector, predicting daily sales across thousands of stores is a critical operational challenge.

* **Overstocking:** Ties up working capital in warehouses and increases the risk of product expiration (waste).
* **Understocking:** Leads to empty shelves, lost revenue, and poor customer satisfaction.
* **Staffing Inefficiencies:** Without knowing the expected foot traffic, store managers struggle to optimize employee shifts, either overpaying for idle time or understaffing during rush hours.

## 🎯 The Solution: Demand Intelligence

**Rossmann Demand Intelligence (RDI)** is a predictive analytics platform designed for store managers and regional directors to forecast daily sales up to 6 weeks in advance across 1,115 stores.

* **🧠 Hybrid Forecasting Engine:** A robust pipeline that combines the trend-capturing power of **Prophet** (Baseline) with the non-linear pattern recognition of **XGBoost** (Residual Learning), trained on clustered store data to capture specific behaviors (promotions, holidays, competitor distance).
* **🏬 Store Manager Dashboard (UI):** A dark-mode, interactive SaaS portal where managers can visualize upcoming demand, simulate conditions (open/closed, local promos), and review historical accuracy by comparing forecasts against the *Holdout Validation Set* (Actual Sales).
* **🗄️ Historical Memory:** A local SQLite database that stores static store profiles and historical ground truth data to evaluate model performance visually without data leakage.

## 🏗️ Architecture & Tech Stack

This project upgrades a standard ML script into a **Containerized Microservices Architecture**.

* **Data Engineering:** Pandas, Scikit-Learn Pipelines (Imputation, Scaling, Target Encoding).
* **Modeling:** XGBoost Regressor grouped by Store Clusters + Meta Prophet Baseline.
* **Backend/API (The Brain):** FastAPI (REST API with Pydantic validation).
* **Database (The Memory):** SQLite (Serving store features and historical Actuals to the UI).
* **Frontend (The Face):** Streamlit (Plotly for interactive visualization, custom CSS for UI/UX).
* **Infrastructure:** Docker & Docker Compose.

## 📊 Project Structure

```text
rossmann_demand_intelligence/
├── data/                  # Raw and processed datasets (Git ignored)
├── notebooks/             # EDA, Feature Engineering & Model Training (Jupyter)
├── backend/               # FastAPI Microservice
│   ├── models/            # Serialized trained models (.joblib)
│   ├── main.py            # API routing and model inference logic
│   ├── rossmann.db        # SQLite Database (Store profiles & Actual Sales)
│   ├── requirements.txt   # Backend dependencies
│   └── Dockerfile         # Backend container config
├── frontend/              # Streamlit Application
│   ├── assets/            # UI Images, logos, and styling assets
│   ├── app.py             # UI and API connection logic
│   ├── requirements.txt   # Frontend dependencies
│   └── Dockerfile         # Frontend container config
├── docker-compose.yml     # Multi-container orchestration (API + UI)
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation

```

## 🚀 How to Run (Docker Compose)

1. **Build and run the cluster:**

```bash
docker-compose up --build

```

2. **Access the ecosystem:**

* **Store Manager UI:** `http://localhost:8501`
* **API Documentation:** `http://localhost:8000/docs`

*(Note: The UI is currently locked to evaluate the validation set starting on June 20, 2015, to ensure honest model assessment without data leakage).*

## 🔮 Roadmap & Future Improvements (v2.0 & Beyond)

While the current architecture is robust, the predictive engine and deployment strategy can be further optimized:

### 1. Modeling Optimization 📉

* **Error Reduction:** The current model smooths out extreme demand peaks, leading to "medium" sized errors on highly volatile days. Future iterations will explore custom loss functions (e.g., Quantile Regression for Confidence Intervals) or deep learning sequence models (LSTMs) to better capture extreme outliers and sudden spikes in demand.

### 2. Continuous Learning & Cloud Deployment ☁️

* **Live Deployment:** Move the application from a local Docker environment to a managed cloud service (e.g., Google Cloud Run or AWS ECS).
* **Online Learning (CI/CD for ML):** Upgrade the static SQLite database to a cloud-managed PostgreSQL instance. Implement a feedback loop where the system ingests new daily sales data, evaluates data drift, and automatically triggers model retraining (using Airflow or GitHub Actions) to continuously learn and adapt to recent market changes.

### 3. Business Intelligence Features 📈

* **ROI Impact Calculator:** Translate the model's RMSE improvement directly into Euros saved by reducing overstock.
* **Intervals of Confidence:** Upgrade the UI to show not just the expected sales, but a shaded area representing the "best case" and "worst case" scenarios for safer inventory planning.

---

*Built with ❤️ by Pablo.*