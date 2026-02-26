# 📈 Rossmann Demand Intelligence (RDI): Retail Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Model-Time%20Series%20Forecasting-8A2BE2)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-316192)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Docker Compose](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

## 💼 Business Problem
In the retail and pharmacy sector, predicting daily sales is a critical operational challenge.
* **Overstocking:** Ties up working capital in warehouses and increases the risk of product expiration (waste).
* **Understocking:** Leads to empty shelves, lost revenue, and poor customer satisfaction.
* **Staffing Inefficiencies:** Without knowing the expected foot traffic, store managers struggle to optimize employee shifts, either overpaying for idle time or understaffing during rush hours.

## 🎯 The Solution: Demand Intelligence
**Rossmann Demand Intelligence (RDI)** is a predictive analytics platform designed for store managers and regional directors to forecast daily sales up to 6 weeks in advance across 1,115 stores.



* **🧠 Forecasting Engine:** A robust Time Series Machine Learning model that captures weekly seasonality, holiday effects, competitor distance, and the impact of active marketing promotions.
* **🏬 Store Manager Dashboard (UI):** An interactive portal where managers can visualize upcoming demand, review historical accuracy (Predicted vs. Actual), and plan inventory/staffing accordingly.
* **🗄️ Historical Memory:** A PostgreSQL database that continuously logs the AI's predictions alongside the real sales data to monitor model drift and performance over time.

## 🏗️ Architecture & Tech Stack
This project upgrades a standard ML pipeline into a **3-Tier Microservices Architecture**, adding a persistent database layer for production-grade tracking.



* **Data Engineering:** Pandas, Feature Engineering (Lags, Rolling Means, Time Features).
* **Modeling:** Advanced Regression for Time Series (e.g., XGBoost, LightGBM).
* **Backend/API (The Brain):** FastAPI.
* **Database (The Memory):** PostgreSQL (Storing predictions and actuals).
* **Frontend (The Face):** Streamlit.
* **Infrastructure:** Docker & Docker Compose.

## 📊 Project Structure

```text
rossmann_demand_intelligence/
├── data/                  # Raw and processed datasets (Git ignored)
├── notebooks/             # EDA, Feature Engineering & Model Training (Jupyter)
├── backend/               # FastAPI Microservice
│   ├── models/            # Serialized trained models (.joblib)
│   ├── database.py        # PostgreSQL connection and CRUD operations
│   ├── main.py            # API routing and model inference logic
│   ├── requirements.txt   # Backend dependencies
│   └── Dockerfile         # Backend container config
├── frontend/              # Streamlit Application
│   ├── assets/            # UI Images, logos, and styling assets
│   ├── app.py             # UI and API connection logic
│   ├── requirements.txt   # Frontend dependencies
│   └── Dockerfile         # Frontend container config
├── init.sql               # Database initialization script (Table creation)
├── docker-compose.yml     # Multi-container orchestration (DB + API + UI)
├── .gitignore             # Git ignore rules
├── .dockerignore          # Docker ignore rules
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
* **Database Port:** `5432` (Accessible via DBeaver/PgAdmin using credentials in docker-compose)

## 🔮 Roadmap & Future Improvements (v2.0 & Beyond)

As we scale RDI, the following features are planned:

### 1. Deep Learning Transition 🧠

* Upgrade the traditional ML baseline to Deep Learning architectures specifically designed for sequences, such as **LSTM (Long Short-Term Memory)** or **Temporal Fusion Transformers (TFT)** using PyTorch.

### 2. MLOps & Experiment Tracking 🛠️

* Implement **MLflow** to systematically track hyperparameters, model versions, and evaluation metrics (MAPE, RMSE) during the training phase.
* Set up **Apache Airflow** to automatically ingest weekly sales data, retrain the model if data drift is detected, and push the new weights to the FastAPI backend.

### 3. Business Intelligence Features 📈

* **ROI Impact Calculator:** Translate the MAPE (Mean Absolute Percentage Error) improvement directly into Euros saved by reducing overstock.
* **What-If Scenarios:** Allow managers to simulate interventions in the UI: *"What if I run a promotion next Tuesday? How much will sales increase?"*

---

*Built with ❤️ by Pablo.*