from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import traceback

from database import get_store_context

app = FastAPI(
    title="Rossmann Demand Intelligence API",
    description="API with advanced error tracking",
    version="2.1.0"
)

# 1. LOAD MODELS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'backend/models'))

try:
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'rossmann_preprocessor.joblib'))
    xgb_models = joblib.load(os.path.join(MODEL_DIR, 'xgboost_clusters.joblib'))
except Exception as e:
    print(f"❌ Error loading models: {e}")

class SalesRequest(BaseModel):
    Store: int
    Date: str          
    DayOfWeek: int     
    Promo: int         
    StateHoliday: str  
    SchoolHoliday: int 
    Open: int = 1      

@app.post("/predict")
def predict_sales(request: SalesRequest):
    if request.Open == 0:
        return {"Status": "Closed", "Predicted_Sales": 0.0}

    # --- DETECTIVE STEP 1: Database Extraction ---
    try:
        df_history = get_store_context(request.Store, request.Date, days_back=35)
        if df_history.empty:
            raise ValueError("History is empty.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🚨 ERROR IN DATABASE STEP: {e}")

    # --- DETECTIVE STEP 2: Extracting Static Variables ---
    try:
        static_cols = ['StoreType', 'Assortment', 'CompetitionDistance', 
                       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                       'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
        
        # Check if columns actually exist in the DataFrame returned by DB
        missing = [c for c in static_cols if c not in df_history.columns]
        if missing:
            raise KeyError(f"Missing columns from DB: {missing}. Available columns: {list(df_history.columns)}")
            
        static_vars = df_history.iloc[-1][static_cols].to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🚨 ERROR EXTRACTING COLUMNS: {e}")

    # --- DETECTIVE STEP 3: Building Final DataFrame ---
    try:
        target_day_dict = request.model_dump()
        target_day_dict.update(static_vars)
        target_day_dict['Sales'] = 0      
        target_day_dict['Customers'] = 0  
        
        df_target = pd.DataFrame([target_day_dict])
        df_full = pd.concat([df_history, df_target], ignore_index=True)
        
        # Fix for Pandas Integer/NaN issue
        df_full = df_full.fillna(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🚨 ERROR MERGING DATA: {e}")

    # --- DETECTIVE STEP 4: Preprocessor ---
    try:
        X_proc = preprocessor.transform(df_full)
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"🚨 ERROR IN PREPROCESSOR: {e} | Trace: {error_trace}")

# --- DETECTIVE STEP 5: XGBoost Prediction ---
    try:
        X_final = X_proc.iloc[[-1]].copy()

        # 1. DEFINIR EL ORDEN EXACTO (Copiado de tu error de mismatch)
        expected_columns = [
            'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
            'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2', 
            'Cluster', 'sales_lag_7', 'sales_lag_14', 'sales_lag_21', 'sales_lag_28', 
            'sales_rolling_mean_7', 'Year', 'Month', 'Day', 'WeekOfYear', 
            'Quarter', 'Is_Weekend', 'IsMonthStart', 'IsMonthEnd', 
            'Month_Sin', 'Month_Cos', 'CompetitionOpenMonths', 
            'Promo2OpenMonths', 'IsPromo2ActiveMonth', 'Store_Avg_Sales'
        ]

        # 2. ASEGURAR QUE TODAS EXISTAN (Si falta alguna, la creamos con 0)
        for col in expected_columns:
            if col not in X_final.columns:
                X_final[col] = 0

        # 3. REORDENAR Y FILTRAR (Solo las que el modelo espera y en ese orden)
        X_final = X_final[expected_columns]

        # 4. CONVERTIR A FLOAT (XGBoost ama los floats)
        X_final = X_final.astype(float)

        # 5. PREDICCIÓN
        cluster = int(X_final['Cluster'].iloc[0])
        baseline = float(X_final['Store_Avg_Sales'].iloc[0])

        xgb_model = xgb_models.get(cluster)
        
        # Predicción del residuo
        residual_pred = float(xgb_model.predict(X_final)[0])
        
        # Resultado final: Baseline + Residuo
        final_sales = max(baseline + residual_pred, 0.0)

        return {
            "Store": request.Store,
            "Date": request.Date,
            "Predicted_Sales": round(final_sales, 2),
            "Status": "Success",
            "Model_Info": {"Cluster": cluster, "Baseline": round(baseline, 2)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🚨 FINAL ERROR: {e}")