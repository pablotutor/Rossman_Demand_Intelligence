import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Data Cleaning and Merging

def clean_attributes(df_attributes: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the store attributes dataset, handling missing values 
    for competition and promotions.
    """
    df_attr = df_attributes.copy()
    
    # 1. CompetitionDistance: If null, we assume the competitor is far away.
    # We use twice the maximum distance found in the dataset.
    max_dist = df_attr['CompetitionDistance'].max()
    df_attr['CompetitionDistance'] = df_attr['CompetitionDistance'].fillna(max_dist * 2)

    # 2. CompetitionOpenSince: Fill missing dates with the start of the training period.
    # We use 2013 and Month 1 as integers.
    df_attr['CompetitionOpenSinceMonth'] = df_attr['CompetitionOpenSinceMonth'].fillna(1).astype(int)
    df_attr['CompetitionOpenSinceYear'] = df_attr['CompetitionOpenSinceYear'].fillna(2013).astype(int)

    # 3. Promo2Since: If null, we set week and year to 0 to indicate no participation.
    df_attr['Promo2SinceWeek'] = df_attr['Promo2SinceWeek'].fillna(0).astype(int)
    df_attr['Promo2SinceYear'] = df_attr['Promo2SinceYear'].fillna(0).astype(int)
    
    # PromoInterval: Fill with '0' as a string to maintain categorical consistency
    df_attr['PromoInterval'] = df_attr['PromoInterval'].fillna('0')

    return df_attr
    

def clean_data(df_sales: pd.DataFrame, df_attributes: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning pipeline: filters open stores, merges data, 
    and ensures correct data types.
    """
    # Filter only open stores and drop the 'Open' column as it becomes a constant
    df_sales_clean = df_sales.copy()
    
    #df_sales_clean = df_sales[df_sales["Open"] == 1].copy()
    #df_sales_clean = df_sales_clean.drop("Open", axis=1)
    
    # Clean store attributes
    df_attr_cleaned = clean_attributes(df_attributes)
    
    # Merge datasets on 'Store'
    df_clean = df_sales_clean.merge(df_attr_cleaned, on="Store", how="inner")
    
    # 🔥 DATA TYPE TRANSFORMATION
    # Ensure Date is a datetime object for time feature extraction
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    # Ensure IDs and categorical-numeric flags are integers
    df_clean['Store'] = df_clean['Store'].astype(int)
    df_clean['DayOfWeek'] = df_clean['DayOfWeek'].astype(int)
    
    df_clean = df_clean.set_index('Date').sort_index()
    
    return df_clean



# Data Transformation
def prepare_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms cleaned data into features ready for Machine Learning.
    Calculates competition/promo seniority, extracts cyclical time features, 
    and generates business-specific flags.
    """
    df_prep = df.copy()
    
    if 'Date' in df_prep.columns:
        df_prep['Date'] = pd.to_datetime(df_prep['Date'])
        df_prep = df_prep.set_index('Date')

    # 1. TIME FEATURES EXTRACTION
    # Using the DatetimeIndex to extract calendar information
    df_prep['Year'] = df_prep.index.year
    df_prep['Month'] = df_prep.index.month
    df_prep['Day'] = df_prep.index.day
    df_prep['WeekOfYear'] = df_prep.index.isocalendar().week.astype(int)
    df_prep['DayOfWeek'] = df_prep.index.dayofweek + 1 # 1=Monday, 7=Sunday
    
    # Business Quarter (Q1 to Q4)
    df_prep['Quarter'] = df_prep.index.quarter
    
    # Weekend Flag (1 for Sat/Sun, 0 for weekdays)
    df_prep['Is_Weekend'] = df_prep['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)

    # Payday proximity (Paydays are usually at the start/end of the month)
    df_prep['IsMonthStart'] = df_prep.index.is_month_start.astype(int)
    df_prep['IsMonthEnd'] = df_prep.index.is_month_end.astype(int)

    # Cyclical Encoding for Months
    # This helps the model understand that Month 12 and Month 1 are close in time
    df_prep['Month_Sin'] = np.sin(2 * np.pi * df_prep['Month'] / 12.0)
    df_prep['Month_Cos'] = np.cos(2 * np.pi * df_prep['Month'] / 12.0)

    # 2. COMPETITION FEATURES
    # Calculate months elapsed since competition opened
    if 'CompetitionOpenSinceYear' in df_prep.columns and 'CompetitionOpenSinceMonth' in df_prep.columns:
        df_prep['CompetitionOpenMonths'] = 12 * (df_prep['Year'] - df_prep['CompetitionOpenSinceYear']) + \
                                        (df_prep['Month'] - df_prep['CompetitionOpenSinceMonth'])
        # Opcional: Rellenar con 0 si el resultado es negativo (competencia que aún no abre)
        df_prep['CompetitionOpenMonths'] = df_prep['CompetitionOpenMonths'].apply(lambda x: x if x > 0 else 0)
    else:
        # Si no están, creamos la columna con 0 para que el modelo no explote después
        df_prep['CompetitionOpenMonths'] = 0

    # Log transformation for Distance (to handle high skewness)
    if 'CompetitionDistance' in df_prep.columns:
        max_dist = df_prep['CompetitionDistance'].max()
        df_prep['CompetitionDistance'] = df_prep['CompetitionDistance'].fillna(max_dist * 2)
        df_prep['CompetitionDistance'] = np.log1p(df_prep['CompetitionDistance'])

    # 3. PROMO2 FEATURES
    # Calculate months elapsed since Promo2 started
    # 3. PROMO2 FEATURES (Versión Protegida)
    if 'Promo2SinceYear' in df_prep.columns and 'Promo2SinceWeek' in df_prep.columns:
        df_prep['Promo2OpenMonths'] = 12 * (df_prep['Year'] - df_prep['Promo2SinceYear']) + \
                                           (df_prep['WeekOfYear'] - df_prep['Promo2SinceWeek']) / 4.2
        df_prep['Promo2OpenMonths'] = df_prep['Promo2OpenMonths'].apply(lambda x: x if x > 0 else 0)
    else:
        df_prep['Promo2OpenMonths'] = 0

    # Check if Promo2 is ACTIVE in the current month
    if 'PromoInterval' in df_prep.columns and 'Promo2' in df_prep.columns:
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                     7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df_prep['CurrentMonthStr'] = df_prep['Month'].map(month_map)
        
        df_prep['IsPromo2ActiveMonth'] = df_prep.apply(
            lambda x: 1 if (x['Promo2'] == 1 and str(x['CurrentMonthStr']) in str(x['PromoInterval'])) else 0, 
            axis=1
        )
    else:
        df_prep['IsPromo2ActiveMonth'] = 0

    # 4. CATEGORICAL ENCODING (Label Encoding)
    # StoreType: a, b, c, d -> 1, 2, 3, 4
    store_type_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    if 'StoreType' in df_prep.columns:
        # Usamos fillna('a') por si acaso viene un nulo de la BD
        df_prep['StoreType'] = df_prep['StoreType'].fillna('a').map(lambda x: store_type_map.get(str(x).lower(), 1))
    else:
        df_prep['StoreType'] = 1

    # Assortment: a, b, c -> 1, 2, 3
    assort_map = {'a': 1, 'b': 2, 'c': 3}
    if 'Assortment' in df_prep.columns:
        df_prep['Assortment'] = df_prep['Assortment'].fillna('a').map(lambda x: assort_map.get(str(x).lower(), 1))
    else:
        df_prep['Assortment'] = 1

    # StateHoliday: 0, a, b, c -> 0, 1, 2, 3
    holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    if 'StateHoliday' in df_prep.columns:
        # Forzamos a string y limpiamos espacios
        df_prep['StateHoliday'] = df_prep['StateHoliday'].astype(str).str.strip().map(lambda x: holiday_map.get(x, 0))
    else:
        df_prep['StateHoliday'] = 0

    # 5. CLEAN UP
    # Remove temporary columns and leakage variables
    cols_to_drop = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                    'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'CurrentMonthStr']
    
    existing_cols_to_drop = [c for c in cols_to_drop if c in df_prep.columns]
    
    # Avoid Data Leakage: 'Customers' is not known at the time of prediction
    if 'Customers' in df_prep.columns:
        existing_cols_to_drop.append('Customers')
        
    df_prep = df_prep.drop(columns=existing_cols_to_drop)

    return df_prep


def add_time_series_features(df):
    df = df.sort_values(['Store', 'Date'])
    
    # 1. Lags (Retardos) - ¿Qué pasó hace 7, 14, 21 y 28 días?
    # Usamos múltiplos de 7 para capturar la estacionalidad semanal
    for lag in [7, 14, 21, 28]:
        df[f'sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
    
    # 2. Moving Averages (Medias Móviles) - ¿Cuál es la tendencia de las últimas 2 semanas?
    df['sales_rolling_mean_7'] = df.groupby('Store')['sales_lag_7'].transform(lambda x: x.rolling(window=7).mean())
    
    # Rellenamos los NaNs que generan los desplazamientos con 0 o la media
    df = df.fillna(0)
    return df


# --- The base Feature Egnineering CLASS ---

class RossmannDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df_attributes, n_clusters=5):
        self.df_attributes = df_attributes
        self.n_clusters = n_clusters
        self.store_means = {}
        self.global_mean = 0
        self.store_cluster_map = {} # Aquí guardaremos el cluster de cada tienda

    def fit(self, X):
        if 'Sales' in X.columns:
            # 1. Limpieza inicial (aquí todavía tenemos Customers)
            df_temp = clean_data(X, self.df_attributes).reset_index()
            #open_mask = (X['Open'] == 1)
            #y = y[open_mask]
            
            y = df_temp['Sales']
            
            # 2. CÁLCULO DE CLUSTERS (Antes de borrar nada)
            # Creamos el perfil de la tienda
            store_profiles = df_temp.groupby('Store').agg({
                'Sales': 'mean',
                'Customers': 'mean',
                'CompetitionDistance': 'first'
            })
            store_profiles['Avg_Ticket'] = store_profiles['Sales'] / (store_profiles['Customers'] + 1)
            
            # Entrenamos un K-Means rápido (puedes usar uno de sklearn aquí)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            # Escalado simple para el fit
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            scaled_profs = sc.fit_transform(store_profiles[['Sales', 'Avg_Ticket', 'CompetitionDistance']])
            
            store_profiles['Cluster'] = km.fit_predict(scaled_profs)
            self.store_cluster_map = store_profiles['Cluster'].to_dict()

            # 3. CÁLCULO DE STORE MEANS (Tu baseline anterior)
            self.store_means = df_temp.groupby('Store')['Sales'].mean().to_dict()
            self.global_mean = y.mean()
        
        return self

    def transform(self, X):
        # 1. Limpieza base (Conserva Customers temporalmente)
        X_proc = clean_data(X, self.df_attributes)
        
        # 2. ASIGNAR CLUSTER (Usando el mapa que aprendimos en el fit)
        X_proc['Cluster'] = X_proc['Store'].map(self.store_cluster_map).fillna(0).astype(int)
        
        # 3. TIME SERIES FEATURES (Lags)
        X_proc = add_time_series_features(X_proc)
        
        # 4. PREPARE FOR MODELING (Aquí es donde se borra Customers al final)
        X_proc = prepare_for_modeling(X_proc)
        
        # 5. BASELINE DE TIENDA
        X_proc['Store_Avg_Sales'] = X_proc['Store'].map(self.store_means).fillna(self.global_mean)
            
        return X_proc