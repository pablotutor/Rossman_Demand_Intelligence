import sqlite3
import pandas as pd
import os

# Define the local database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rossmann.db')

def init_relational_db(train_csv_path: str, store_csv_path: str):
    """
    Reads the raw CSV files and builds a relational SQLite database.
    This creates a Dimension table (stores) and a Fact table (historical_sales).
    Execute this function only once to set up the DB.
    """
    print("⏳ Reading raw CSV files...")
    
    try:
        df_train = pd.read_csv(train_csv_path, low_memory=False)
        df_store = pd.read_csv(store_csv_path)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}. Please check your CSV paths.")
        return

    # Ensure Date format is standard
    df_train['Date'] = pd.to_datetime(df_train['Date']).dt.strftime('%Y-%m-%d')
    
    print("💾 Building relational tables in SQLite...")
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Create Dimension Table: 'stores'
    df_store.to_sql('stores', conn, if_exists='replace', index=False)
    
    # 2. Create Fact Table: 'historical_sales'
    df_train.to_sql('historical_sales', conn, if_exists='replace', index=False)
    
    # 3. Create Indexes for hyper-fast querying
    print("⚡ Creating database indexes...")
    conn.execute('CREATE INDEX idx_store_id ON stores(Store)')
    conn.execute('CREATE INDEX idx_sales_store_date ON historical_sales(Store, Date)')
    
    conn.close()
    print(f"✅ Relational database successfully built at: {DB_PATH}")


def get_store_context(store_id: int, target_date: str, days_back: int = 35) -> pd.DataFrame:
    """
    Executes a SQL JOIN query to fetch both the historical sales (for lags)
    and the static store attributes (competition, promo2) for a given store.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Relational SQL Query: INNER JOIN between fact and dimension tables
    query = f"""
        SELECT 
            h.Store, h.DayOfWeek, h.Date, h.Sales, h.Customers, h.Open, h.Promo, h.StateHoliday, h.SchoolHoliday,
            s.StoreType, s.Assortment, s.CompetitionDistance,
            s.CompetitionOpenSinceMonth, s.CompetitionOpenSinceYear,
            s.Promo2, s.Promo2SinceWeek, s.Promo2SinceYear, s.PromoInterval
        FROM historical_sales h
        INNER JOIN stores s ON h.Store = s.Store
        WHERE h.Store = {store_id} AND h.Date < '{target_date}'
        ORDER BY h.Date DESC
        LIMIT {days_back}
    """
    
    # Execute query and load into a Pandas DataFrame
    df_context = pd.read_sql(query, conn)
    conn.close()
    
    # Reverse the order so it is chronological (oldest to newest)
    if not df_context.empty:
        df_context = df_context.sort_values('Date').reset_index(drop=True)
        
    return df_context

# --- EXECUTION BLOCK (To build the DB for the first time) ---
if __name__ == "__main__":
    # UPDATE THESE PATHS to match the location of your raw data files
    TRAIN_CSV = '../data/train.csv' 
    STORE_CSV = '../data/store.csv'
    
    init_relational_db(TRAIN_CSV, STORE_CSV)