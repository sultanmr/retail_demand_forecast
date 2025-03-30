# data/data_utils.py
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from app.config import PARQUET_DATA_PATH, PARQUET_URLS
import requests
import io
from app.config import DATA_PATH, TRAIN_CONF
from app.utils import update_session_logs

def load_parquet_files_locally():
    """
    Load each Parquet file into a separate DataFrame and store in a dictionary.
    Returns None if no Parquet files are found.
    """

    parquet_files = glob.glob(os.path.join(PARQUET_DATA_PATH, "*.parquet"))
    
    if not parquet_files:  # Check if no files exist
        update_session_logs("❌ No Parquet files found in the directory.")
        return None

    dataframes = {}

    for file in parquet_files:
        file_name = os.path.basename(file).replace(".parquet", "") 
        try:
            dataframes[file_name] = pd.read_parquet(file, engine="pyarrow") 
            update_session_logs(f"✅ Loaded {file} into DataFrame: {file_name}")
        except Exception as e:
            update_session_logs(f"⚠️ Error loading {file}: {e}")

    return dataframes


def load_parquet_from_url():
    """
    Loads Parquet files directly from URLs into a dictionary of DataFrames.
    """
    dataframes = {}

    for name, url in PARQUET_URLS.items():
        update_session_logs(f"⬇️ Loading {name} ...")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                df = pd.read_parquet(io.BytesIO(response.content), engine="pyarrow")
                dataframes[name] = df              
            else:
                update_session_logs(f"❌ Failed to load {name} (HTTP {response.status_code})")
        except Exception as e:
            update_session_logs(f"⚠️ Error loading {name}: {e}")

    return dataframes


def load_data():
    """Loads the training data and related datasets."""
    update_session_logs("Loading data...")
    
    dfs = load_parquet_files_locally()
    if dfs is None:
        dfs = load_parquet_from_url()

    df_stores = dfs["stores"]
    df_items = dfs["items"]
    df_transactions = dfs["transactions"]
    df_oil = dfs["oil"]
    df_holidays = dfs["holidays_events"]
    df_train = dfs["train"]
        
    # Convert date columns to datetime
    update_session_logs("Converting date columns to datetime...")
    date_columns = ['date']
    for df in [df_train, df_transactions, df_oil, df_holidays]:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    
    update_session_logs("Data loading complete", level='success')
    return df_stores, df_items, df_transactions, df_oil, df_holidays, df_train

def preprocess_input_data(store_id, item_id, date, df_stores, df_items, df_train):
    """Preprocess input data for prediction"""
    update_session_logs("Preprocessing input data...")
    
    # Create date range for the specific store-item combination
    start_date = df_train['date'].min()
    end_date = date

    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Filter data for specific store and item
    df_filtered = df_train[
        (df_train['store_nbr'] == store_id) & 
        (df_train['item_nbr'] == item_id)
    ].copy()
    
    # Create a complete DataFrame with all dates
    df_complete = pd.DataFrame({'date': full_date_range})
    df_complete['store_nbr'] = store_id
    df_complete['item_nbr'] = item_id
    
    # Merge with original data to get sales
    df_filled = pd.merge(
        df_complete,
        df_filtered[['date', 'unit_sales']],
        on='date',
        how='left'
    )
    
    median = df_filled['unit_sales'].median()
    df_filled['unit_sales'] = df_filled['unit_sales'].fillna(median)
    
    # Add store and item features
    store_features = df_stores[df_stores['store_nbr'] == store_id].iloc[0]
    item_features = df_items[df_items['item_nbr'] == item_id].iloc[0]
    
    # Add features
    df_filled['store_type'] = store_features['type']
    df_filled['store_cluster'] = store_features['cluster']
    df_filled['item_family'] = item_features['family']
    df_filled['item_class'] = item_features['class']
    df_filled['item_perishable'] = item_features['perishable']
    
    # Add time-based features
    df_filled['month'] = df_filled['date'].dt.month.astype(int)
    df_filled['day'] = df_filled['date'].dt.day.astype(int)
    df_filled['weekofyear'] = df_filled['date'].dt.isocalendar().week.astype(int)
    df_filled['dayofweek'] = df_filled['date'].dt.dayofweek.astype(int)
    
    # Add rolling features
    df_filled['rolling_mean_7'] = df_filled['unit_sales'].rolling(window=7, min_periods=1).mean().astype(float)
    df_filled['rolling_std_7'] = df_filled['unit_sales'].rolling(window=7, min_periods=1).std().astype(float)
    
    # Add lag features
    df_filled['lag_1'] = df_filled['unit_sales'].shift(1).fillna(0).astype(float)
    df_filled['lag_7'] = df_filled['unit_sales'].shift(7).fillna(0).astype(float)
    
    # Convert categorical features to numeric using LabelEncoder
    le = LabelEncoder()
    categorical_columns = ['store_type', 'store_cluster', 'item_family', 'item_class']
    for col in categorical_columns:
        df_filled[col] = le.fit_transform(df_filled[col].astype(str)).astype(int)
    
    # Ensure all numeric columns are float type
    numeric_columns = [
        'store_nbr', 'item_nbr', 'unit_sales', 'month', 'day', 
        'weekofyear', 'dayofweek', 'rolling_mean_7', 'rolling_std_7',
        'lag_1', 'lag_7', 'store_type', 'store_cluster', 
        'item_family', 'item_class', 'item_perishable'
    ]
    
    for col in numeric_columns:
        df_filled[col] = df_filled[col].astype(float)
    
    update_session_logs("Input data preprocessing completed")
    return df_filled

