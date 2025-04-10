import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import requests
import tempfile
import os

# Constants
FEATURE_COLUMNS = [
    'store_nbr', 'item_nbr', 'month', 'day', 'weekofyear', 'dayofweek',
    'rolling_mean_7', 'rolling_std_7', 'lag_1', 'lag_7',
    'store_type', 'store_cluster', 'item_family', 'item_class', 'item_perishable'
]
CATEGORICAL_COLUMNS = ['store_type', 'store_cluster', 'item_family', 'item_class', 'item_perishable']
MODEL_URL = "https://www.dropbox.com/scl/fi/e6im63aml9fgh5yosr4e0/xgb_model.xgb?rlkey=i7kshv5cqvv7nlgo29e56cglr&st=grn4zfgv&dl=1"

def download_model(url):
    """Download the XGBoost model from the given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise Exception(f"Failed to fetch model, status code: {response.status_code}")

def load_model(model_path):
    """Load the XGBoost model from the given path."""
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

def preprocess_data(data, categorical_columns):
    """Preprocess the input data by encoding categorical variables."""
    encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])
    return data

def create_dummy_data():
    """Generate dummy input data."""
    return pd.DataFrame({
        'store_nbr': [1],
        'item_nbr': [564533],
        'month': [3],
        'day': [15],
        'weekofyear': [11],
        'dayofweek': [3],
        'rolling_mean_7': [0.25],
        'rolling_std_7': [0.1],
        'lag_1': [15],
        'lag_7': [12],
        'store_type': ['A'],
        'store_cluster': ['Cluster1'],
        'item_family': ['FamilyX'],
        'item_class': ['Class1'],
        'item_perishable': ['Yes']
    })

def main():
    try:
        # Download and load the model
        model_path = download_model(MODEL_URL)
        model = load_model(model_path)
        os.remove(model_path)  # Clean up temporary file

        # Prepare dummy data
        dummy_data = create_dummy_data()
        dummy_data = preprocess_data(dummy_data, CATEGORICAL_COLUMNS)
        dummy_data = dummy_data[FEATURE_COLUMNS]

        # Convert to DMatrix and make predictions
        dmat = xgb.DMatrix(dummy_data)
        predictions = model.predict(dmat)

        print("XGBoost Prediction:", predictions)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
