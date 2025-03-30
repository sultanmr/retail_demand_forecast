import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import requests
import io
import tempfile
import os
# Dummy feature columns
feature_columns = [
    'store_nbr', 'item_nbr', 'month', 'day', 'weekofyear', 'dayofweek',
    'rolling_mean_7', 'rolling_std_7', 'lag_1', 'lag_7',
    'store_type', 'store_cluster', 'item_family', 'item_class', 'item_perishable'
]

# Load XGBoost model
#model_path = "D:/Masterschool/projects/retail_demand_forecast/retail_demand_forecast/model/xgb_model.xgb"
model_url ="https://www.dropbox.com/scl/fi/e6im63aml9fgh5yosr4e0/xgb_model.xgb?rlkey=i7kshv5cqvv7nlgo29e56cglr&st=grn4zfgv&dl=1"
# Ensure the request was successful
response = requests.get(model_url)
if response.status_code == 200:
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
    
        model = xgb.XGBRegressor()
        model.load_model(temp_file_path)
    
    # Clean up the temporary file after loading
    os.remove(temp_file_path)

else:
    print(f"Failed to fetch model, status code: {response.status_code}")


# Generate a dummy input data (using random values here)
dummy_data = {
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
}

# Convert to DataFrame
dummy_df = pd.DataFrame(dummy_data)

# Handle categorical variables: Encode them with LabelEncoder
categorical_columns = ['store_type', 'store_cluster', 'item_family', 'item_class', 'item_perishable']
encoder = LabelEncoder()

for col in categorical_columns:
    dummy_df[col] = encoder.fit_transform(dummy_df[col])

# Extract features based on the feature_columns
dummy_df = dummy_df[feature_columns]

# Convert to a DMatrix (XGBoost input format)
dmat = xgb.DMatrix(dummy_df)

# Predict with the model
xgb_prediction = model.predict(dmat)

print("XGBoost Prediction:", xgb_prediction)
