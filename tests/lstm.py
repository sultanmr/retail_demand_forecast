from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
import requests
import io
import tempfile
import pickle

def load_lstm_model_locally():
    # Load the LSTM model
    model_path = "D:/Masterschool/projects/retail_demand_forecast/retail_demand_forecast/model/lstm_model.h5"
    model = load_model(model_path)

    # Define expected input shape
    time_steps = 30   # Based on model summary
    num_features = 15 # Based on model summary

    # Create a dummy input with correct shape
    lstm_input = np.random.rand(1, time_steps, num_features).astype(np.float32)  # Shape (1, 30, 15)

    # Perform prediction
    lstm_pred = model.predict(lstm_input)

    # Print results
    print("LSTM Prediction:", lstm_pred)
    print(model.summary())

LIVE_URLS = {
    "lstm": "https://www.dropbox.com/scl/fi/k2v29l6f4hw413i0y6eva/lstm_model.h5?rlkey=hlsay0e1uqkg720tkobjv2t3x&st=w24uunyt&dl=1",
    "lstm_feature_scaler": "https://www.dropbox.com/scl/fi/uoe6ji0y7q1z6hozf3pn0/lstm_feature_scaler.pkl?rlkey=jz59hjjht4we70peloam7iidi&st=926iabak&dl=1",
    "lstm_target_scaler": "https://www.dropbox.com/scl/fi/5kqz23ywda23syby4oygp/lstm_target_scaler.pkl?rlkey=vwhcvftrgwca5849igemiqkb2&st=hv3egctf&dl=1",
}


def load_remote_file (id):   
    url = LIVE_URLS[id]
    ext = 'pkl'
    if id=="xgb":
        ext = "xgb"
    elif id=="lstm":
        ext = "h5"     
        
    response = requests.get(url)
    if response.status_code == 200:
        fd, temp_file_path = tempfile.mkstemp(suffix=f".{ext}")   
        os.close(fd)       
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)                    
        return temp_file_path
    return None







def load_lstm_model(model_path, feature_scaler_path, target_scaler_path):
    """Load the LSTM model and its scalers"""
    print("Loading LSTM model and scalers...")

    model = load_model(model_path)

    with open(feature_scaler_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    print("LSTM model and scalers loaded successfully")
    return model, feature_scaler, target_scaler



model_path = load_remote_file("lstm")
lstm_feature_scaler = load_remote_file("lstm_feature_scaler")
lstm_target_scaler = load_remote_file("lstm_target_scaler")    

model, feature_scaler, target_scaler = load_lstm_model(model_path, lstm_feature_scaler, lstm_target_scaler)


time_steps = 30   # Based on model summary
num_features = 15 # Based on model summary

# Create a dummy input with correct shape
lstm_input = np.random.rand(1, time_steps, num_features).astype(np.float32)  # Shape (1, 30, 15)

# Perform prediction
lstm_pred = model.predict(lstm_input)
# Print results
print("LSTM Prediction:", lstm_pred)
print(model.summary())