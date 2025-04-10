import streamlit as st
from datetime import datetime
import logging

import concurrent.futures
import mlflow
import mlflow.xgboost
import mlflow.keras
import xgboost as xgb
from pyngrok import ngrok
import os
import tempfile
from app.config import NGROK_TOKEN, MLFLOW_CONFIG
#import app.config

log_container = None




def update_logs(lc=None):    
    global log_container
    if lc is not None:
        log_container = lc
    
    # Initialize logs if they don't exist
    if "logs" not in st.session_state:
        st.session_state.logs = []
    
    # Keep only the last 5 logs
    st.session_state.logs = st.session_state.logs[-5:]
    logs_text = "\n".join(st.session_state.logs) if st.session_state.logs else "No logs available yet."        
    
    # Only update the container if it exists and we're in Streamlit context
    if log_container is not None and is_streamlit_running():
        try:
            # Use a container with custom CSS to reduce spacing
            with log_container:
                st.markdown("""
                    <style>
                    .stTextArea {
                        margin-bottom: 0px;
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.text_area("Logs", value=logs_text, height=150, disabled=True)
        except Exception as e:
            print(f"Error updating logs: {e}")

def setup_logger():
    """Setup logging configuration"""
    logger = logging.getLogger('retail_forecast')
    logger.setLevel(logging.INFO)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)

    return logger, mlflow_init()

mlflow_url = None

def mlflow_init():
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_CONFIG['MLFLOW_TRACKING_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_CONFIG['MLFLOW_TRACKING_PASSWORD']
    mlflow.set_tracking_uri(MLFLOW_CONFIG['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(MLFLOW_CONFIG['MLFLOW_EXPERIMENT_NAME'])
    mlflow_url = MLFLOW_CONFIG['MLFLOW_TRACKING_URI']
    mlflow.set_tracking_uri(mlflow_url)     
    # experiment_url = mlflow_url    
    # with mlflow.start_run() as run:      
    #     print ("*"*100)
    #     run_id = run.info.run_id        
    #     experiment_url = f"{mlflow_url}/#/experiments/0/runs/{run_id}"
    #     print (f"Experiment URL: {experiment_url}")
    #     print ("*"*100)
    
    return mlflow_url


import tempfile
import mlflow

def log_image(fig, artifact_path="plots"):
    """Save a Plotly figure and log it as an image artifact in MLflow."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        fig.write_image(tmp_img.name)  # Removed bbox_inches, Plotly doesn't use it
        mlflow.log_artifact(tmp_img.name, artifact_path=artifact_path)


def log_model(model, name):   
    global mlflow_url
    if mlflow_url is None:
        return
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if "xgb" in name:
                executor.submit(log_xgb, model, name)
            else:
                executor.submit(log_lstm, model, name)
    except:
        pass

def log_xgb(model, name):
    with mlflow.start_run():
        # Log the XGBoost model
        mlflow.xgboost.log_model(model, name)
        
        # Log model parameters
        params = model.get_params()
        for param, value in params.items():
            mlflow.log_param(param, value)

def log_lstm(model, name):
    with mlflow.start_run():   
         # Log LSTM model
        mlflow.keras.log_model(model, name)

        # Log LSTM model parameters
        params = model.get_config() 
        for param, value in params.items():
            mlflow.log_param(param, str(value))  

def is_streamlit_running():
    """Check if the code is running in Streamlit"""
    try:
        st.write("")
        return True
    except:
        return False

def update_session_logs(message, level='info'):
    """Update only the session state logs without UI updates"""
    # Ensure logs are initialized
    if "logs" not in st.session_state:
        st.session_state.logs = []
    #check if message have only spaces or is None
    if message is None or message.strip() == "":    
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Define icons for different log levels
    icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }
    
    # Get the appropriate icon
    icon = icons.get(level, 'ℹ️')
    
    # Format the message with timestamp and icon
    formatted_message = f"{timestamp} {icon} {message}"
    print(formatted_message)
    
    # Append new log and keep only last 5
    st.session_state.logs.append(formatted_message)    
    st.session_state.logs = st.session_state.logs[-5:]  # Keep only last 5 logs

def log(message, level='info', update_ui=True):
    """Log a message to both console and Streamlit UI"""
    # Update session state logs
    update_session_logs(message, level)
    try:
        mlflow.log_param(level, str(message))
    except:
        pass
    # Only update UI if requested and we're in Streamlit context
    if update_ui and is_streamlit_running():
        update_logs()

"""

    # Display logs in Streamlit UI
    st.subheader("Latest Logs")
    for log_message, log_level in st.session_state.logs:
        if log_level == 'error':
            st.error(log_message)
        elif log_level == 'warning':
            st.warning(log_message)
        elif log_level == 'success':
            st.success(log_message)
        else:
            st.info(log_message)
"""

