import streamlit as st
from datetime import datetime
import logging

import concurrent.futures
import mlflow
import mlflow.xgboost
import mlflow.keras
import xgboost as xgb
from pyngrok import ngrok

from app.config import NGROK_TOKEN

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

    return logger, setup_mlflow()

def setup_mlflow():
    ngrok.set_auth_token(NGROK_TOKEN)
    ngrok.kill()
    mlflow_storage_path = "/content/mlruns"
    
    mlflow_url = ngrok.connect(8501)
    
    mlflow.set_tracking_uri(f"file:{mlflow_storage_path}")
    mlflow.set_experiment("Sales Forecasting")
    
    return mlflow_url

def log_model(model, name):   
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if "xgb" in name:
            executor.submit(log_xgb, model, name)
        else:
            executor.submit(log_lstm, model, name)

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

