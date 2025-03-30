import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import warnings
# Disable all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
# Disable specific deprecation warnings from TensorFlow and Keras
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='keras')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow.python')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow.python.keras')
# Disable FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Disable UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)


import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import datetime
import pandas as pd
from app.config import TRAIN_CONF, DEFAULTS, ITEM_IDS
from app.utils import log, log_model



# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.config import DATA_PATH, MODEL_PATH
from data.data_utils import load_data,  preprocess_input_data
from model.model_utils import load_models, predict
from app.utils import log, setup_logger, update_logs

# Setup logger
logger, mlflow_url = setup_logger()

@st.cache_data
def load_cached_data():
    """Cache the data loading to prevent multiple loads"""
    return load_data()

@st.cache_data
def load_cached_data_without_training():
    """Cache the data loading to prevent multiple loads"""
    return load_data_without_training()

def plot_historical_sales(df_filtered, store_id, item_id):
    """Plot historical sales data for selected store and item"""
    store_data = df_filtered[
        (df_filtered['store_nbr'] == store_id) & 
        (df_filtered['item_nbr'] == item_id)
    ]
    
    fig = px.line(
        store_data,
        x='date',
        y='unit_sales',
        title=f'Historical Sales for Store {store_id} - Item {item_id}',
        labels={'unit_sales': 'Sales', 'date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode='x unified'
    )
    
    return fig

def plot_sales_distribution(df_filtered, store_id, item_id):
    """Plot sales distribution for selected store and item"""
    store_data = df_filtered[
        (df_filtered['store_nbr'] == store_id) & 
        (df_filtered['item_nbr'] == item_id)
    ]
    
    fig = px.histogram(
        store_data,
        x='unit_sales',
        title=f'Sales Distribution for Store {store_id} - Item {item_id}',
        labels={'unit_sales': 'Sales', 'count': 'Frequency'}
    )
    
    fig.update_layout(
        xaxis_title="Sales",
        yaxis_title="Frequency",
        showlegend=False
    )
    
    return fig

def plot_weekly_patterns(df_filtered, store_id, item_id):
    """Plot weekly sales patterns"""
    # Create a copy of the filtered data to avoid the warning
    store_data = df_filtered[
        (df_filtered['store_nbr'] == store_id) & 
        (df_filtered['item_nbr'] == item_id)
    ].copy()
    
    # Use .loc to modify the DataFrame
    store_data.loc[:, 'dayofweek'] = store_data['date'].dt.dayofweek
    weekly_avg = store_data.groupby('dayofweek')['unit_sales'].mean().reset_index()
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg['day'] = weekly_avg['dayofweek'].map(lambda x: days[x])
    
    fig = px.bar(
        weekly_avg,
        x='day',
        y='unit_sales',
        title=f'Average Sales by Day of Week - Store {store_id} - Item {item_id}',
        labels={'unit_sales': 'Average Sales', 'day': 'Day of Week'}
    )
    
    fig.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Average Sales",
        showlegend=False
    )
    
    return fig

def main():
    global mlflow_url

    st.set_page_config(page_title="Sales Forecasting", layout="wide")


    use_xgb = TRAIN_CONF['use_xgb']
    use_lstm = TRAIN_CONF['use_lstm']

    store_id = DEFAULTS['store_id']
    item_id = DEFAULTS['item_id']

    st.markdown("""
        <style>
            /* Remove the space above the header/title */
            .css-1v3fvcr {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Sales Forecasting")

    # Initialize logs if not already initialized
    if "logs" not in st.session_state:
        st.session_state.logs = []   

    # Create the log container first
    log_container = st.empty()
    
    # Now that we have the container, we can safely log messages
    if len(st.session_state.logs) == 0:
        log("Application started", level='info')    
    # Update the logs display
    update_logs(log_container)

    # Create a container for spinners
    spinner_container = st.empty()
    st.markdown(f"[🔗 MLFlow URL]({mlflow_url})")
    # Main content
    with spinner_container:
        with st.spinner('Loading Data ...'):                        
            # Load all data including training data
            df_stores, df_items, df_transactions, df_oil, df_holidays, df_train = load_cached_data()
            log("Data loaded successfully", level='success', update_ui=False)       
            
    spinner_container.empty()

    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        store_id = st.selectbox("Store", [1], key="store_id")
    with col2:
        item_id = st.selectbox("Item", ITEM_IDS, key="item_id")

    default_date = DEFAULTS['date']
    min_date = DEFAULTS['min_date']
    max_date = DEFAULTS['max_date']    

    date = st.date_input("Forecast Date", value=default_date, min_value=min_date, max_value=max_date)

    # Create tabs for different views
    tab1, tab2 = st.tabs(["Forecast", "Analysis"])

    with tab1:

        use_xgb = st.checkbox("XGB Forecasting", value=use_xgb)  
        use_lstm = st.checkbox("LSTM Forecasting", value=use_lstm)  
    
        if st.button("Get Forecast"):
            spinner_container = st.empty()
            with spinner_container:
                with st.spinner('Processing data and generating forecast...'):
                    store_id = st.session_state.store_id
                    item_id = st.session_state.item_id
                    input_data = preprocess_input_data(
                        store_id=store_id,
                        item_id=item_id,
                        date=date,
                        df_stores=df_stores,
                        df_items=df_items,
                        df_train=df_train
                    )
                    predictions, xgb_model, lstm_model = predict(store_id, item_id, input_data, df_stores, df_items, df_train)

            spinner_container.empty()   
                # Create forecast visualization
            fig = go.Figure()
                
            # Add historical data
            historical_data = df_train[
                (df_train['store_nbr'] == store_id) & 
                (df_train['item_nbr'] == item_id) &
                (df_train['date'] < pd.Timestamp(date))  # Convert date to pandas Timestamp
            ]

            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['unit_sales'],
                name='Historical Sales',
                line=dict(color='blue')
            ))

            if use_xgb:  
                fig.add_trace(go.Scatter(
                    x=input_data['date'],  
                    y=predictions['xgb_pred'],  
                    name='XGBoost Prediction',
                    mode='lines',        
                    line=dict(color='red', width=2)                    
                ))

                fig.add_trace(go.Scatter(
                    x=[date],
                    y=[predictions['xgb_pred'][-1]],
                    mode='markers',
                    name='XGBoost Prediction',
                    marker=dict(color='red', size=10),
                    showlegend=False                    
                ))

            if use_lstm:  

                fig.add_trace(go.Scatter(
                    x=input_data['date'].iloc[TRAIN_CONF["seq_length"]:], 
                    y=predictions['lstm_pred'],  
                    name='LSTM Prediction',
                    mode='lines',        
                    line=dict(color='green', width=2),
                    
                ))

                fig.add_trace(go.Scatter(
                    x=[date],
                    y=[predictions['lstm_pred'][-1]],
                    mode='markers',
                    name='LSTM Prediction',
                    marker=dict(color='green', size=10),
                    showlegend=False          
                ))


            fig.update_layout(
                title=f'Sales Forecast for Store {store_id} - Item {item_id}',
                xaxis_title="Date",
                yaxis_title="Sales",
                hovermode='x unified'
            )
                
            st.plotly_chart(fig, use_container_width=True)
                
            # Display predictions in a more readable format
            col1, col2 = st.columns(2)
            with col1:
                if use_xgb:
                    st.metric(f"XGBoost Prediction on {date} for Item: {item_id}", f"{predictions['xgb_pred'][-1]:.2f}")
                    log(f"XGBoost Prediction for {date}: {predictions['xgb_pred'][-1]:.2f}", level='success')
            with col2:
                if use_lstm:                    
                    st.metric(f"LSTM Prediction on {date} for Item: {item_id}", f"{predictions['lstm_pred'][-1]:.2f}")
                    log(f"LSTM Prediction for {date}: {predictions['lstm_pred'][-1]:.2f}", level='success')                


            if use_xgb:
                log_model(xgb_model, f"xgb model for { st.session_state.item_id}")
            if use_lstm:   
                log_model(lstm_model, f"lstm model  for { st.session_state.item_id}")

    with tab2:        
        st.plotly_chart(plot_historical_sales(df_train, store_id, item_id), use_container_width=True)        
        col1, col2 = st.columns(2)        
        with col1:
            st.plotly_chart(plot_sales_distribution(df_train, store_id, item_id), use_container_width=True)        
        with col2:
            st.plotly_chart(plot_weekly_patterns(df_train, store_id, item_id), use_container_width=True)

if __name__ == "__main__":
    main()