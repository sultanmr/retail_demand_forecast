import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from app.config import MODEL_PATH, MODEL_URLS, TRAIN_CONF
from data.data_utils import preprocess_input_data, load_data
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import os
import requests
from sklearn.metrics import make_scorer, mean_squared_error
import optuna
import io
import tempfile
from app.utils import log, log_model

def create_sequences(X, y, seq_length):
    log(f"Creating sequences with length {seq_length}...")
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:(i + seq_length)])
        y_seq.append(y[i + seq_length])  # Use the scaled target values
    log(f"Created {len(X_seq)} sequences")
    return np.array(X_seq), np.array(y_seq)

def objective_xgb(trial, X, y, tscv):
    """Objective function for Optuna XGBoost optimization"""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0.1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5)
    }
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=TRAIN_CONF["random_state"],
        n_jobs=-1,
        **param
    )
    
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(mean_squared_error(y_val, y_pred, squared=False))
    
    return np.mean(scores)

def objective_lstm(trial, X, y, tscv):
    """Objective function for Optuna LSTM optimization"""
    param = {
        'lstm_units1': trial.suggest_int('lstm_units1', 32, 256),
        'lstm_units2': trial.suggest_int('lstm_units2', 32, 256),
        'dropout1': trial.suggest_float('dropout1', 0.2, 0.6),
        'dropout2': trial.suggest_float('dropout2', 0.2, 0.6),
        'learning_rate': trial.suggest_float('learning_rate', 0.00005, 0.005),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'epochs': trial.suggest_int('epochs', 10, 50)
    }
    
    scores = []
    # Split the sequences directly
    n_splits = TRAIN_CONF["lstm_n_splits"]
    split_size = len(X) // (n_splits + 1)
    
    for i in range(n_splits):
        # Calculate indices for this split
        val_start = (i + 1) * split_size
        val_end = (i + 2) * split_size
        
        # Split the data
        X_train = np.concatenate([X[:val_start], X[val_end:]])
        X_val = X[val_start:val_end]
        y_train = np.concatenate([y[:val_start], y[val_end:]])
        y_val = y[val_start:val_end]
        
        # Create model using Input layer
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        x = LSTM(param['lstm_units1'], return_sequences=True)(inputs)
        x = Dropout(param['dropout1'])(x)
        x = LSTM(param['lstm_units2'], return_sequences=False)(x)
        x = Dropout(param['dropout2'])(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=param['learning_rate']),
            loss=tf.keras.losses.MeanSquaredError()
        )
        
        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=param['batch_size'],
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        # Make predictions on validation set
        y_pred = model.predict(X_val)
        
        # Ensure shapes match before calculating MSE
        y_val = y_val.reshape(-1)  # Flatten to 1D array
        y_pred = y_pred.reshape(-1)  # Flatten to 1D array
        
        # Calculate RMSE
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        scores.append(rmse)
    
    if not scores:  # If no valid scores were calculated
        return float('inf')  # Return a high value to indicate this trial failed
    
    return np.mean(scores)


def train_xgboost_model(store_id, item_id, df_train, df_stores, df_items):
    """Train XGBoost model with hyperparameter optimization"""
    log("Starting XGBoost model training...")

    # Preprocess training data
    df_processed = preprocess_input_data(
        store_id=store_id,  
        item_id=item_id,   
        date=df_train['date'].max(),          
        df_stores=df_stores,
        df_items=df_items,
        df_train=df_train
    )


    # Prepare features and target
    feature_columns = [
        'store_nbr', 'item_nbr', 'month', 'day', 'weekofyear', 'dayofweek',
        'rolling_mean_7', 'rolling_std_7', 'lag_1', 'lag_7',
        'store_type', 'store_cluster', 'item_family', 'item_class', 'item_perishable'
    ]

    X = np.asarray(df_processed[feature_columns], dtype=np.float32)
    y = np.asarray(df_processed['unit_sales'], dtype=np.float32)

    # Split the data using a simple train-test split
    train_size = int(len(X) * TRAIN_CONF["train_size"])
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]


    # Define the objective function for Optuna
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'objective': 'reg:squarederror'
        }

        model = xgb.XGBRegressor(**param, random_state=TRAIN_CONF["random_state"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=TRAIN_CONF["total_trials"])

    # Train final model with best parameters
    best_params = study.best_params
    best_params['objective'] = 'reg:squarederror'
    final_model = xgb.XGBRegressor(**best_params, random_state=TRAIN_CONF["random_state"])
    final_model.fit(X_train, y_train)

    # Create model path for the store-item combination
    model_path = os.path.join(MODEL_PATH, str(store_id), str(item_id))
    os.makedirs(model_path, exist_ok=True)

    # Save the model and parameters
    model_file_path = os.path.join(model_path, "xgb_model.xgb")
    params_path = os.path.join(model_path, "xgb_params.pkl")

    final_model.save_model(model_file_path)
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)

    log("XGBoost model training completed", level='success')
    return final_model


def create_lstm_model(input_shape, trial=None):
    """Create LSTM model with hyperparameter tuning"""
    if trial is None:
        # Default hyperparameters
        lstm_units = TRAIN_CONF["lstm_units"]
        dropout_rate = TRAIN_CONF["dropout_rate"]
        learning_rate = TRAIN_CONF["learning_rate"]
    else:
        # Hyperparameters from trial
        lstm_units = trial.suggest_int('lstm_units', 32, 128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)

    # Reset TensorFlow graph using recommended method
    tf.compat.v1.reset_default_graph()

    # Create model using functional API
    inputs = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

def train_lstm_model(store_id, item_id, df_train, df_stores, df_items):    
    """Trains an LSTM model with hyperparameter optimization."""
    log("Starting LSTM Model Training with Hyperparameter Optimization", level='info')

    # Preprocess the training data
    log("Preprocessing training data...")
    # Use the first store and item for training data preparation    
    date = df_train['date'].max()
    
    train_data = preprocess_input_data(
        store_id=store_id,
        item_id=item_id,
        date=date,
        df_stores=df_stores,
        df_items=df_items,
        df_train=df_train
    )
    
    log("Preparing features and target...")
    # Separate features and target
    feature_columns = [
        'store_nbr', 'item_nbr', 'month', 'day', 'weekofyear', 'dayofweek',
        'rolling_mean_7', 'rolling_std_7', 'lag_1', 'lag_7',
        'store_type', 'store_cluster', 'item_family', 'item_class', 'item_perishable'
    ]
    X = train_data[feature_columns]
    y = train_data['unit_sales']
    
    # Check for NaN values and handle them
    if X.isna().any().any():
        log("Found NaN values in features. Filling with forward fill and backward fill...")
        X = X.fillna(method='ffill').fillna(method='bfill')
    
    if y.isna().any():
        log("Found NaN values in target. Filling with forward fill and backward fill...")
        y = y.fillna(method='ffill').fillna(method='bfill')
    
    log("Scaling data...")
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))
    
    # Create sequences
    seq_length = TRAIN_CONF["seq_length"]
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    # Validate sequences
    if np.isnan(X_seq).any() or np.isnan(y_seq).any():
        log("Found NaN values in sequences. Removing invalid sequences...")
        valid_indices = ~(np.isnan(X_seq).any(axis=(1, 2)) | np.isnan(y_seq).any(axis=1))
        X_seq = X_seq[valid_indices]
        y_seq = y_seq[valid_indices]
    
    if len(X_seq) == 0 or len(y_seq) == 0:
        raise ValueError("No valid sequences after removing NaN values")
    
    # Ensure y_seq has the correct shape
    y_seq = y_seq.reshape(-1, 1)  # Reshape to (n_samples, 1)
    
    log(f"Sequence shape: {X_seq.shape}, Target shape: {y_seq.shape}")
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    log("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_lstm(trial, X_seq, y_seq, tscv), n_trials=TRAIN_CONF["total_trials"])
    
    log(f"Best hyperparameters: {study.best_params}")
    log(f"Best RMSE: {study.best_value}")
    
    # Train final model with best parameters
    log("Training final model with best parameters...")
    best_params = study.best_params
    
    # Create final model using Input layer
    inputs = Input(shape=(seq_length, X.shape[1]))
    x = LSTM(best_params['lstm_units1'], return_sequences=True)(inputs)
    x = Dropout(best_params['dropout1'])(x)
    x = LSTM(best_params['lstm_units2'], return_sequences=False)(x)
    x = Dropout(best_params['dropout2'])(x)
    outputs = Dense(1)(x)
    
    final_model = Model(inputs=inputs, outputs=outputs)
    
    final_model.compile(
        optimizer=Adam(learning_rate=best_params['learning_rate']),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    final_model.fit(
        X_seq, y_seq,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # Save the model, scalers, and hyperparameters
    log("Saving model, scalers, and hyperparameters...")
    
    model_path = os.path.join(MODEL_PATH, str(store_id), str(item_id))
    os.makedirs(model_path, exist_ok=True)

    model_file_path = os.path.join(model_path, "lstm_model.h5")
    feature_scaler_path = os.path.join(model_path, "lstm_feature_scaler.pkl")
    target_scaler_path = os.path.join(model_path, "lstm_target_scaler.pkl")
    params_path = os.path.join(model_path, "lstm_params.pkl")
    
    final_model.save(model_file_path)
    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)
    
    log(f"Model saved to {model_file_path}")
    log(f"Feature scaler saved to {feature_scaler_path}")
    log(f"Target scaler saved to {target_scaler_path}")
    log(f"Hyperparameters saved to {params_path}")
    
    log("LSTM Model Training Complete", level='success')
    return final_model, feature_scaler, target_scaler

def load_remote_file (file_type, store_id, item_id):   
    model_id = f"{store_id}-{item_id}"
    
    if model_id not in MODEL_URLS or file_type not in MODEL_URLS[model_id]:
        log(f"❌ No URL found for model_id: {model_id}, file_type: {file_type}")
        return None

    url = MODEL_URLS[model_id][file_type]
    if not url:
        log(f"⚠️ URL for {file_type} is empty for model_id: {model_id}")
        return None

    ext = 'pkl'
    if file_type=="xgb":
        ext = "xgb"
    elif file_type=="lstm":
        ext = "h5"     
        
    response = requests.get(url)
    if response.status_code == 200:
        fd, temp_file_path = tempfile.mkstemp(suffix=f".{ext}")   
        os.close(fd)       
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)                    
        return temp_file_path
    return None


def load_xgboost_model(model_path):
    """Load the XGBoost model"""
    log("Loading XGBoost model...")
    
    # Load the model
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    log("XGBoost model loaded successfully", level='success')
    return model
def save_to_session(store_id, item_id, xgb_model, lstm_model, lstm_scaler, lstm_target_scaler):
    """Save models to session state"""
    if "models" not in st.session_state:
        st.session_state.models = {}

    st.session_state.models[(store_id, item_id)] = {
        "xgb_model": xgb_model,
        "lstm_model": lstm_model,
        "lstm_scaler": lstm_scaler,
        "lstm_target_scaler": lstm_target_scaler
    }

    

def get_from_session(store_id, item_id):
    """Retrieve models from session state for a given store-item combination."""    
    if "models" not in st.session_state:
        st.warning("No models found in session state.")
        return None    
    # Get the model data if available
    model_data = st.session_state.models.get((store_id, item_id))    
    
    if model_data is None:
        st.warning(f"No models found for Store {store_id}, Item {item_id}.")
        return None
    
    return model_data


def load_models(store_id, item_id, df_stores, df_items, df_train):
    """Load or train models based on availability and configuration flags"""
    log(f"Loading models for {store_id}, {item_id}...")
    is_xgb_enabled=TRAIN_CONF["use_xgb"]
    is_lstm_enabled=TRAIN_CONF["use_lstm"]
    
    model_path = os.path.join(MODEL_PATH, str(store_id), str(item_id))
    os.makedirs(model_path, exist_ok=True)

    if not is_xgb_enabled and not is_lstm_enabled:
        raise ValueError("At least one model must be enabled")

    if "xgb_model" in st.session_state and "lstm_model" in st.session_state:
        return get_from_session(store_id, item_id)

    models = {"xgb": None, "lstm": None, "lstm_scaler": None, "lstm_target_scaler": None}

    
    # Define model paths
    paths = {
        "xgb": os.path.join(model_path, "xgb_model.xgb"),
        "lstm": os.path.join(model_path, "lstm_model.h5"),
        "lstm_scaler": os.path.join(model_path, "lstm_feature_scaler.pkl"),
        "lstm_target_scaler": os.path.join(model_path,  "lstm_target_scaler.pkl"),
    }

    def try_load_local():
        """Attempt to load models from local storage."""
        try:
            if is_xgb_enabled and os.path.exists(paths["xgb"]):
                models["xgb"] = load_xgboost_model(paths["xgb"])

            if is_lstm_enabled and all(os.path.exists(paths[key]) for key in ["lstm", "lstm_scaler", "lstm_target_scaler"]):
                models["lstm"], models["lstm_scaler"], models["lstm_target_scaler"] = load_lstm_model(
                    paths["lstm"], paths["lstm_scaler"], paths["lstm_target_scaler"]
                )

            if any(models.values()):
                log("Local models loaded successfully", level='success')
                return True
        except Exception as e:
            log(f"Error loading local models: {e}", level='error')
        return False

    def try_load_remote():
        """Attempt to load models from remote storage."""
        try:
            store_id = st.session_state.store_id
            item_id = st.session_state.item_id

            for key in ["xgb", "lstm", "lstm_scaler", "lstm_target_scaler"]:
                if (is_xgb_enabled and key == "xgb") or (is_lstm_enabled and key.startswith("lstm")):
                    paths[key] = load_remote_file(key, store_id, item_id)

            if is_xgb_enabled and paths["xgb"]:
                models["xgb"] = load_xgboost_model(paths["xgb"])
                os.remove(paths["xgb"])

            if is_lstm_enabled and all(paths[key] for key in ["lstm", "lstm_scaler", "lstm_target_scaler"]):
                models["lstm"], models["lstm_scaler"], models["lstm_target_scaler"] = load_lstm_model(
                    paths["lstm"], paths["lstm_scaler"], paths["lstm_target_scaler"]
                )
                for key in ["lstm", "lstm_scaler", "lstm_target_scaler"]:
                    os.remove(paths[key])

            if any(models.values()):
                log("Remote models loaded successfully", level='success')
                return True
        except Exception as e:
            log(f"Error loading remote models: {e}", level='error')
        return False

    def train_models():
        """Train new models if no pre-trained ones are found."""
        if df_train is None:
            raise ValueError("Training data is required to train new models")

        log("Training new models...")
        spinner_container = st.empty()

        if is_xgb_enabled:
            with spinner_container:
                with st.spinner('Training XGBoost model...'):
                    models["xgb"] = train_xgboost_model(store_id, item_id, df_train, df_stores, df_items)

        if is_lstm_enabled:
            with spinner_container:
                with st.spinner('Training LSTM model...'):
                    models["lstm"], models["lstm_scaler"], models["lstm_target_scaler"] = train_lstm_model(
                        store_id, item_id, df_train, df_stores, df_items
                    )

        log("Models trained and saved successfully", level='success')

    # Attempt to load local, then remote, then train if necessary
    if not try_load_local() and not try_load_remote():
        train_models()

    save_to_session(store_id, item_id, models["xgb"], models["lstm"], models["lstm_scaler"], models["lstm_target_scaler"])
    return get_from_session(store_id, item_id)

def get_models (model_data):
    if not model_data:
        raise ValueError("Model data not found")

    xgb_model = model_data.get("xgb_model")
    lstm_model = model_data.get("lstm_model")
    feature_scaler = model_data.get("lstm_scaler")
    target_scaler = model_data.get("lstm_target_scaler")
    return xgb_model, lstm_model, feature_scaler, target_scaler

def predict(store_id, item_id, input_data, df_stores, df_items, df_train):
    """Make predictions using XGBoost and/or LSTM models based on enabled flags."""
    use_xgb=TRAIN_CONF["use_xgb"]
    use_lstm=TRAIN_CONF["use_lstm"]      
    log("\n=== Making Predictions ===")
    session_obj = get_from_session(store_id, item_id)
    if session_obj is None:
        session_obj = load_models(store_id, item_id, df_stores, df_items, df_train)
    
    xgb_model, lstm_model, feature_scaler, target_scaler = get_models(session_obj)

    feature_columns = [
        'store_nbr', 'item_nbr', 'month', 'day', 'weekofyear', 'dayofweek',
        'rolling_mean_7', 'rolling_std_7', 'lag_1', 'lag_7',
        'store_type', 'store_cluster', 'item_family', 'item_class', 'item_perishable'
    ]


    log("\nPreparing input data for models...")
    input_data_array = np.asarray(input_data[feature_columns], dtype=np.float32)
    predictions = {"xgb_pred": None, "lstm_pred": None}

    if use_xgb:
        def predict_xgb(input_data_array):
            """Make prediction using XGBoost model."""
            log("Making XGBoost prediction...")
            pred = xgb_model.predict(input_data_array)
            log(f"XGBoost prediction: {pred[-1]:.2f}")           
            return pred

        predictions["xgb_pred"] = predict_xgb(input_data_array)

    if use_lstm:

        def predict_lstm(input_data_array):
            """Make prediction using LSTM model for all input values."""
            log("Preparing data for LSTM prediction...")
            lstm_input_scaled = feature_scaler.transform(input_data_array)
            seq_length = TRAIN_CONF["seq_length"]
            total_length = len(lstm_input_scaled)
                
            # Ensure there are enough data points to create sequences
            if total_length < seq_length:
                raise ValueError("Not enough data to create sequences")

            sequences = []
            for i in range(total_length - seq_length + 1):
                sequences.append(lstm_input_scaled[i:i+seq_length])

            sequences = np.array(sequences)
            sequences = sequences.reshape(sequences.shape[0], seq_length, -1)  # Shape: (num_sequences, seq_length, num_features)
            sequences = tf.convert_to_tensor(sequences, dtype=tf.float32)            
            predictions = lstm_model.predict(sequences)
            predictions = target_scaler.inverse_transform(predictions)            
            all_predictions = predictions.flatten()            
            return all_predictions       

        predictions["lstm_pred"] = predict_lstm(input_data_array)

    log("\n=== Prediction Complete ===")
    return predictions, xgb_model, lstm_model

def load_lstm_model(model_path, feature_scaler_path, target_scaler_path):
    """Load the LSTM model and its scalers"""
    log("Loading LSTM model and scalers...")

    model = load_model(model_path)

    with open(feature_scaler_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    log("LSTM model and scalers loaded successfully", level='success')
    return model, feature_scaler, target_scaler
