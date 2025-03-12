import pickle  # Import pickle to handle model loading
import xgboost as xgb
from app.config import MODEL_PATH, GOOGLE_DRIVE_LINKS_MODELS
from data.data_utils import download_file

def load_model(model_path=MODEL_PATH):
    """Downloads necessary data from Google Drive and loads a pre-trained model."""
    # Define paths to model files
    files = {
        #"xgboost_model": f"{model_path}xgboost_model.pkl"
        "xgboost_model": f"{model_path}model.xgb"
    }

    # Download files if they don’t exist
    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS_MODELS[key])

    # Load the pre-trained model from a pickle file
    #with open(files["xgboost_model"], 'rb') as f:
        #xgboost_model = pickle.load(f)
   
   # Load the pre-trained XGBoost model from xgb file
    xgboost_model = xgb.XGBRegressor() 
    xgboost_model.load_model(files["xgboost_model"])
    #print("=== Model type",type(xgboost_model))
    return xgboost_model

def predict(model, input_data):
    """Runs prediction on input data using the pre-trained model."""
    # Drop the original 'date' column
    input_data = input_data.drop(columns=['date'])
    # Drop the 'unit_sales' column
    input_data = input_data.drop(columns=['unit_sales'])

    prediction = model.predict(input_data)
    return prediction
