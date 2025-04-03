**Sales Forecasting with XGBoost and LSTM**

[Live Demo](https://retaildemandforecastgit-6hz7vt22tpvh3j8z4knpg3.streamlit.app/)

## Overview


**Overview**

This project aims to build a sales forecasting model using machine learning techniques, specifically **XGBoost** and **Long Short-Term Memory (LSTM)**. The goal is to predict future sales based on historical sales data, utilizing both traditional machine learning models (XGBoost) and deep learning models (LSTM) to improve the forecasting accuracy.

**Project Structure**

- data/: Folder containing the raw and processed data files.
- notebooks/: Jupyter notebooks used for exploratory data analysis (EDA), data preprocessing, and model training.
- models/: Folder containing trained models and scripts for model training and evaluation.
- app/: Streamlit app to visualize the sales forecasting results and interact with the models.
- utils/: Helper functions for logging, model management, and other utilities.
- requirements.txt: Python dependencies required for the project.
- README.md: Project documentation.

**Technologies Used**

- **XGBoost**: A gradient boosting algorithm used for the sales forecasting model.
- **LSTM**: A type of Recurrent Neural Network (RNN) used for time-series forecasting.
- **Streamlit**: A Python framework for creating interactive web applications.
- **MLflow**: A tool for tracking experiments, logging models, and managing the machine learning lifecycle.
- **Pandas**: Data manipulation and analysis.
- **Numpy**: Numerical operations.
- **Matplotlib/Seaborn**: Visualization tools.

**Data**

The dataset used in this project includes historical sales data, item and store details, oil prices, and holiday events. The data is preprocessed and cleaned before being used to train the models.

Key features in the dataset:

- **Store Details**: Information about the stores, such as location and type.
- **Item Details**: Information about the items being sold.
- **Sales Data**: Historical sales transactions.
- **Oil Prices**: Prices of oil affecting product sales.
- **Holidays and Events**: Special dates that might impact sales.

**Model Training**

**XGBoost Model**

The XGBoost model is trained using historical sales data and other relevant features. The model leverages gradient boosting techniques to predict sales.

Key Steps:

1. Data preprocessing: Encoding categorical variables and scaling numerical features.
2. Hyperparameter tuning: Optimizing the XGBoost model's parameters for better accuracy.
3. Model training: Fitting the XGBoost model on the training data.
4. Model evaluation: Testing the model's performance on a validation set.

**LSTM Model**

The LSTM model is a deep learning approach that takes into account the temporal nature of the sales data. It uses previous sales values and other features to forecast future sales.

Key Steps:

1. Data preprocessing: Normalizing the data and creating time-series sequences.
2. LSTM model architecture: Building the LSTM network using Keras.
3. Model training: Training the LSTM model using the training data.
4. Model evaluation: Testing the LSTM model on unseen data.

**How to Run**

**Setup**

1. Clone the repository:
2. git clone <https://github.com/sultanmr/retail_demand_forecast.git>
3. cd sales-forecasting
4. Install dependencies:
5. pip install -r requirements.txt
6. Set up **MLflow** and **ngrok** for logging and tracking experiments.

**Running the Streamlit App**

1. Navigate to the app/ directory.
2. Run the Streamlit app:
3. streamlit run app.py
4. Open the app in your browser to interact with the sales forecasting models.

**Training Models**

To train the models, run the following notebooks in the notebooks/ directory:

1. **EDA and Preprocessing**: Explore the data and perform necessary preprocessing.
2. **XGBoost Model**: Train the XGBoost model.
3. **LSTM Model**: Train the LSTM model.

**Model Logging with MLflow**

The models (XGBoost and LSTM) are logged using **MLflow** for experiment tracking, including metrics, parameters, and models.

- For XGBoost, the model and evaluation results are logged with mlflow.xgboost.log_model.
- For LSTM, the model and hyperparameters are logged with mlflow.keras.log_model.

**Results**

The models' performance is evaluated using metrics such as **RMSE (Root Mean Squared Error)** and **MAE (Mean Absolute Error)**. The goal is to compare the performance of the XGBoost and LSTM models to determine which model provides better sales forecasting accuracy.

**Future Work**

- Incorporate **external features** such as weather data or economic indicators that might affect sales.
- Deploy the models to a production environment for real-time sales forecasting.

**License**

This project is licensed under the MIT License