# app/config.py

DATA_PATH = "data/"
your_file_id_for_stores_csv = '1YgE0K5mrlsHg8QmWnozuAsX9YPk4EYgf'
your_file_id_for_items_csv = '1gsts50e_UfBPonkAg67VtLE44zG85It4'
your_file_id_for_transactions_csv = '1-4RSVKRYYlnIY4Cd-G_ZYbkJqt-rPXVx'
your_file_id_for_oil_csv = '1QDzb7nsVN_1qISS6iTBel4gwlYNtrMjU'
your_file_id_for_holidays_csv = '1asSLMupocsH8CEoykirb74BwppGNNQLG'
your_file_id_for_train_csv = '1-2t5PWG6ngtqNMCwi6Nyd3IAGl_1__eA'

# Google Drive links for each file (replace with actual file IDs)
GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",
    "transactions": f"https://drive.google.com/uc?id={your_file_id_for_transactions_csv}",
    "oil": f"https://drive.google.com/uc?id={your_file_id_for_oil_csv}",
    "holidays_events": f"https://drive.google.com/uc?id={your_file_id_for_holidays_csv}",
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}"
}

MODEL_PATH = 'model/'
your_file_id_for_xgboost_model_xgb = "10s41mHM-6CopwrAAGYRWSuXinjz0FE4-"  # Replace with the actual file ID for model.xgb from MLFlow aritifacts
GOOGLE_DRIVE_LINKS_MODELS = {
    "xgboost_model": f"https://drive.google.com/uc?id={your_file_id_for_xgboost_model_xgb}"
}