import pandas as pd
import os
import glob
import requests
import io

def convert_csv_to_parquet(csv_files, output_folder, compression="snappy"):
    """
    Convert a list of CSV files into compressed Parquet files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    parquet_files = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        parquet_file = os.path.join(output_folder, os.path.basename(csv_file).replace(".csv", ".parquet"))
        df.to_parquet(parquet_file, engine="pyarrow", compression=compression, index=False)
        parquet_files.append(parquet_file)
        print(f"✅ Converted: {csv_file} → {parquet_file}")

    return parquet_files



def load_parquet_files(parquet_folder):
    """
    Load all Parquet files from a given folder into a single DataFrame.
    """
    parquet_files = glob.glob(os.path.join(parquet_folder, "*.parquet"))
    df = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in parquet_files], ignore_index=True)
    print(f"✅ Loaded {len(parquet_files)} Parquet files into DataFrame")
    return df


def load_parquet_files_separately(parquet_folder):
    """
    Load each Parquet file into a separate DataFrame and store in a dictionary.
    """
    parquet_files = glob.glob(os.path.join(parquet_folder, "*.parquet"))
    dataframes = {}

    for file in parquet_files:
        file_name = os.path.basename(file).replace(".parquet", "")  # Extract filename without extension
        dataframes[file_name] = pd.read_parquet(file, engine="pyarrow")  # Load into a DataFrame
        print(f"✅ Loaded {file} into DataFrame: {file_name}")

    return dataframes
    

PARQUET_URLS_LIST = {
    "stores": "https://www.dropbox.com/scl/fi/qg59hyt4j2enf8chirmxc/stores.parquet?rlkey=k4qpbts5z5z7e9vj88svz1l1p&st=7js28mm3&dl=1",
    "items": "https://www.dropbox.com/scl/fi/mddbhs0a5mopz3ee3f9r0/items.parquet?rlkey=m1jfge4597in5ebopirupb015&st=am4styzm&dl=1",
    "transactions": "https://www.dropbox.com/scl/fi/u03xnhjv4er3m9iyrfyfy/transactions.parquet?rlkey=fmyx972zcmhpsaglzr6ng8jhi&st=a321t4zo&dl=1",
    "oil": "https://www.dropbox.com/scl/fi/22vwswwqquuidamrs13ph/oil.parquet?rlkey=uwmuckrefznoba8yy23v68901&st=r5s8ppn0&dl=1",
    "holidays_events": "https://www.dropbox.com/scl/fi/8bbukgb8yiqhre2hsa9tn/holidays_events.parquet?rlkey=k16gfluy8ptiuwmz3r7qrbmph&st=k4xopv3u&dl=1",
    "train": "https://www.dropbox.com/scl/fi/era49npmh6z06mop7wvcu/train.parquet?rlkey=6hex970ie7jgp12l6aj3lefwd&st=88r39fbh&dl=1",
}

def load_parquet_from_url():
    """
    Loads Parquet files directly from URLs into a dictionary of DataFrames.
    """
    dataframes = {}

    for name, url in PARQUET_URLS_LIST.items():
        print(f"⬇️ Loading {name} ...")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                df = pd.read_parquet(io.BytesIO(response.content), engine="pyarrow")
                dataframes[name] = df              
            else:
                print(f"❌ Failed to load {name} (HTTP {response.status_code})")
        except Exception as e:
            print(f"⚠️ Error loading {name}: {e}")

    return dataframes

def convert_all():
    # List of CSV files
    csv_files = [    
        "train.csv",
        "stores.csv",
        "items.csv",
        "transactions.csv",
        "oil.csv",
        "holidays_events.csv",
    ] 

    output_folder = "parquet_data"
    # Convert CSVs to Parquet
    parquet_files = convert_csv_to_parquet(csv_files, output_folder, compression="snappy")

    # Load all Parquet files into a DataFrame
    df_all = load_parquet_files(output_folder)

    # Display first few rows
    print(df_all.head())


dfs = load_parquet_from_url()
print (dfs["train"].head())