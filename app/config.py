import datetime
# app/config.py

MLFLOW_CONFIG = {
"MLFLOW_TRACKING_USERNAME": "sultanmr",
"MLFLOW_TRACKING_PASSWORD": "79869daf4b3a7f1cdc2fffb3cd3d867c67454e2a",
"MLFLOW_TRACKING_URI" : "https://dagshub.com/sultanmr/my-first-repo.mlflow/",
"MLFLOW_EXPERIMENT_NAME": "retail_forecast_experiment",
"MLFLOW_MODEL_NAME":  "retail_forecast_model",
}

DATA_PATH = "data/csv_data"
PARQUET_DATA_PATH = "data/parquet_data"
MODEL_PATH = 'learned_models/'
NGROK_TOKEN = "2uj6ia3aPjGFf0A4arHdWFTU4xl_3HoYKVeNMkSd31veLbVKc"

PARQUET_URLS = {
    "stores": "https://www.dropbox.com/scl/fi/qg59hyt4j2enf8chirmxc/stores.parquet?rlkey=k4qpbts5z5z7e9vj88svz1l1p&st=7js28mm3&dl=1",
    #"items": "https://www.dropbox.com/scl/fi/mddbhs0a5mopz3ee3f9r0/items.parquet?rlkey=m1jfge4597in5ebopirupb015&st=am4styzm&dl=1",
    "items": "https://www.dropbox.com/scl/fi/3lk240vajdikohn3js88g/items_filtered.parquet?rlkey=mb02fjipvosjrxrmy7ygtsqxc&st=zpyohhmn&dl=1",
    #"transactions": "https://www.dropbox.com/scl/fi/u03xnhjv4er3m9iyrfyfy/transactions.parquet?rlkey=fmyx972zcmhpsaglzr6ng8jhi&st=a321t4zo&dl=1",
    "transactions": "https://www.dropbox.com/scl/fi/x2cn1hn233mapfqugato2/transactions_filtered.parquet?rlkey=tbwlo8c4gd87b6td0xtu6r9xq&st=14at9dwc&dl=1",
    "oil": "https://www.dropbox.com/scl/fi/22vwswwqquuidamrs13ph/oil.parquet?rlkey=uwmuckrefznoba8yy23v68901&st=r5s8ppn0&dl=1",
    "holidays_events": "https://www.dropbox.com/scl/fi/8bbukgb8yiqhre2hsa9tn/holidays_events.parquet?rlkey=k16gfluy8ptiuwmz3r7qrbmph&st=k4xopv3u&dl=1",
    #"train": "https://www.dropbox.com/scl/fi/era49npmh6z06mop7wvcu/train.parquet?rlkey=6hex970ie7jgp12l6aj3lefwd&st=88r39fbh&dl=1",
    "train": "https://www.dropbox.com/scl/fi/g4yqq4dvuaa7jj7s6yioz/train_filtered.parquet?rlkey=m7rpku6f7ranrydw9givg0m4f&st=48azgjkf&dl=1",
}

MODEL_URLS = {
    "1-564533" : {
        "xgb": "https://www.dropbox.com/scl/fi/nneo2umneuf0gsxpjsryo/xgb_model.xgb?rlkey=s07om4s6zs5u15lhn6gvooq7q&st=c2jikpfu&dl=1",
        "lstm": "https://www.dropbox.com/scl/fi/k2v29l6f4hw413i0y6eva/lstm_model.h5?rlkey=hlsay0e1uqkg720tkobjv2t3x&st=w24uunyt&dl=1",
        "lstm_scaler": "https://www.dropbox.com/scl/fi/uoe6ji0y7q1z6hozf3pn0/lstm_feature_scaler.pkl?rlkey=jz59hjjht4we70peloam7iidi&st=926iabak&dl=1",
        "lstm_target_scaler": "https://www.dropbox.com/scl/fi/5kqz23ywda23syby4oygp/lstm_target_scaler.pkl?rlkey=vwhcvftrgwca5849igemiqkb2&st=hv3egctf&dl=1",
    },
    "1-838216" : {
        "xgb": "https://www.dropbox.com/scl/fi/3sk4agmx53edtubjpcnje/xgb_model.xgb?rlkey=88rsi3tijmad01ql03rsukchk&st=pzm003bv&dl=1",
        "lstm": "https://www.dropbox.com/scl/fi/pdfreda9ztunx5cvjsaet/lstm_model.h5?rlkey=zsnqslxynmcsf10gwye49l5w4&st=2qyvdxku&dl=1",
        "lstm_scaler": "https://www.dropbox.com/scl/fi/tu2azhzjh0ql6nevebb7y/lstm_feature_scaler.pkl?rlkey=ejlmmf04rl27e7eed7pj3hnd0&st=ianqlm30&dl=1",
        "lstm_target_scaler": "https://www.dropbox.com/scl/fi/1elrk6609uo2p2ddiih2x/lstm_target_scaler.pkl?rlkey=su4f48x8mh5ctzybhti7k56o7&st=l2zkcdry&dl=1",
    }
    ,
    "1-582865" : {
        "xgb": "https://www.dropbox.com/scl/fi/gku2e9sh2aqzhousojqj9/xgb_model.xgb?rlkey=f6lx1b6k7grhv9t20ehukwaih&st=ynpbp8jh&dl=1",
        "lstm": "https://www.dropbox.com/scl/fi/89kh3wi1ru9r21ucd8h5l/lstm_model.h5?rlkey=bse781bhw8ibj1jhcupi02vjl&st=roh9it4w&dl=1",
        "lstm_scaler": "https://www.dropbox.com/scl/fi/jsx3sytz5gtnqj5rnrfd3/lstm_feature_scaler.pkl?rlkey=hzy167haf7wn0ehkr6367dbhm&st=5ropvr2s&dl=1",
        "lstm_target_scaler": "https://www.dropbox.com/scl/fi/8wcyb7g81po8qthfs6kyw/lstm_target_scaler.pkl?rlkey=q174rkx13scduyd44ycyut4c5&st=la55pzwl&dl=1",
    }
    ,
    "1-364606" : {
        "xgb": "https://www.dropbox.com/scl/fi/tnl4sdurkztow952j69da/xgb_model.xgb?rlkey=cbqziewtdiy66ekboasjt3ewz&st=ynqmmmw8&dl=1",
        "lstm": "https://www.dropbox.com/scl/fi/i03hu0ruyp9dxfhkfl78d/lstm_model.h5?rlkey=retg31v7xiubu5pcobqpmztw9&st=7jrf3dft&dl=1",
        "lstm_scaler": "https://www.dropbox.com/scl/fi/da8ujb1qrmzr36anurk3u/lstm_feature_scaler.pkl?rlkey=tu0mcre9cct1z4dtvoxgxaghc&st=g1wjxcey&dl=1",
        "lstm_target_scaler": "https://www.dropbox.com/scl/fi/ku33d0we1iwo4fyual847/lstm_target_scaler.pkl?rlkey=lnlcmo7pnpgjrml49ftwvqppy&st=er01u7o0&dl=1",
    }    
}

DEFAULTS = {
    "store_id": 1,
    "item_id": 564533,
    "date": datetime.date(2014, 3, 1),
    "min_date": datetime.date(2013, 1, 1),
    "max_date": datetime.date(2014, 6, 7),    
}
ITEM_IDS = [564533,838216,582865,364606]

TRAIN_CONF = {
    "use_xgb": True,
    "use_lstm": True,
    "total_trials": 5,  
    "learning_rate": 0.001,
    "lstm_units": 64,
    "dropout_rate": 0.2,
    "train_size": 0.8,
    "lstm_n_splits": 5,
    "random_state": 42,
    "seq_length": 30,
    "batch_size": 128,
    "n_features": 12,
}