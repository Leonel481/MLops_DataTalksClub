import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error


import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.models import infer_signature

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def init_mlflow():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi-experiment")


def read_dataframe(year:int, month:int) -> pd.DataFrame:

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df:pd.DataFrame, dv:Optional[DictVectorizer]=None) -> Tuple[csr_matrix, DictVectorizer]:
    
    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:        
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def process_data(year:int, month:int) -> dict:

    init_mlflow()

    print('se inicializa el proceso de procesamiento de datos') 

    next_year = year if month < 12 else year +1
    next_month = month + 1 if month <12 else 1 

    df_train =  read_dataframe(year=year, month=month)
    df_val   =  read_dataframe(year=next_year, month=next_month)

    records_train = len(df_train)
    print(f'records:{records_train}')

    X_train, dv = create_X(df_train)
    X_val, _    = create_X(df_val, dv)

    target  = 'duration'
    y_train = df_train[target].values
    y_val   = df_val[target].values

    print('Se han creado las matrices X_train, X_val, y_train, y_val')

    # Carpeta temporal para guardar artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    with open(artifacts_dir / "X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(artifacts_dir / "X_val.pkl", "wb") as f:
        pickle.dump(X_val, f)
    with open(artifacts_dir / "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(artifacts_dir / "y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)
    with open(artifacts_dir / "dv.pkl", "wb") as f:
        pickle.dump(dv, f)

    # artifact_root = os.environ.get("ARTIFACT_ROOT", "s3://mlflow/")

    
    # Iniciar MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("data_year", year)
        mlflow.log_param("data_month", month)

        # Subir los artifacts a MinIO vía MLflow y capturar el run_id
        mlflow.log_artifacts(str(artifacts_dir))
        run_id = run.info.run_id

    mlflow.end_run()

    return run_id

    # return {
    # "X_train": "artifacts/X_train.pkl",
    # "X_val": "artifacts/X_val.pkl",
    # "y_train": "artifacts/y_train.pkl",
    # "y_val": "artifacts/y_val.pkl",
    # "dv": "artifacts/dv.pkl",
    # "run_id":run_id
    # }


def train_model(run_id: str) -> None:

    init_mlflow()
    
    print(f"Training model with run_id: {run_id}")

    X_train = pickle.load(open(download_artifacts(run_id=run_id, artifact_path="X_train.pkl"), "rb"))
    X_val   = pickle.load(open(download_artifacts(run_id=run_id, artifact_path="X_val.pkl"), "rb"))
    y_train = pickle.load(open(download_artifacts(run_id=run_id, artifact_path="y_train.pkl"), "rb"))
    y_val   = pickle.load(open(download_artifacts(run_id=run_id, artifact_path="y_val.pkl"), "rb"))
    dv      = pickle.load(open(download_artifacts(run_id=run_id, artifact_path="dv.pkl"), "rb"))

    print('se inicializa el proceso de entrenamiento del modelo')

    
    with mlflow.start_run(run_id=run_id) as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:squarederror',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        signature = infer_signature(X_val, y_pred)
        input_example = X_val[:1]

        booster.save_model("models/model.json")
        mlflow.log_artifact("models/model.json", artifact_path="models_mlflow")

        mlflow.xgboost.log_model(booster, 
                                 artifact_path="models_mlflow",
                                 signature=signature,
                                 input_example=input_example)

        return run_id