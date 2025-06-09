from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow.plugins_manager import AirflowPlugin
from plugins.task_pipeline import pipeline_proces_data, pipeline_model


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'depends_on_past': False,
    'retries': 1,
}


with DAG(
    dag_id='train_nyc_taxi_model',
    default_args=default_args,
    schedule=None,
    catchup=False,
    params={'year': 2023, 'month': 3},
    tags=['mlops', 'mlflow', 'xgboost'],
) as dag:

    t1 = PythonOperator(
        task_id='prepare_data',
        python_callable=pipeline_proces_data,
    )

    t2 = PythonOperator(
        task_id='train_model',
        python_callable=pipeline_model,
    )

    t1 >> t2