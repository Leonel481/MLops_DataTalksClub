import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from ml_utils import train_model, process_data

def pipeline_proces_data(**kwargs):
    year = kwargs['params']['year']
    month = kwargs['params']['month']
    _artifacts = process_data(year=year, month=month)
    kwargs['ti'].xcom_push(key='artifacts_model', value=_artifacts)

def pipeline_model(**kwargs):
    artifacts = kwargs['ti'].xcom_pull(key='artifacts_model', task_ids='prepare_data')
    _model = train_model(artifacts)