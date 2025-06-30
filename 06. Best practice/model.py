import json
import base64
import boto3
import os
import mlflow

def get_model_location(run_id):

    model_location = os.getenv('MODEL_LOCATION')

    if model_location is None:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET','mlflow-leo-bucket-useast2')
    experiment_id = os.getenv('EXPERIMENT_ID', '1')

    model_location = f's3://{model_bucket}/{experiment_id}/models/{run_id}/artifacts'
    return model_location

def load_model(run_id):
    model_path = get_model_location(run_id)
    model = mlflow.pyfunc.load_model(model_path)
    return model

def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    ride_event = json.loads(decoded_data)
    return ride_event

class ModelService:

    def __init__(self, model, model_version=None, callbacks=None):
        self.model = model
        self.model_version = model_version
        self.callbacks = callbacks or []

    def prepare_feature(self,ride):
        feature = {}
        feature['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
        feature['trip_distance'] = ride['trip_distance']

        return feature

    def predict(self, features):
        preds = self.model.predict(features)
        return float(preds[0])


    def lambda_handler(self, event):

        predictions = []

        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            encoded_data = base64.b64decode(encoded_data).decode('utf-8')
            ride_event = json.loads(encoded_data)

            print(ride_event)
            ride = ride_event['ride']
            ride_id = ride_event['ride_id']

            features = self.prepare_feature(ride)
            prediction = self.predict(features)

            predictions_event = {
                'model': 'ride_duration_prediuction_model',
                'version' : '123',
                'prediction' : {
                    'ride_duration': prediction,
                    'ride_id': ride_id
                }
            }

            for callback in self.callbacks:
                callback(predictions_event)

            predictions.append(predictions_event)

        return {
            'predictions' : predictions
        }
    

class KinesisCallback:

    def __init__(self, kinesis_client, prediction_stream_name):
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name

    def put_record(self, prediction_event):
        ride_id = prediction_event['prediction']['ride_id']

        self.kinesis_client.put_record(
            StreamName=self.prediction_stream_name,
            Data=json.dumps(prediction_event),
            PartitionKey=ride_id,
        )

def create_kinesis_client():
    endpoint_url = os.getenv('KINESIS_ENDPOINT_URL', None)

    if endpoint_url is None:
        return boto3.client('kinesis')
    
    return boto3.client('kinesis', endpoint_url=endpoint_url)


def init(prediction_stream_name: str, run_id: str, test_run: bool):
    model = load_model(run_id)

    callbacks = []

    if not test_run:
        kinesis_client = create_kinesis_client()
        kinesis_callback = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callback.put_record)

    model_service = ModelService(model=model,model_version=run_id , callbacks=callbacks)

    return model_service