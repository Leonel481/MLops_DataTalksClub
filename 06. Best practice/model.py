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

    def prepare_feature(ride):
        feature = {}
        feature['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
        feature['trip_distance'] = ride['trip_distance']

        return feature


    def predict(features):
        preds = model.predict(features)
        return float(preds[0])


    def lambda_handler(event, context):

        # print(json.dumps(event))

        predictions = []

        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            encoded_data = base64.b64decode(encoded_data).decode('utf-8')
            ride_event = json.loads(encoded_data)

            print(ride_event)
            ride = ride_event['ride']
            ride_id = ride_event['ride_id']

            features = prepare_feature(ride)
            prediction = predict(features)

            predictions_event = {
                'model': 'ride_duration_prediuction_model',
                'version' : '123',
                'prediction' : {
                    'ride_duration': prediction,
                    'ride_id': ride_id
                }
            }

            if not TEST_RUN:
                kinesis_client.put_record(
                    StreamName = PREDICTION_STREAM_NAME,
                    Data = json.dumps(predictions_event),
                    PartitionKey = str(ride_id)
                )

            predictions.append(predictions_event)

        return {
            'predictions' : predictions
        }

    kinesis_client = boto3.client('kinesis')
    class ModelService():


    def init(prediction_stream_name: str, run_id: str, test_run: bool):

        return 