import os
import json
import base64

import boto3
import mlflow

kinesis_client = boto3.client("kinesis")

PREDICTION_STREAM_NAME = os.getenv("PREDICTION_STREAM_NAME", "mlops_ride_events")

RUN_ID = os.getenv("RUN_ID")

# logged_model = 's3://mlflow-leo-bucket-useast2/1/models/m-ea1dfabd247448bbaaba443d31ce73b9/artifacts'
logged_model = f"s3://mlflow-leo-bucket-useast2/1/models/{RUN_ID}/artifacts"
# logged_model = 'models:/ride_duration_model/1'
model = mlflow.pyfunc.load_model(logged_model)

TEST_RUN = os.getenv("TEST_RUN", "false") == "true"


def prepare_feature(ride):
    feature = {}
    feature["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    feature["trip_distance"] = ride["trip_distance"]

    return feature


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


def lambda_handler(event, context):

    # print(json.dumps(event))

    predictions = []

    for record in event["Records"]:
        encoded_data = record["kinesis"]["data"]
        encoded_data = base64.b64decode(encoded_data).decode("utf-8")
        ride_event = json.loads(encoded_data)

        print(ride_event)
        ride = ride_event["ride"]
        ride_id = ride_event["ride_id"]

        features = prepare_feature(ride)
        prediction = predict(features)

        predictions_event = {
            "model": "ride_duration_prediuction_model",
            "version": "123",
            "prediction": {"ride_duration": prediction, "ride_id": ride_id},
        }

        if not TEST_RUN:
            kinesis_client.put_record(
                StreamName=PREDICTION_STREAM_NAME,
                Data=json.dumps(predictions_event),
                PartitionKey=str(ride_id),
            )

        predictions.append(predictions_event)

    return {"predictions": predictions}
