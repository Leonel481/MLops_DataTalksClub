import os

import model

PREDICTION_STREAM_NAME = os.getenv('PREDICTION_STREAM_NAME', 'mlops_ride_events')
RUN_ID = os.getenv('RUN_ID')
TEST_RUN = os.getenv('TEST_RUN', 'false') == 'true'

# logged_model = f's3://mlflow-leo-bucket-useast2/1/models/{RUN_ID}/artifacts'

model_service = model.init(
    prediction_stream_name=PREDICTION_STREAM_NAME, run_id=RUN_ID, test_run=TEST_RUN
)


def lambda_handler(event, context):
    return model_service.lambda_handler(event)
