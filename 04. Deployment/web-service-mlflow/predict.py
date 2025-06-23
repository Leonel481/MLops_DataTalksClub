import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify


RUN_ID = '1b3f47c608fe4d6f91b174f6ac5c0094'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


path = client.download_artifacts(run_id = RUN_ID, path = 'dict_vectorizer.bin')
print(f'dowloading the dict vectorizer to {path}')

with open(path, 'rb') as f:
    dv = pickle.load(f)

logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_feature(ride):
    feature = {}
    feature['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    feature['trip_distance'] = ride['trip_distance']

    return feature


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-predictor')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    feature = prepare_feature(ride)
    pred = predict(feature)

    result = {
        'duration': pred,
        'model_version': RUN_ID,
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)