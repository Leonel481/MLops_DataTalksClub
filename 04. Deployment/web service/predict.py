import pickle
from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f:
    (dv, model) = pickle.load(f)

def prepare_feature(ride):
    feature = {}
    feature['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    feature['trip_distance'] = ride['trip_distance']

    return feature


def predict(feature):
    X = dv.transform([feature])
    y_pred = model.predict(X)
    return float(y_pred[0])


app = Flask('duration-predictor')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    feature = prepare_feature(ride)
    pred = predict(feature)

    result = {
        'duration': pred,
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)