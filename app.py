import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Deserialize the JSON payload
    obs_dict = request.get_json()
    _id = obs_dict.get('id')
    observation = obs_dict.get('observation')

    # Validate the observation
    if not observation:
        return jsonify({'error': 'Observation data is missing'}), 400

    # Check if all required fields are present
    required_fields = list(dtypes.keys())  # Use the keys from dtypes as required fields
    for field in required_fields:
        if field not in observation:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    # Validate data types using the dtypes dictionary
    for field, expected_dtype in dtypes.items():
        value = observation.get(field)
        if value is None:
            continue  # Skip if the field is optional (not in required_fields)

        try:
            # Attempt to convert the value to the expected data type
            if expected_dtype == 'int64':
                observation[field] = int(value)
            elif expected_dtype == 'float64':
                observation[field] = float(value)
            elif expected_dtype == 'object':
                observation[field] = str(value)
            else:
                return jsonify({'error': f'Unsupported data type for field {field}: {expected_dtype}'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': f'Invalid value for field {field}. Expected type: {expected_dtype}'}), 400

    # Convert the observation into a DataFrame
    try:
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    except Exception as e:
        return jsonify({'error': f'Invalid observation data: {str(e)}'}), 400

    # Make a prediction
    try:
        proba = pipeline.predict_proba(obs)[0, 1]
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Save the prediction to the database
    response = {'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=json.dumps(observation)  # Store the observation as a JSON string
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = f'Observation ID {_id} already exists'
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()

    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = f'Observation ID {obs['id']} does not exist'
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)





