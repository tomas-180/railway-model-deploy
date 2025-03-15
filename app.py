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
    # Flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json.
    obs_dict = request.get_json()
    
    _id = obs_dict.get('id')
    observation = obs_dict.get('observation')
    
    if not observation:
        return jsonify({'error': 'Observation is missing!'}), 400

    # Example: check that required keys are in the observation and that their values are valid.
    required_columns = columns  # The columns expected by your model
    for col in required_columns:
        if col not in observation:
            return jsonify({'error': f'Missing required field: {col}'}), 400
        
        value = observation[col]
        
        # Add validation for specific fields
        if col == 'age':
            try:
                observation[col] = float(value)
            except ValueError:
                return jsonify({'error': f'Invalid value for age: {value}'}), 400
        
        elif col == 'hours-per-week':
            try:
        # Validate that hours-per-week is a number
                observation[col] = float(value)
            except ValueError:
                return jsonify({'error': f'Invalid value for hours-per-week: {value}. Please provide a valid number.'}), 400
        
        elif col == 'education':
            if value not in ['Bachelors', 'Masters', 'PhD', 'HS-grad']:  # Modify as per your valid values
                return jsonify({'error': f'Invalid value for education: {value}'}), 400
        
        # You can add more checks for other fields as well
    
    # Now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline.
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)

    # Now get ourselves an actual prediction of the positive class.
    try:
        proba = pipeline.predict_proba(obs)[0, 1]
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    response = {'proba': proba}

    # Save the prediction to the database
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
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




