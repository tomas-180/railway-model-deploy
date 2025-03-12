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
# Configuração do Banco de Dados

# Conecta ao SQLite (ou a um banco de dados remoto, se DATABASE_URL estiver definido)
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)  # ID único da observação
    observation = TextField()  # Observação em formato JSON
    proba = FloatField()  # Probabilidade prevista
    true_class = IntegerField(null=True)  # Classe verdadeira (inicialmente nula)

    class Meta:
        database = DB

# Cria a tabela no banco de dados, se não existir
DB.create_tables([Prediction], safe=True)

########################################
# Carregando o Modelo e Configurações

# Carrega as colunas esperadas
with open('columns.json') as fh:
    columns = json.load(fh)

# Carrega o pipeline do modelo
pipeline = joblib.load('pipeline.pickle')

# Carrega os tipos de dados esperados
with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

########################################
# Configuração do Flask

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Recebe o JSON da requisição
    obs_dict = request.get_json()

    # Extrai o ID e a observação
    _id = obs_dict.get('id')
    observation = obs_dict.get('observation')

    # Valida se o ID e a observação estão presentes
    if not _id or not observation:
        return jsonify({"error": "ID e observation são obrigatórios"}), 400

    # Valida os campos da observação
    required_fields = ['age', 'education', 'hours-per-week', 'native-country']
    if not all(field in observation for field in required_fields):
        return jsonify({"error": f"A observação deve conter os campos: {required_fields}"}), 400

    # Converte a observação em um DataFrame
    try:
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Dados de observação inválidos: {str(e)}"}), 400

    # Faz a previsão usando o modelo
    proba = pipeline.predict_proba(obs)[0, 1]

    # Tenta salvar no banco de dados
    try:
        p = Prediction(
            observation_id=_id,
            observation=json.dumps(observation),  # Salva a observação como JSON
            proba=proba
        )
        p.save()
    except IntegrityError:
        # Se o ID já existir, retorna um erro e a probabilidade correspondente
        existing_prediction = Prediction.get(Prediction.observation_id == _id)
        return jsonify({
            "error": f"Observation ID {_id} já existe",
            "proba": existing_prediction.proba
        }), 400

    # Retorna a probabilidade prevista
    return jsonify({"id": _id, "proba": proba})

@app.route('/update', methods=['POST'])
def update():
    # Atualiza a classe verdadeira (true_class) de uma observação existente
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        return jsonify({"error": f"Observation ID {obs['id']} não existe"}), 404

@app.route('/list-db-contents')
def list_db_contents():
    # Lista todo o conteúdo do banco de dados
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

########################################
# Execução do Aplicativo

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)