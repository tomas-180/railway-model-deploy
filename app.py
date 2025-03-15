from flask import Flask, request, jsonify
import json
import pandas as pd
import joblib
from peewee import *

app = Flask(__name__)

# Conectar ao banco de dados SQLite com Peewee
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


# Definir o modelo da tabela usando Peewee
class Prediction(Model):
    observation_id = CharField(primary_key=True)  # ID da observação
    observation = TextField()  # JSON da observação
    proba = FloatField()  # Probabilidade prevista
    true_class = CharField(null=True)  # Classe real (pode ser NULL)

    class Meta:
        database = DB


# Criar tabela se não existir
DB.connect()
DB.create_tables([Prediction], safe=True)

# Carregar modelo treinado e colunas esperadas
with open('columns.json', 'r') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if "id" not in data or "observation" not in data:
            return jsonify({"error": "Faltam campos obrigatorios (id, observation)"}), 400

        obs_id = str(data["id"])  # Garantimos que seja string
        observation = data["observation"]

        # Verificar se ID já existe
        if Prediction.get_or_none(Prediction.observation_id == obs_id):
            return jsonify({"error": "ID ja existe", "id": obs_id}), 400

        # Converter observação para DataFrame
        obs_df = pd.DataFrame([observation], columns=columns)

        # Fazer previsão
        proba = pipeline.predict_proba(obs_df)[:, 1][0]

        # Inserir no banco de dados
        Prediction.create(observation_id=obs_id, observation=json.dumps(observation), proba=proba, true_class=None)

        return jsonify({"id": obs_id, "probability": proba})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/update', methods=['POST'])
def update():
    try:
        data = request.get_json()

        if "id" not in data or "true_class" not in data:
            return jsonify({"error": "Faltam campos obrigatorios (id, true_class)"}), 400

        obs_id = str(data["id"])
        true_class = data["true_class"]

        # Verificar se a observação existe
        prediction = Prediction.get_or_none(Prediction.observation_id == obs_id)
        if not prediction:
            return jsonify({"error": "ID nao encontrado", "id": obs_id}), 404

        # Atualizar a classe real
        prediction.true_class = true_class
        prediction.save()

        return jsonify({
            "id": obs_id,
            "observation": json.loads(prediction.observation),
            "probability": prediction.proba,
            "true_class": true_class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)





