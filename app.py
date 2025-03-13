from flask import Flask, request, jsonify
import pickle
import sqlite3
import json
import pandas as pd
import joblib

app = Flask(__name__)

# Carregar modelo treinado e colunas esperadas
with open('columns.json', 'r') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

# Criar a base de dados SQLite se não existir
conn = sqlite3.connect('predictions.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY,
        observation TEXT,
        proba REAL,
        true_class TEXT
    )
''')
conn.commit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber JSON
        data = request.get_json()

        # Validar estrutura do JSON
        if "id" not in data or "observation" not in data:
            return jsonify({"error": "Faltam campos obrigatórios (id, observation)"}), 400

        obs_id = data["id"]
        observation = data["observation"]

        # Verificar se o ID já existe na BD
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (obs_id,))
        if cursor.fetchone():
            return jsonify({"error": "ID ja existe", "id": obs_id}), 400

        # Converter para DataFrame e garantir ordem correta das colunas
        obs_df = pd.DataFrame([observation], columns=columns)

        # Fazer previsão
        proba = pipeline.predict_proba(obs_df)[:, 1][0]

        # Guardar na base de dados
        cursor.execute(
            "INSERT INTO predictions (id, observation, proba, true_class) VALUES (?, ?, ?, ?)",
            (obs_id, json.dumps(observation), proba, None)
        )
        conn.commit()

        return jsonify({"id": obs_id, "probability": proba})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update', methods=['POST'])
def update():
    try:
        # Receber JSON
        data = request.get_json()

        # Validar estrutura do JSON
        if "id" not in data or "true_class" not in data:
            return jsonify({"error": "Faltam campos obrigatórios (id, true_class)"}), 400

        obs_id = data["id"]
        true_class = data["true_class"]

        # Verificar se o ID existe na BD
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (obs_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({"error": "ID não encontrado", "id": obs_id}), 404

        # Atualizar true_class
        cursor.execute("UPDATE predictions SET true_class = ? WHERE id = ?", (true_class, obs_id))
        conn.commit()

        # Retornar observação atualizada
        return jsonify({"id": obs_id, "observation": json.loads(row[1]), "probability": row[2], "true_class": true_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Executar a aplicação
if __name__ == '__main__':
    app.run(debug=True)








