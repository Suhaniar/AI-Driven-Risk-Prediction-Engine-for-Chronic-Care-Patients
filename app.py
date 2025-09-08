
from flask import Flask, request, jsonify
from flask_cors import CORS
import db
import pandas as pd
import io
import sqlite3
import json
import numpy as np

app = Flask(__name__)
CORS(app)  # allow frontend to call


# --- Initialize database (run once at startup)
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        file_name TEXT,
        prediction TEXT,
        probabilities TEXT,
        vitals TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


init_db()


def make_json_serializable_probs(probs):
    """Convert model probabilities to JSON-serializable dict"""
    if isinstance(probs, dict):
        return {str(k): float(v) for k, v in probs.items()}
    elif isinstance(probs, (list, np.ndarray)):
        return {str(i): float(v) for i, v in enumerate(probs)}
    elif isinstance(probs, (np.floating, float, int)):
        return {"value": float(probs)}
    else:
        return {"raw": str(probs)}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient_id = request.form.get("patient_id", "001")
        file = request.files["file"]

        # read file in memory
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))

        print(f"📂 Received file {file.filename} for patient {patient_id}")
        print(f"🔎 CSV inspected: {df.shape}")

        # Insert into DB + predict (your model)
        db.manual_insert_with_mapping(patient_id, df)
        result = db.predict_patient(patient_id)

        # extract last row vitals (latest measurement)
        vitals = {}
        vital_cols = ["heart_rate", "blood_pressure", "temperature",
                      "respiratory_rate", "spo2"]

        for col in vital_cols:
            if col in df.columns:
                vitals[col] = str(df[col].iloc[-1])  # ensure JSON serializable

        print(f"🩺 Vitals extracted: {vitals}")

        if result:
            pred_label, probs = result

            clean_probs = make_json_serializable_probs(probs)

            # --- Store in SQLite history
            try:
                conn = sqlite3.connect("history.db")
                c = conn.cursor()
                c.execute("""
                    INSERT INTO history (patient_id, file_name, prediction, probabilities, vitals)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    patient_id,
                    file.filename,
                    pred_label,
                    json.dumps(clean_probs),
                    json.dumps(vitals)
                ))
                conn.commit()
                conn.close()
                print("✅ Inserted into history.db successfully")
            except Exception as db_err:
                print("❌ DB insert failed:", db_err)
                return jsonify({"error": f"DB insert failed: {db_err}"}), 500

            return jsonify({
                "prediction": pred_label,
                "probabilities": clean_probs,
                "vitals": vitals
            })

        else:
            return jsonify({"error": "Prediction failed"}), 400

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/history", defaults={'patient_id': None}, methods=["GET"])
@app.route("/history/<patient_id>", methods=["GET"])
def history(patient_id):
    try:
        conn = sqlite3.connect("history.db")
        c = conn.cursor()

        if patient_id:
            c.execute("""
                SELECT id, patient_id, file_name, prediction, probabilities, vitals, created_at
                FROM history
                WHERE patient_id = ?
                ORDER BY created_at DESC
            """, (patient_id,))
        else:
            c.execute("""
                SELECT id, patient_id, file_name, prediction, probabilities, vitals, created_at
                FROM history
                ORDER BY created_at DESC
            """)

        rows = c.fetchall()
        conn.close()

        history_data = []
        for row in rows:
            try:
                probs = json.loads(row[4]) if row[4] else {}
            except Exception:
                probs = {}
            try:
                vitals = json.loads(row[5]) if row[5] else {}
            except Exception:
                vitals = {}

            history_data.append({
                "id": row[0],
                "patient_id": row[1],
                "file_name": row[2],
                "prediction": row[3],
                "probabilities": probs,
                "vitals": vitals,
                "created_at": row[6]
            })

        return jsonify(history_data)

    except Exception as e:
        print(f"❌ Error in /history: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
=======
from flask import Flask, request, jsonify
from flask_cors import CORS
import db
import pandas as pd
import io
import sqlite3
import json
import numpy as np

app = Flask(__name__)
CORS(app)  # allow frontend to call


# --- Initialize database (run once at startup)
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        file_name TEXT,
        prediction TEXT,
        probabilities TEXT,
        vitals TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


init_db()


def make_json_serializable_probs(probs):
    """Convert model probabilities to JSON-serializable dict"""
    if isinstance(probs, dict):
        return {str(k): float(v) for k, v in probs.items()}
    elif isinstance(probs, (list, np.ndarray)):
        return {str(i): float(v) for i, v in enumerate(probs)}
    elif isinstance(probs, (np.floating, float, int)):
        return {"value": float(probs)}
    else:
        return {"raw": str(probs)}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient_id = request.form.get("patient_id", "001")
        file = request.files["file"]

        # read file in memory
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))

        print(f"📂 Received file {file.filename} for patient {patient_id}")
        print(f"🔎 CSV inspected: {df.shape}")

        # Insert into DB + predict (your model)
        db.manual_insert_with_mapping(patient_id, df)
        result = db.predict_patient(patient_id)

        # extract last row vitals (latest measurement)
        vitals = {}
        vital_cols = ["heart_rate", "blood_pressure", "temperature",
                      "respiratory_rate", "spo2"]

        for col in vital_cols:
            if col in df.columns:
                vitals[col] = str(df[col].iloc[-1])  # ensure JSON serializable

        print(f"🩺 Vitals extracted: {vitals}")

        if result:
            pred_label, probs = result

            clean_probs = make_json_serializable_probs(probs)

            # --- Store in SQLite history
            try:
                conn = sqlite3.connect("history.db")
                c = conn.cursor()
                c.execute("""
                    INSERT INTO history (patient_id, file_name, prediction, probabilities, vitals)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    patient_id,
                    file.filename,
                    pred_label,
                    json.dumps(clean_probs),
                    json.dumps(vitals)
                ))
                conn.commit()
                conn.close()
                print("✅ Inserted into history.db successfully")
            except Exception as db_err:
                print("❌ DB insert failed:", db_err)
                return jsonify({"error": f"DB insert failed: {db_err}"}), 500

            return jsonify({
                "prediction": pred_label,
                "probabilities": clean_probs,
                "vitals": vitals
            })

        else:
            return jsonify({"error": "Prediction failed"}), 400

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/history", defaults={'patient_id': None}, methods=["GET"])
@app.route("/history/<patient_id>", methods=["GET"])
def history(patient_id):
    try:
        conn = sqlite3.connect("history.db")
        c = conn.cursor()

        if patient_id:
            c.execute("""
                SELECT id, patient_id, file_name, prediction, probabilities, vitals, created_at
                FROM history
                WHERE patient_id = ?
                ORDER BY created_at DESC
            """, (patient_id,))
        else:
            c.execute("""
                SELECT id, patient_id, file_name, prediction, probabilities, vitals, created_at
                FROM history
                ORDER BY created_at DESC
            """)

        rows = c.fetchall()
        conn.close()

        history_data = []
        for row in rows:
            try:
                probs = json.loads(row[4]) if row[4] else {}
            except Exception:
                probs = {}
            try:
                vitals = json.loads(row[5]) if row[5] else {}
            except Exception:
                vitals = {}

            history_data.append({
                "id": row[0],
                "patient_id": row[1],
                "file_name": row[2],
                "prediction": row[3],
                "probabilities": probs,
                "vitals": vitals,
                "created_at": row[6]
            })

        return jsonify(history_data)

    except Exception as e:
        print(f"❌ Error in /history: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

