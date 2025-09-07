import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import re

# -----------------------------
# Load pretrained model safely
# -----------------------------
try:
    loaded = joblib.load("chronic_deterioration_model.pkl")
    print("✅ Model loaded successfully")
    
    if isinstance(loaded, dict) and 'model' in loaded:
        best_model = loaded['model']
        le = loaded.get('label_encoder', None)
        model_cols = loaded.get('features', [])
        print("✅ Loaded dict with model, label encoder, and features")
    else:
        best_model = loaded
        le = None
        model_cols = []
        print("✅ Loaded direct model/pipeline")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# -----------------------------
# Step 1: Setup Database
# -----------------------------
def init_db():
    try:
        conn = sqlite3.connect("patients.db")
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_features (
            patient_id TEXT,
            prediction_timestamp TEXT,
            Heart_Rate_mean REAL,
            Heart_Rate_min REAL,
            Heart_Rate_max REAL,
            Heart_Rate_slope REAL,
            Respiratory_Rate_mean REAL,
            Respiratory_Rate_min REAL,
            Respiratory_Rate_max REAL,
            Respiratory_Rate_slope REAL,
            SpO2_mean REAL,
            SpO2_min REAL,
            SpO2_max REAL,
            SpO2_slope REAL,
            Temperature_mean REAL,
            Temperature_min REAL,
            Temperature_max REAL,
            Temperature_slope REAL,
            Non_Invasive_BP_systolic_mean REAL,
            Non_Invasive_BP_systolic_min REAL,
            Non_Invasive_BP_systolic_max REAL,
            Non_Invasive_BP_systolic_slope REAL,
            Non_Invasive_BP_diastolic_mean REAL,
            Non_Invasive_BP_diastolic_min REAL,
            Non_Invasive_BP_diastolic_max REAL,
            Non_Invasive_BP_diastolic_slope REAL,
            GLUCOSE_mean REAL,
            GLUCOSE_min REAL,
            GLUCOSE_max REAL,
            GLUCOSE_slope REAL,
            SODIUM_mean REAL,
            SODIUM_min REAL,
            SODIUM_max REAL,
            SODIUM_slope REAL,
            POTASSIUM_mean REAL,
            POTASSIUM_min REAL,
            POTASSIUM_max REAL,
            POTASSIUM_slope REAL,
            CREATININE_mean REAL,
            CREATININE_min REAL,
            CREATININE_max REAL,
            CREATININE_slope REAL,
            HEMOGLOBIN_mean REAL,
            HEMOGLOBIN_min REAL,
            HEMOGLOBIN_max REAL,
            HEMOGLOBIN_slope REAL,
            prediction TEXT,
            PRIMARY KEY (patient_id, prediction_timestamp)
        )
        """)
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")

# -----------------------------
# Step 2: Check what's in the CSV file
# -----------------------------
def inspect_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"✅ CSV loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

# -----------------------------
# Step 3: Manual insert with column mapping
# -----------------------------
def manual_insert_with_mapping(patient_id, df):
    try:
        # Map CSV columns to database columns (replace spaces with underscores)
        column_mapping = {}
        for csv_col in df.columns:
            db_col = csv_col.replace(' ', '_')
            column_mapping[csv_col] = db_col
        
        print("Column mapping:")
        for csv_col, db_col in column_mapping.items():
            print(f"  {csv_col} -> {db_col}")
        
        # Get the first row of data
        row_data = df.iloc[0].to_dict()
        
        # Prepare SQL insert
        conn = sqlite3.connect("patients.db")
        cursor = conn.cursor()
        
        # Build column names and values
        columns = ['patient_id', 'prediction_timestamp']
        values = [patient_id, datetime.now().isoformat()]
        
        for csv_col, db_col in column_mapping.items():
            columns.append(db_col)
            values.append(row_data[csv_col])
        
        columns.append('prediction')
        values.append(None)
        
        # Create SQL query
        placeholders = ', '.join(['?'] * len(columns))
        column_names = ', '.join(columns)
        
        sql = f"INSERT INTO patient_features ({column_names}) VALUES ({placeholders})"
        
        # Execute insert
        cursor.execute(sql, values)
        conn.commit()
        conn.close()
        
        print(f"✅ Successfully inserted data for patient {patient_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error in manual insert: {e}")
        return False

# -----------------------------
# Step 4: Convert database columns to exact model format
# -----------------------------
def convert_to_model_format(df):
    """Convert database column names to exact format expected by the model"""
    # The model expects the original CSV format with specific underscore/space pattern
    column_mapping = {
        'Heart_Rate_mean': 'Heart Rate_mean',
        'Heart_Rate_min': 'Heart Rate_min', 
        'Heart_Rate_max': 'Heart Rate_max',
        'Heart_Rate_slope': 'Heart Rate_slope',
        'Respiratory_Rate_mean': 'Respiratory Rate_mean',
        'Respiratory_Rate_min': 'Respiratory Rate_min',
        'Respiratory_Rate_max': 'Respiratory Rate_max',
        'Respiratory_Rate_slope': 'Respiratory Rate_slope',
        'SpO2_mean': 'SpO2_mean',
        'SpO2_min': 'SpO2_min',
        'SpO2_max': 'SpO2_max',
        'SpO2_slope': 'SpO2_slope',
        'Temperature_mean': 'Temperature_mean',
        'Temperature_min': 'Temperature_min',
        'Temperature_max': 'Temperature_max',
        'Temperature_slope': 'Temperature_slope',
        'Non_Invasive_BP_systolic_mean': 'Non Invasive BP systolic_mean',
        'Non_Invasive_BP_systolic_min': 'Non Invasive BP systolic_min',
        'Non_Invasive_BP_systolic_max': 'Non Invasive BP systolic_max',
        'Non_Invasive_BP_systolic_slope': 'Non Invasive BP systolic_slope',
        'Non_Invasive_BP_diastolic_mean': 'Non Invasive BP diastolic_mean',
        'Non_Invasive_BP_diastolic_min': 'Non Invasive BP diastolic_min',
        'Non_Invasive_BP_diastolic_max': 'Non Invasive BP diastolic_max',
        'Non_Invasive_BP_diastolic_slope': 'Non Invasive BP diastolic_slope',
        'GLUCOSE_mean': 'GLUCOSE_mean',
        'GLUCOSE_min': 'GLUCOSE_min',
        'GLUCOSE_max': 'GLUCOSE_max',
        'GLUCOSE_slope': 'GLUCOSE_slope',
        'SODIUM_mean': 'SODIUM_mean',
        'SODIUM_min': 'SODIUM_min',
        'SODIUM_max': 'SODIUM_max',
        'SODIUM_slope': 'SODIUM_slope',
        'POTASSIUM_mean': 'POTASSIUM_mean',
        'POTASSIUM_min': 'POTASSIUM_min',
        'POTASSIUM_max': 'POTASSIUM_max',
        'POTASSIUM_slope': 'POTASSIUM_slope',
        'CREATININE_mean': 'CREATININE_mean',
        'CREATININE_min': 'CREATININE_min',
        'CREATININE_max': 'CREATININE_max',
        'CREATININE_slope': 'CREATININE_slope',
        'HEMOGLOBIN_mean': 'HEMOGLOBIN_mean',
        'HEMOGLOBIN_min': 'HEMOGLOBIN_min',
        'HEMOGLOBIN_max': 'HEMOGLOBIN_max',
        'HEMOGLOBIN_slope': 'HEMOGLOBIN_slope'
    }
    
    # Rename columns
    df_renamed = df.rename(columns=column_mapping)
    return df_renamed

# -----------------------------
# Step 5: Predict next 90 days
# -----------------------------
def predict_patient(patient_id):
    try:
        conn = sqlite3.connect("patients.db")
        df = pd.read_sql(f"SELECT * FROM patient_features WHERE patient_id='{patient_id}' ORDER BY prediction_timestamp DESC LIMIT 1", conn)
        conn.close()

        if df.empty:
            print(f"❌ No records found for patient {patient_id}")
            return None

        print(f"✅ Found records for patient {patient_id}")
        
        # Remove non-feature columns
        non_feature_cols = ['patient_id', 'prediction_timestamp', 'prediction']
        feature_columns = [col for col in df.columns if col not in non_feature_cols]
        feat_df = df[feature_columns]

        print(f"Database feature columns: {feat_df.columns.tolist()[:5]}...")  # Show first 5 only
        
        # Convert column names to exact model format
        feat_df_model = convert_to_model_format(feat_df)
        print(f"Model feature columns: {feat_df_model.columns.tolist()[:5]}...")  # Show first 5 only
        
        # Predict
        probs = best_model.predict_proba(feat_df_model)[0]
        pred = best_model.predict(feat_df_model)[0]

        # Handle label decoding
        if le is not None:
            pred_label = le.inverse_transform([pred])[0]
            classes = le.classes_
        else:
            pred_label = pred
            classes = best_model.classes_

        print(f"\n🎯 PREDICTION for {patient_id} (next 90 days): {pred_label}")
        print("Probability Distribution:")
        for i, cls in enumerate(classes):
            print(f"  {cls}: {probs[i]:.4f}")

        # Save prediction
        conn = sqlite3.connect("patients.db")
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE patient_features
        SET prediction=?
        WHERE patient_id=? AND prediction_timestamp=?
        """, (str(pred_label), patient_id, df['prediction_timestamp'].iloc[0]))
        conn.commit()
        conn.close()

        # Bar chart
        plt.figure(figsize=(10, 6))
        colors = ['green', 'yellow', 'orange', 'red']
        bars = plt.bar(classes, probs, color=colors)
        plt.title(f"Prediction for {patient_id} (next 90 days): {pred_label}", fontsize=14, fontweight='bold')
        plt.ylabel("Probability", fontsize=12)
        plt.xlabel("Deterioration Level", fontsize=12)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.05)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        return pred_label, dict(zip(classes, probs))
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("CHRONIC PATIENT DETERIORATION PREDICTION SYSTEM")
    print("=" * 60)
    
    # Initialize database
    init_db()
    
    # File path and patient ID
    file_path = "C:/Users/suhan/OneDrive/Desktop/PYTHON/AIPROJ/deepseek_csv_20250906_d59182.txt"
    patient_id = "001"
    
    print(f"\n📁 Processing file: {file_path}")
    print(f"👤 Patient ID: {patient_id}")
    
    # Step 1: Inspect the CSV file
    df = inspect_csv(file_path)
    if df is None:
        exit()
    
    # Step 2: Manual insert with proper column mapping
    success = manual_insert_with_mapping(patient_id, df)
    if not success:
        exit()
    
    # Step 3: Make prediction
    print(f"\n🔮 Making prediction for patient {patient_id}...")
    result = predict_patient(patient_id)
    
    if result:
        pred_label, probabilities = result
        print(f"\n✅ Prediction completed successfully!")
        print(f"   Result: {pred_label}")
        print(f"   Probabilities: {probabilities}")
    else:
        print(f"\n❌ Prediction failed")