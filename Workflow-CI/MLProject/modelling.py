import pandas as pd
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    filename = "german_credit_preprocessing.csv"
    if os.path.exists(filename):
        print(f"✅ Data ditemukan: {filename}")
        return pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"❌ File '{filename}' tidak ada di folder ini.")

def train_model(X_train, y_train, X_test, y_test):
    print("🚀 Memulai training...")
    
    # Init Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✅ Training Selesai! Accuracy: {acc:.4f}")

    run_id = os.environ.get("MLFLOW_RUN_ID")
    
    if run_id is None:
        active_run = mlflow.active_run()
        if active_run:
            run_id = active_run.info.run_id
    
    # Validasi terakhir
    if run_id:
        print(f"💾 Menyimpan Run ID: {run_id}")
        with open("run_id.txt", "w") as f:
            f.write(run_id)
    else:
        print("⚠️ Peringatan: Tidak dapat menemukan Run ID untuk disimpan.")
        
    return model

def main():
    df = load_data()

    # Split Data
    target_col = 'Risk'
    feature_cols = [
        "Age", "Sex", "Job", "Housing", "Saving accounts", 
        "Checking account", "Credit amount", "Duration", "Purpose"
    ]

    print(f"🎯 Menggunakan kolom target: {target_col}")
    print(f"📋 Menggunakan fitur: {feature_cols}")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aktifkan Autolog
    mlflow.sklearn.autolog()

    if os.environ.get('MLFLOW_RUN_ID'):
        print("🤖 Terdeteksi berjalan via MLflow Project (CI/CD)")
        train_model(X_train, y_train, X_test, y_test)
        
    else:
        print("💻 Terdeteksi berjalan manual (Local)")
        mlflow.set_experiment("Eksperimen_German_Credit")
        with mlflow.start_run(run_name="Manual_Run"):
            train_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()