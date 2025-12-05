import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import mlflow.sklearn
import dagshub
import pickle 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# KONFIGURASI DAGSHUB
DAGSHUB_URI = "https://dagshub.com/Redityaa/Submission_Sistem-Machine-Learning.mlflow"

def load_data():
    """Memuat data dari file CSV yang sudah diproses."""
    filename = "german_credit_preprocessing.csv"
    
    if os.path.exists(filename):
        print(f"✅ Data ditemukan: {filename}")
        return pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"❌ File '{filename}' tidak ada di folder ini.")

def main():
    # Load Data
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(e)
        return
    
    target_col = None
    if 'Risk' in df.columns:
        target_col = 'Risk'
    elif 'Credit Risk' in df.columns:
        target_col = 'Credit Risk'
        
    if target_col is None:
        print("❌ Kolom target ('Risk' atau 'Credit Risk') tidak ditemukan.")
        return
        
    X = df.drop(target_col, axis=1)
    y = df[target_col].astype(int) # Pastikan target adalah integer
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Setup DagsHub & MLflow
    try:
        dagshub.init(repo_owner='Redityaa', repo_name='Submission_Sistem-Machine-Learning', mlflow=True)
    except Exception as e:
        print(f"⚠️ Gagal inisialisasi DagsHub (Mungkin sudah diinisialisasi atau masalah koneksi): {e}")

    mlflow.set_experiment("Eksperimen_Tuning_Credit")
    print(f"📡 Tracking URI: {mlflow.get_tracking_uri()}")

    with mlflow.start_run(run_name="RandomForest_Hyperopt"):
        print("🔍 Memulai Hyperparameter Tuning...")
        
        # Define Hyperparameters
        rf = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        # RandomizedSearchCV (Lebih cepat dari GridSearch)
        search = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, scoring='accuracy', random_state=42)
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        print(f"🏆 Best Params: {search.best_params_}")
        
        # LOGGING
        mlflow.log_params(search.best_params_)

        # Evaluasi
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Akurasi Model Terbaik: {acc:.4f}")

        # LOGGING
        mlflow.log_metric("accuracy", acc)


        model_filename = "model.pkl"
        
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
            
        mlflow.log_artifact(model_filename, artifact_path="random_forest_model")
        
        requirements_content = f"""\
scikit-learn
pandas
numpy
matplotlib
seaborn
mlflow
dagshub
"""
        requirements_path = "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write(requirements_content)
        mlflow.log_artifact(requirements_path, artifact_path="random_forest_model")
        


        # 2. Gambar Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Simpan sementara lalu log ke MLflow
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        # Text Classification Report
        report = classification_report(y_test, y_pred)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Bersihkan file sampah lokal
        os.remove(model_filename)
        os.remove(requirements_path)
        os.remove(cm_path)
        os.remove(report_path)
        
        print("\n🚀 Selesai! Cek DagsHub Anda di bawah direktori 'random_forest_model' untuk melihat hasilnya. Model tersimpan sebagai model.pkl.")

if __name__ == "__main__":
    main()