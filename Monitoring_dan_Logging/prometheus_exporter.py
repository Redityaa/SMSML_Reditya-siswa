import time
import pandas as pd
import mlflow.sklearn
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

# 1. SETUP METRIKS
# Metrik Dasar
REQUEST_COUNT = Counter('app_requests_total', 'Total request yang masuk')
ERROR_COUNT = Counter('app_errors_total', 'Total error yang terjadi')
LATENCY = Histogram('app_latency_seconds', 'Waktu proses prediksi')
IN_PROGRESS = Gauge('app_requests_in_progress', 'Request yang sedang berjalan')

# Metrik Bisnis/Data
PREDICTION_RISK_GOOD = Counter('pred_risk_good_total', 'Total prediksi Good (0)')
PREDICTION_RISK_BAD = Counter('pred_risk_bad_total', 'Total prediksi Bad (1)')
INPUT_CREDIT_AMOUNT = Histogram('input_credit_amount', 'Distribusi nilai Credit Amount')
INPUT_AGE = Gauge('input_age_average', 'Rata-rata umur peminjam (dari batch terakhir)')
INPUT_DURATION = Histogram('input_duration_month', 'Distribusi durasi peminjaman')

# Metrik Sistem (Simulasi)
MODEL_CONFIDENCE = Gauge('model_last_confidence', 'Confidence score prediksi terakhir')

# LOAD MODEL
MODEL_URI = "runs:/b871f5c1723041d188b91e8311a5809d/model" 

print("⏳ Sedang memuat model...")
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("✅ Model berhasil dimuat!")

except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    print("Pastikan Run ID benar atau gunakan path absolut ke folder artifacts/model")
    exit()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    IN_PROGRESS.inc()
    REQUEST_COUNT.inc()
    
    try:
        content = request.json
        df = pd.DataFrame(content)
        
        # Simpan metrik input data
        if 'Credit amount' in df.columns:
            amount = df['Credit amount'].iloc[0]
            INPUT_CREDIT_AMOUNT.observe(amount)
            
        if 'Age' in df.columns:
            age = df['Age'].iloc[0]
            INPUT_AGE.set(age) 
            
        if 'Duration' in df.columns:
            duration = df['Duration'].iloc[0]
            INPUT_DURATION.observe(duration)

        # Prediksi
        prediction = model.predict(df)
        result = int(prediction[0])
        
        # Simpan metrik hasil prediksi
        if result == 1: 
            PREDICTION_RISK_GOOD.inc()
        else:
            PREDICTION_RISK_BAD.inc()
            
        # Simulasi confidence 
        MODEL_CONFIDENCE.set(0.95) 

        process_time = time.time() - start_time
        LATENCY.observe(process_time)
        
        return jsonify({'risk_prediction': result, 'status': 'success'})

    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({'error': str(e), 'status': 'failed'}), 500
    finally:
        IN_PROGRESS.dec()

if __name__ == '__main__':
    # Jalankan server metrics Prometheus di port 8000
    start_http_server(8000)
    print("🚀 Prometheus Metrics berjalan di port 8000")
    
    # Jalankan aplikasi Flask di port 5000
    app.run(host='0.0.0.0', port=5000)