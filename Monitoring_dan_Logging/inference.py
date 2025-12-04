import requests
import time
import random
import json

url = "http://localhost:5000/predict"

def get_dummy_data():
    # Pilih acak: Kirim data orang Good atau Bad
    tipe = random.choice(['good', 'bad'])
    
    if tipe == 'good':
        return [{
            'Age': 2.76945450304928, 
            'Sex': 0.6739942823814804, 
            'Job': 0.1388281870166868, 
            'Housing': -0.1261711556300402, 
            'Saving accounts': 1.8279794691119589, 
            'Checking account': -1.256312506254786, 
            'Credit amount': -0.7399178994342078, 
            'Duration': -1.227634292640598, 
            'Purpose': 1.0815194851776415
        }]
    else:
        return [{
            'Age': -1.1870407320890095, 
            'Sex': -1.4836921115511783, 
            'Job': 0.1388281870166868, 
            'Housing': -0.1261711556300402, 
            'Saving accounts': -0.7003168897612686, 
            'Checking account': -0.4615365573376336, 
            'Credit amount': 0.9369064245483124, 
            'Duration': 2.260689289908404, 
            'Purpose': 1.0815194851776415
        }]

print("🚀 Memulai simulasi traffic DATA ASLI...")
print("Tekan Ctrl+C untuk berhenti.")

counter_good = 0
counter_bad = 0

while True:
    try:
        data = get_dummy_data()
        
        # Kirim request
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json().get('risk_prediction')
            
            # Cek hasil prediksi
            if result == 1:
                counter_good += 1
                print(f"Prediksi: 🟢 GOOD (1) | Total Good: {counter_good} | Total Bad: {counter_bad}")
            elif result == 2:
                counter_bad += 1
                print(f"Prediksi: 🔴 BAD  (2) | Total Good: {counter_good} | Total Bad: {counter_bad}")
            else:
                print(f"Prediksi: ❓ UNKNOWN ({result})")

        else:
            print(f"⚠️ Gagal: {response.text}")
            
    except Exception as e:
        print(f"❌ Error koneksi: {e}")
    
    time.sleep(1) # Delay 1 detik