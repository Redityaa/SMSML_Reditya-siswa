import argparse
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    """Memuat data dari path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File dataset tidak ditemukan di: {path}")
    
    print(f"📥 Memuat dataset dari: {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    """Melakukan cleaning, encoding, dan scaling."""
    print("⚙️ Memulai proses preprocessing...")
    
    # Hapus kolom index bawaan jika ada
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Standardisasi Nama Target
    # Mengubah 'Credit Risk' menjadi 'Risk'
    target_map = {'Credit Risk': 'Risk', 'class': 'Risk', 'Target': 'Risk'}
    df.rename(columns=target_map, inplace=True)
    
    if 'Risk' in df.columns:
        print("✅ Kolom target terdeteksi dan dinormalisasi menjadi 'Risk'")
    
    # 3. Handling Missing Values
    # Isi NaN di akun bank dengan 'unknown'
    cols_missing = ['Saving accounts', 'Checking account']
    for col in cols_missing:
        if col in df.columns:
            df[col].fillna('unknown', inplace=True)
            
    # 4. Encoding (Object -> Angka)
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    
    for col in object_cols:
        df[col] = le.fit_transform(df[col])
        
    print(f"✅ Encoding selesai untuk {len(object_cols)} kolom kategorikal.")

    # 5. Scaling (Standarisasi Angka)
    scaler = StandardScaler()
    
    # Ambil semua kolom numerik
    num_cols = df.select_dtypes(include=['number']).columns
    
    # PENTING: Jangan scale kolom Target ('Risk')
    if 'Risk' in num_cols:
        num_cols = num_cols.drop('Risk')
        
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"✅ Scaling selesai untuk {len(num_cols)} kolom numerik.")
    
    return df

def main(input_path, output_path):
    # 1. Load
    df = load_data(input_path)
    
    # 2. Process
    df_clean = preprocess_data(df)
    
    # 3. Save
    # Pastikan folder output tersedia
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_clean.to_csv(output_path, index=False)
    print(f"🎉 SUKSES! Data bersih disimpan di: {output_path}")

if __name__ == "__main__":
    # Menggunakan argparse agar script bisa menerima input dari Terminal/GitHub Actions
    parser = argparse.ArgumentParser(description="Script Otomatisasi Preprocessing")
    
    # Argumen untuk lokasi input (dataset mentah)
    parser.add_argument("--input", type=str, required=True, help="Path ke dataset raw (.csv)")
    
    # Argumen untuk lokasi output (dataset bersih)
    parser.add_argument("--output", type=str, required=True, help="Path tujuan simpan data clean (.csv)")
    
    args = parser.parse_args()
    
    # Jalankan fungsi utama
    main(args.input, args.output)