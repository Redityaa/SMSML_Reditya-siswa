def load_data():
    filename = "german_credit_preprocessing.csv"
    
    if os.path.exists(filename):
        print(f"✅ Data ditemukan: {filename}")
        return pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"❌ File '{filename}' tidak ada di folder ini.")
