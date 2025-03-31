import pickle

file_path = "artifacts/transform_config.pkl"

try:
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    print("✅ Pickle file loaded successfully!")
    print("Contents:", data)
except Exception as e:
    print(f"❌ Error loading pickle file: {e}")