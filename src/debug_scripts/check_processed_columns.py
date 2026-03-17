import pandas as pd
import os

def check():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed_data_v2.csv")
    
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Columns: {df.columns.tolist()}")
    
    if 'target_score' in df.columns:
        print("target_score found!")
        print(df['target_score'].head())
    else:
        print("target_score NOT found!")

if __name__ == "__main__":
    check()
