import pandas as pd
import os

def confirm():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw_data.csv")
    
    print(f"Loading {raw_path}...")
    df = pd.read_csv(raw_path, encoding='gbk', index_col=0)
    
    # Clean columns
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    print("\n--- First 5 rows of XNMC ---")
    print(df['XNMC'].head().tolist())
    
    print("\n--- First 5 rows of XQMC ---")
    print(df['XQMC'].head().tolist())

if __name__ == "__main__":
    confirm()
