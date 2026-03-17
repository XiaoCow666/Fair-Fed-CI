import pandas as pd
import os

def inspect():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw_data.csv")
    
    print(f"Loading {raw_path}...")
    # Use index_col=0 as confirmed previously
    df = pd.read_csv(raw_path, encoding='gbk', index_col=0)
    
    # Clean columns
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    print(f"Columns: {df.columns.tolist()}")
    
    if 'XNMC' in df.columns:
        print("\n--- XNMC (Academic Year?) ---")
        print(df['XNMC'].unique())
        print(df['XNMC'].value_counts().head())
    else:
        print("\nERROR: XNMC column not found!")

    if 'XQMC' in df.columns:
        print("\n--- XQMC (Semester?) ---")
        print(df['XQMC'].unique())
        print(df['XQMC'].value_counts().head())
    else:
        print("\nERROR: XQMC column not found!")
        
    # Print a few sample rows with these columns
    print("\n--- Sample Rows ---")
    print(df[['JMXH', 'XNMC', 'XQMC']].head(10))

if __name__ == "__main__":
    inspect()
