import pandas as pd
import os

def analyze():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw_data.csv")
    
    print(f"Loading {raw_path}...")
    # Read with index_col=0
    df = pd.read_csv(raw_path, encoding='gbk', index_col=0)
    
    # Clean columns
    df.columns = df.columns.str.strip().str.replace('"', '')
    for i, col in enumerate(df.columns):
        print(f"Col {i}: '{col}'")
        
    print("First 5 rows:")
    print(df.head())
    
    print(f"Total rows: {len(df)}")
    print(f"Unique Students: {df['JMXH'].nunique()}")
    
    # Analyze XNMC
    print("\nUnique Academic Years (XNMC):")
    print(df['XNMC'].value_counts())
    
    # Analyze Enrollment Years
    # Handle non-numeric JMXH
    df['enroll_year'] = pd.to_numeric(df['JMXH'].astype(str).str[:4], errors='coerce')
    print("\nEnrollment Years:")
    print(df['enroll_year'].value_counts())
    
    # Check for longitudinal data
    # For each student, how many unique XNMC do they have?
    student_years = df.groupby('JMXH')['XNMC'].nunique()
    print("\nYears per Student distribution:")
    print(student_years.value_counts().sort_index())
    
    # Check for specific student history
    # Pick a student with max years
    if not student_years.empty:
        top_student = student_years.idxmax()
        print(f"\nExample Student ({top_student}) History:")
        print(df[df['JMXH'] == top_student][['XNMC', 'XQMC', 'KCMC', 'CJ']].sort_values(['XNMC', 'XQMC']))

if __name__ == "__main__":
    analyze()
