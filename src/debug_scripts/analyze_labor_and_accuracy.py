import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.metrics import mean_squared_error, accuracy_score

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_core import FairAttentionMLP

def analyze():
    print("--- Analyzing 'Public Service Labor' & Model Accuracy ---")
    
    # 1. Load Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed_data_v2.csv")
    df = pd.read_csv(data_path)
    
    # 2. Analyze 'Public Service Labor' (CJ_公益劳动)
    labor_col = 'CJ_公益劳动'
    target_col = 'avg_score'
    
    if labor_col in df.columns:
        print(f"\n[Feature Analysis: {labor_col}]")
        correlation = df[labor_col].corr(df[target_col])
        print(f"Correlation with Avg Score: {correlation:.4f}")
        
        # Check distribution
        print(df[labor_col].describe())
        
        # Check if it distinguishes pass/fail
        pass_threshold = 0.6
        pass_students = df[df[target_col] >= pass_threshold]
        fail_students = df[df[target_col] < pass_threshold]
        
        print(f"Mean {labor_col} for Passing Students: {pass_students[labor_col].mean():.4f}")
        print(f"Mean {labor_col} for Failing Students: {fail_students[labor_col].mean():.4f}")
    else:
        print(f"Warning: {labor_col} not found in data.")

    # 3. Evaluate Model Accuracy
    print("\n[Model Evaluation]")
    model_path = os.path.join(base_dir, "client_0_model.pth")
    
    # Identify features
    exclude_cols = ['JMXH', 'avg_score', 'sensitive_attribute']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Load Model
    model = FairAttentionMLP(input_dim=len(feature_cols))
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare Data
    X = torch.tensor(df[feature_cols].values.astype(float), dtype=torch.float32)
    y_true = df[target_col].values
    
    # Predict
    with torch.no_grad():
        y_pred_score, _ = model(X)
        y_pred_score = y_pred_score.numpy().flatten()
        
    # Calculate Metrics
    mse = mean_squared_error(y_true, y_pred_score)
    print(f"Overall MSE: {mse:.6f}")
    
    # Classification Accuracy (Pass/Fail)
    pass_threshold = 0.6
    y_pred_class = (y_pred_score >= pass_threshold).astype(int)
    y_true_class = (y_true >= pass_threshold).astype(int)
    
    acc = accuracy_score(y_true_class, y_pred_class)
    print(f"Pass/Fail Classification Accuracy: {acc:.4f} (Threshold={pass_threshold})")
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_class).ravel()
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == "__main__":
    analyze()
