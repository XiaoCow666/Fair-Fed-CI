"""
Fair-Fed-CI: Privacy-Preserving Educational Prediction via Federated Learning
-------------------------------------------------------------------------
Fairness Evaluation Script: Loads the aggregated global model and assesses
prediction discrepancies (RMSE/R2) across different subpopulations (colleges).
"""
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.model_core import EnhancedNet

def load_global_model(input_dim, model_path="global_model_weights.npz"):
    model = EnhancedNet(input_dim)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Global model weights not found at {model_path}")
        
    # Load numpy arrays
    params_np = np.load(model_path)
    arrays = [params_np[key] for key in params_np.files]
    
    # In model_core.py get_shared_parameters(), the OrderedDict is used.
    # However, Python dict iteration order might have shifted if elements were added later.
    # We must explicitly map arrays to the keys that match their shapes.
    shared_params = model.get_shared_parameters()
    state_dict = {}
    
    # Let's map dynamically by matching array total elements to tensor total elements
    # Since numpy arrays lose names during FL aggregation
    used_arrays = []
    
    for name, tensor in shared_params.items():
        tensor_shape = tuple(tensor.shape)
        
        # Find exactly ONE matching array that hasn't been used
        matched_arr = None
        for i, arr in enumerate(arrays):
            if i in used_arrays:
                continue
            if tuple(arr.shape) == tensor_shape:
                matched_arr = arr
                used_arrays.append(i)
                break
                
        if matched_arr is not None:
            state_dict[name] = torch.tensor(matched_arr)
        else:
            print(f"WARNING: Could not find matching array for {name} with shape {tensor_shape}")
            
    if len(used_arrays) != len(arrays):
        print("WARNING: Not all numpy arrays from the federated model were utilized.")

        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def evaluate_per_college():
    print("Starting Subpopulation (Per-College) Evaluation...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_features = pd.read_csv(os.path.join(base_dir, "data", "processed_data_v2.csv"))
    df_raw = pd.read_csv(os.path.join(base_dir, "data", "raw_data.csv"), encoding='gbk')
    
    # Strictly define feature_cols based purely on training config
    target_col = 'target_score'
    exclude_cols = ['JMXH', target_col, 'sensitive_attribute']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    
    # Validate strictly 110
    input_dim = 110
    feature_cols = feature_cols[:110]
    
    model = load_global_model(input_dim, os.path.join(base_dir, "global_model_weights.npz"))
    
    # Merge XY just for grouping, keeping track of JMXH
    xy_mapping = df_raw[['JMXH', 'XY']].drop_duplicates()
    df_merged = df_features.merge(xy_mapping, on='JMXH', how='left')
    
    if 'XY' not in df_merged.columns:
        print("Error: 'XY' column mapping failed.")
        return
        
    results = []
    unique_colleges = df_merged['XY'].dropna().unique()
    print(f"Evaluating {len(unique_colleges)} colleges...")
    
    for college in unique_colleges:
        college_df = df_merged[df_merged['XY'] == college]
        if len(college_df) < 5: # Skip very small groups
            continue
            
        # Extract only the exact feature columns used during training
        # to ensure dimension perfectly matches 110
        X_np = college_df[feature_cols].values.astype(np.float32)
        X = torch.tensor(X_np, dtype=torch.float32)
        y_true = college_df[target_col].values.astype(np.float32)
        
        with torch.no_grad():
            preds, _ = model(X)
            preds = preds.numpy().flatten()
            
        mse = mean_squared_error(y_true, preds)
        rmse = np.sqrt(mse)
        
        # R2 can be unstable for very small sets, handle exceptions
        try:
            r2 = r2_score(y_true, preds)
        except:
            r2 = float('nan')
            
        results.append({
            'College': college,
            'Sample Size': len(college_df),
            'RMSE': rmse,
            'R2': r2
        })
        
    results_df = pd.DataFrame(results)
    # Sort by size to make charts cleaner
    results_df = results_df.sort_values('Sample Size', ascending=False)
    
    # Save CSV
    out_csv = os.path.join(base_dir, "college_evaluation.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"Detailed results saved to {out_csv}")
    
    # Plotting
    plot_college_metrics(results_df, base_dir)

def plot_college_metrics(df, base_dir):
    # Set font for Chinese characters if possible
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Colleges (XY)')
    ax1.set_ylabel('RMSE (Lower is Better)', color=color)
    bars = ax1.bar(df['College'], df['RMSE'], color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=45, ha='right')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Sample Size', color=color)  
    ax2.plot(df['College'], df['Sample Size'], color=color, marker='o', linestyle='dashed', linewidth=2, markersize=6)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Global Federated Model Performance (RMSE) Across Individual Colleges')
    fig.tight_layout()  
    
    save_path = os.path.join(base_dir, "college_fairness_rmse.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate_per_college()
