"""
Fair-Fed-CI: Privacy-Preserving Student Performance Prediction
-------------------------------------------------------------------------
Interpretability Script: Applies SHAP DeepExplainer to the aggregated global 
model to extract and visualize feature importance across all colleges.
"""
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Using the correct model architecture we just integrated
from model_core import EnhancedNet

def load_global_model(feature_dim, weights_path):
    print("Loading global model weights...")
    model = EnhancedNet(feature_dim)
    
    # Load NPZ file
    data = np.load(weights_path, allow_pickle=True)
    # the FL server saves weights chronologically according to state_dict.values
    # However, since this is just an evaluation script, we will load what was shared
    
    # A safer way to load weights saved by flwr is to map them to the shared parameters
    shared_params = model.get_shared_parameters()
    # The arrays in npz are named 'arr_0', 'arr_1', etc in the order they were yielded
    arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
    
    with torch.no_grad():
        for param, arr in zip(shared_params.values(), arrays):
            param.copy_(torch.from_numpy(arr))
            
    # Note: Personalized Head remains randomly initialized or requires specific local client weights.
    # We only care about the shared feature representation for global SHAP analysis.
    return model

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        # We only return the prediction, ignoring the attention weights for SHAP DeepExplainer
        pred, _ = self.model(x)
        return pred

def explain_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed_data_v2.csv")
    weights_path = os.path.join(base_dir, "global_model_weights.npz")
    
    if not os.path.exists(weights_path):
        print("Error: global_model_weights.npz not found. Must run FL simulation first.")
        return

    print("Loading dataset for evaluation...")
    df = pd.read_csv(data_path)
    
    target_col = 'target_score'
    sensitive_col = 'sensitive_attribute'
    feature_cols = [c for c in df.columns if c not in ['JMXH', sensitive_col, target_col]]
    
    X = torch.tensor(df[feature_cols].astype(float).values, dtype=torch.float32)
    
    # Load model
    base_model = load_global_model(len(feature_cols), weights_path)
    base_model.eval()
    
    model = ModelWrapper(base_model)
    
    print("Applying SHAP DeepExplainer...")
    # Use a random subset for the background reference to compute SHAP values quickly
    background = X[torch.randperm(X.shape[0])[:100]]
    explainer = shap.DeepExplainer(model, background)
    
    # Explain a larger subset
    test_samples = X[torch.randperm(X.shape[0])[:500]]
    shap_values = explainer.shap_values(test_samples)
    
    # shap_values from DeepExplainer on a multi-output is a list, but for 1D output it's a tensor/array
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0]
    else:
        shap_values_to_plot = shap_values
        
    print("Generating SHAP summary plot...")
    plt.figure()
    
    # Provide the actual feature names from the dataframe
    # Also fix font configuration for Chinese characters if applicable
    plt.rcParams['font.sans-serif'] = ['SimHei'] # To avoid square boxes for Chinese text
    plt.rcParams['axes.unicode_minus'] = False # Fix minus sign
    
    shap.summary_plot(shap_values_to_plot, test_samples.numpy(), feature_names=feature_cols, show=False)
    
    out_path = os.path.join(base_dir, "shap_summary.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"SHAP explanation artifact saved to: {out_path}")

if __name__ == "__main__":
    explain_model()
