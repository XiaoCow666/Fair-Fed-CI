import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from collections import OrderedDict

from src.model_core import FairAttentionMLP
from src.data_core import DataCore

def visualize_attention():
    print("Starting Visualization Pipeline...")
    
    # 1. Load Data
    print("Loading data...")
    # We can reuse DataCore logic or just load the processed file directly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    data_path = os.path.join(base_dir, "data", "processed_data_v2.csv")
    
    if not os.path.exists(data_path):
        print("Error: Processed data not found.")
        return

    df = pd.read_csv(data_path)
    
    # Identify feature columns (same logic as fed_core)
    target_col = 'avg_score'
    # We need to exclude non-feature columns. 
    # In data_core, we had 'JMXH' and potentially 'sensitive_attribute' if it wasn't dropped.
    # Let's assume 'sensitive_attribute' is the column name for the sensitive attribute if it exists.
    # But wait, in fed_core we saw: sensitive_col = 'sensitive_attribute'.
    # And we saw: feature_cols = [c for c in df.columns if c not in ['JMXH', sensitive_col, target_col]]
    # We need to be consistent.
    
    # Let's check columns dynamically
    exclude_cols = ['JMXH', 'avg_score', 'sensitive_attribute']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    X = torch.tensor(df[feature_cols].astype(float).values, dtype=torch.float32)
    
    # 2. Load Model Weights
    print("Loading model weights...")
    weights_path = os.path.join(base_dir, "global_model_weights.npz")
    if not os.path.exists(weights_path):
        print("Error: Global model weights not found.")
        return
        
    loaded = np.load(weights_path)
    # The weights are a list of arrays corresponding to get_shared_parameters() order
    # We need to instantiate the model first to get the keys
    model = FairAttentionMLP(input_dim=len(feature_cols))
    
    shared_params_keys = model.get_shared_parameters().keys()
    
    # Map loaded arrays to keys
    state_dict = OrderedDict()
    for key, arr in zip(shared_params_keys, [loaded[f"arr_{i}"] for i in range(len(loaded))]):
        state_dict[key] = torch.tensor(arr)
        
    # Load state dict (strict=False because we are missing the head)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 3. Forward Pass to get Attention Weights
    print("Computing attention weights...")
    with torch.no_grad():
        # We only care about the attention weights, which come from the first layer
        # The forward pass returns (prediction, attn_weights)
        # But prediction will be garbage because the head is not trained/loaded (it's random initialized)
        # However, attn_weights depend ONLY on feature_attention layer, which IS loaded.
        _, attn_weights = model(X)
        
    # 4. Visualize
    print("Generating heatmap...")
    
    # Set Chinese Font
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows Chinese Font
    plt.rcParams['axes.unicode_minus'] = False # Fix minus sign
    
    # Average attention per feature across all samples
    avg_attention = attn_weights.mean(dim=0).numpy()
    
    # Create a DataFrame for plotting
    attn_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': avg_attention
    })
    
    # Filter for Course Features (CJ_) to answer "Which Subject"
    course_attn_df = attn_df[attn_df['Feature'].str.startswith('CJ_')].copy()
    
    # If no course features (fallback), use all
    if course_attn_df.empty:
        print("Warning: No CJ_ features found. Using all features.")
        plot_df = attn_df.sort_values(by='Importance', ascending=False).head(15)
    else:
        # Clean feature names: Remove 'CJ_' prefix
        course_attn_df['Feature'] = course_attn_df['Feature'].str.replace('CJ_', '')
        plot_df = course_attn_df.sort_values(by='Importance', ascending=False).head(15)
    
    print("Top 15 Important Features:")
    print(plot_df)
    
    # Plot Bar Chart (Global Importance)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=plot_df, palette='viridis')
    plt.title('各科目对成绩预测的影响程度 (Global Feature Importance)', fontsize=16)
    plt.xlabel('平均注意力权重 (Average Attention Weight)', fontsize=12)
    plt.ylabel('科目名称 (Subject)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "global_feature_importance.png"))
    print(f"Saved global importance plot to results/global_feature_importance.png")
    
    # Plot Heatmap (Sample of students)
    # Take top 20 students and Top 15 features
    sample_size = 20
    
    # Get indices of top features
    top_indices = [feature_cols.index(f"CJ_{name}" if "CJ_" not in name and not course_attn_df.empty else name) for name in plot_df['Feature']]
    
    subset_attn = attn_weights[:sample_size, top_indices].numpy()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(subset_attn, xticklabels=plot_df['Feature'], yticklabels=[f"学生 {i+1}" for i in range(sample_size)], cmap='viridis', annot=False)
    plt.title(f'个体学生科目关注度热力图 (Top {sample_size} Students)', fontsize=16)
    plt.xlabel('科目名称 (Subject)', fontsize=12)
    plt.ylabel('学生 (Student)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "attention_heatmap_sample.png"))
    print(f"Saved heatmap to results/attention_heatmap_sample.png")

if __name__ == "__main__":
    visualize_attention()
