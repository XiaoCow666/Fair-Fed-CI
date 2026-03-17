import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Load data
    enhanced_fl_path = os.path.join(base_dir, "training_history.csv")
    vanilla_fl_path = os.path.join(base_dir, "baseline_history.csv")
    centralized_path = os.path.join(base_dir, "centralized_history.csv")
    
    results = {}
    
    if os.path.exists(enhanced_fl_path):
        results['EnhancedNet (Federated)'] = pd.read_csv(enhanced_fl_path)
    if os.path.exists(vanilla_fl_path):
        results['Vanilla MLP (Federated)'] = pd.read_csv(vanilla_fl_path)
    if os.path.exists(centralized_path):
        results['EnhancedNet (Centralized)'] = pd.read_csv(centralized_path)
    
    if not results:
        print("No result files found.")
        return

    # 2. Plotting RMSE Convergence
    plt.figure(figsize=(10, 6))
    for label, df in results.items():
        # Ensure we use the correct column for x-axis (round or epoch)
        x_col = 'round' if 'round' in df.columns else 'epoch'
        plt.plot(df[x_col], df['rmse'], label=label, marker='o', markersize=4)
    
    plt.title('RMSE Comparison: Federated vs Centralized vs Ablation')
    plt.xlabel('Communication Rounds / Epochs')
    plt.ylabel('RMSE (Normalized)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(base_dir, "final_comparison_rmse.png")
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")
    
    # 3. Plotting R2 Comparison
    plt.figure(figsize=(10, 6))
    for label, df in results.items():
        x_col = 'round' if 'round' in df.columns else 'epoch'
        plt.plot(df[x_col], df['r2'], label=label, marker='s', markersize=4)
        
    plt.title('R2 Score Comparison')
    plt.xlabel('Communication Rounds / Epochs')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # R2 can be very negative in early FL rounds, let's limit y axis for visibility
    plt.ylim(-10, 1) 
    
    save_path = os.path.join(base_dir, "final_comparison_r2.png")
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    plot_comparison()
