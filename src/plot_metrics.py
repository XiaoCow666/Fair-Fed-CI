import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_history():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_file = os.path.join(base_dir, "training_history.csv")
    
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found. Run federated training first.")
        return
        
    df = pd.read_csv(history_file)
    
    # We will create a figure with subplots for different metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fair-Fed-CI v2 - Federated Training Convergence (50 Rounds)', fontsize=16)
    
    # Plot RMSE
    axs[0, 0].plot(df['round'], df['rmse'], 'b-', marker='o', markersize=4)
    axs[0, 0].set_title('Root Mean Squared Error (RMSE)')
    axs[0, 0].set_xlabel('Communication Round')
    axs[0, 0].set_ylabel('RMSE')
    axs[0, 0].grid(True)
    
    # Plot MAE
    axs[0, 1].plot(df['round'], df['mae'], 'g-', marker='s', markersize=4)
    axs[0, 1].set_title('Mean Absolute Error (MAE)')
    axs[0, 1].set_xlabel('Communication Round')
    axs[0, 1].set_ylabel('MAE')
    axs[0, 1].grid(True)
    
    # Plot MSE
    axs[1, 0].plot(df['round'], df['mse'], 'r-', marker='^', markersize=4)
    axs[1, 0].set_title('Mean Squared Error (MSE)')
    axs[1, 0].set_xlabel('Communication Round')
    axs[1, 0].set_ylabel('MSE')
    axs[1, 0].grid(True)
    
    # Plot R2 Score
    axs[1, 1].plot(df['round'], df['r2'], 'm-', marker='d', markersize=4)
    axs[1, 1].set_title('R-Squared (R²)')
    axs[1, 1].set_xlabel('Communication Round')
    axs[1, 1].set_ylabel('R² Score')
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    out_path = os.path.join(base_dir, "training_convergence.png")
    plt.savefig(out_path, dpi=300)
    print(f"Convergence plot saved successfully to {out_path}")

if __name__ == '__main__':
    plot_training_history()
