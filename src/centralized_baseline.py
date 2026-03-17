import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add current directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_core import EnhancedNet
from data_core import DataCore

def train_centralized():
    print("Starting Centralized Baseline Training...")
    
    # 1. Load Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed_data_v2.csv")
    
    if not os.path.exists(data_path):
        print("Processed data not found. Run data_core.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Define columns to match fed_core and DataCore
    target_col = 'target_score'
    exclude_cols = ['JMXH', target_col, 'sensitive_attribute', 'XY']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_np = df[feature_cols].values.astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X, y)
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    # 2. Setup Model
    device = torch.device("cpu") # For simplicity or use GPU if available
    model = EnhancedNet(len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    history = {"epoch": [], "mse": [], "rmse": [], "mae": [], "r2": []}
    
    # 3. Training Loop
    # We use 50 epochs as a proxy for 50 federated rounds
    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred, _ = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            
        # Evaluation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, _ = model(batch_x)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mse = float(mean_squared_error(all_targets, all_preds))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(all_targets, all_preds))
        r2 = float(r2_score(all_targets, all_preds))
        
        print(f"Epoch {epoch}: MSE={mse:.4f}, R2={r2:.4f}")
        
        history["epoch"].append(epoch)
        history["mse"].append(mse)
        history["rmse"].append(rmse)
        history["mae"].append(mae)
        history["r2"].append(r2)
        
    # 4. Save results
    out_csv = os.path.join(base_dir, "centralized_history.csv")
    pd.DataFrame(history).to_csv(out_csv, index=False)
    print(f"Centralized results saved to {out_csv}")

if __name__ == "__main__":
    train_centralized()
