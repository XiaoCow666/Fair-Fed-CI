"""
Fair-Fed-CI: Privacy-Preserving Educational Prediction via Federated Learning
-------------------------------------------------------------------------
Federated Core Module: Contains the Flower FL Client logic, custom aggregation
strategies, and local training loop for non-IID data distribution.
"""
import flwr as fl
from flwr.common import parameters_to_ndarrays
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .model_core import EnhancedNet, VanillaMLP
from .data_core import DataCore

class FairClient(fl.client.NumPyClient):
    def __init__(self, cid, train_loader, test_loader, input_dim, device, fairness_gamma=0.1, model_type="enhanced"):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.fairness_gamma = fairness_gamma
        
        # Initialize model
        if model_type == "vanilla":
            self.model = VanillaMLP(input_dim).to(self.device)
        else:
            self.model = EnhancedNet(input_dim).to(self.device)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def get_parameters(self, config):
        """Return ONLY shared parameters (Feature Attention + Encoder)."""
        shared_params = self.model.get_shared_parameters()
        return [val.detach().cpu().numpy() for _, val in shared_params.items()]

    def set_parameters(self, parameters):
        """Update ONLY shared parameters."""
        shared_params = self.model.get_shared_parameters()
        params_dict = zip(shared_params.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Load only shared keys, strict=False to ignore missing head params
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        epoch_loss = 0.0
        for epoch in range(5): # Local epochs
            for batch_x, batch_y, batch_sensitive in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                pred, _ = self.model(batch_x)
                
                # 1. Performance Loss
                mse_loss = self.criterion(pred, batch_y)
                
                # 2. Fairness Loss (Variance of Group Errors)
                fairness_loss = self._calculate_fairness_loss(pred, batch_y, batch_sensitive)
                
                # Total Loss
                loss = mse_loss + self.fairness_gamma * fairness_loss
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
        
        # Save Client 0 model (Simulating a deployed system)
        if str(self.cid) == "0":
            torch.save(self.model.state_dict(), "client_0_model.pth")
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": epoch_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y, _ in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred, _ = self.model(batch_x)
                
                loss += self.criterion(pred, batch_y).item() * len(batch_y)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        # Calculate Scikit-Learn Metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mse = float(mean_squared_error(all_targets, all_preds))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(all_targets, all_preds))
        r2 = float(r2_score(all_targets, all_preds))
        
        avg_loss = float(loss / len(self.test_loader.dataset))
        
        # NaN protection for Flower/Ray IPC
        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
        for k, v in metrics.items():
            if np.isnan(v) or np.isinf(v):
                metrics[k] = 0.0
        
        return avg_loss, len(self.test_loader.dataset), metrics

    def _calculate_fairness_loss(self, pred, target, sensitive):
        """Calculate variance of mean errors across groups."""
        errors = (pred - target).pow(2) # Squared error per sample
        
        # Group by sensitive attribute
        # Assuming sensitive is a tensor of group indices or one-hot
        # For simplicity, let's assume sensitive is a 1D tensor of group IDs for now
        # If it's one-hot, we convert to indices
        if sensitive.dim() > 1:
            groups = torch.argmax(sensitive, dim=1)
        else:
            groups = sensitive
            
        unique_groups = torch.unique(groups)
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=self.device)
            
        group_means = []
        for g in unique_groups:
            mask = (groups == g)
            if mask.sum() > 0:
                group_means.append(errors[mask].mean())
                
        if len(group_means) < 2:
            return torch.tensor(0.0, device=self.device)
            
        # Variance of group means
        group_means_tensor = torch.stack(group_means)
        return torch.var(group_means_tensor)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated parameters...")
            # Convert to numpy list
            params_np = parameters_to_ndarrays(aggregated_parameters)
            # Save to file
            np.savez("global_model_weights.npz", *params_np)
            
        return aggregated_parameters, aggregated_metrics

def run_simulation(model_type="enhanced", output_filename="training_history.csv"):
    """Run a simulated FL experiment."""
    print("Starting Fair-Fed-CI v2 Simulation...")
    
    # 1. Load Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed_data_v2.csv")
    
    if not os.path.exists(data_path):
        print("Processed data not found. Run data_core.py first.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Prepare Tensors
    # Identify columns
    target_col = 'target_score'
    sensitive_col = 'sensitive_attribute' # This was created in data_core but might be dropped or encoded.
    # In data_core, we kept 'sensitive_attribute' as a column before encoding? 
    # Wait, in data_core: self.df[self.sensitive_col] = self.df['XY']
    # Then get_dummies. So 'sensitive_attribute' should be there if not dropped.
    # But get_dummies might have kept it if we didn't specify columns correctly or if we explicitly kept it.
    # Let's check columns in data_core.py... 
    # "self.feature_cols = [c for c in self.df.columns if c not in ['JMXH', self.sensitive_col, self.target_col]]"
    # So sensitive_col IS in the dataframe.
    
    # We need to map sensitive_col (string) to indices for the loss function
    if df[sensitive_col].dtype == 'object':
        df[sensitive_col] = df[sensitive_col].astype('category').cat.codes
    
    feature_cols = [c for c in df.columns if c not in ['JMXH', sensitive_col, target_col]]
    
    X = torch.tensor(df[feature_cols].astype(float).values, dtype=torch.float32)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).view(-1, 1)
    s = torch.tensor(df[sensitive_col].values, dtype=torch.long)
    
    # 3. Partition Data by College (Non-IID)
    client_data = []
    
    # We group the dataframe by the sensitive_col (which was mapped from 'XY')
    grouped = df.groupby(sensitive_col)
    num_clients = len(grouped)
    print(f"Partitioning data into {num_clients} Non-IID clients based on College (XY).")
    
    for name, group in grouped:
        group_X = torch.tensor(group[feature_cols].astype(float).values, dtype=torch.float32)
        group_y = torch.tensor(group[target_col].values, dtype=torch.float32).view(-1, 1)
        group_s = torch.tensor(group[sensitive_col].values, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(group_X, group_y, group_s)
        
        # Split train/test (80/20) for this specific client
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        # if the dataset is too small, we might have issues with test_size=0, but assume it's large enough per college
        if train_size > 0 and test_size > 0:
            train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
        else:
            # Fallback if college has very few students
            train_ds, test_ds = dataset, dataset
            
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
        client_data.append((train_loader, test_loader))
        
    # 4. Define Client Fn
    def client_fn(cid):
        # flwr passes cid as string
        cid_int = int(cid)
        if cid_int >= len(client_data):
            cid_int = cid_int % len(client_data)
        train_loader, test_loader = client_data[cid_int]
        return FairClient(cid, train_loader, test_loader, len(feature_cols), torch.device("cpu"), model_type=model_type).to_client()

    # History tracker for plots
    history = {"round": [], "mse": [], "rmse": [], "mae": [], "r2": []}
    
    # Define metric aggregation function
    def weighted_average(metrics):
        examples = [num_examples for num_examples, _ in metrics]
        total_examples = sum(examples)
        
        # Aggregate multiple metrics
        aggregated_metrics = {}
        for key in ["mse", "rmse", "mae"]:
            weighted_sum = sum([num_examples * m[key] for num_examples, m in metrics])
            aggregated_metrics[key] = weighted_sum / total_examples
            
        # R2 score doesn't perfectly aggregate by weighted average, but it's an acceptable proxy for FL logs
        weighted_r2 = sum([num_examples * m["r2"] for num_examples, m in metrics])
        aggregated_metrics["r2"] = weighted_r2 / total_examples
        
        print(f" [METRICS] RMSE: {aggregated_metrics['rmse']:.4f} | MAE: {aggregated_metrics['mae']:.4f} | R2: {aggregated_metrics['r2']:.4f}")
        
        # Save to history
        current_round = len(history["round"]) + 1
        history["round"].append(current_round)
        for k, v in aggregated_metrics.items():
            history[k].append(v)
            
        # Write to csv every round
        out_csv = os.path.join(base_dir, output_filename)
        pd.DataFrame(history).to_csv(out_csv, index=False)
            
        return {"mse": aggregated_metrics["mse"]}

    # 5. Run Simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=SaveModelStrategy(
            evaluate_metrics_aggregation_fn=weighted_average,
        ), 
    )

if __name__ == "__main__":
    run_simulation()
