"""
Fair-Fed-CI: Privacy-Preserving Student Performance Prediction
-------------------------------------------------------------------------
Ablation Entry Point: Initiates the Federated Learning simulation using the 
baseline 'VanillaMLP' model architecture without attention mechanisms.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fed_core import run_simulation

if __name__ == "__main__":
    # Run simulation with Vanilla MLP and save to a unique filename
    run_simulation(model_type="vanilla", output_filename="baseline_history.csv")
