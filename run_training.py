"""
Fair-Fed-CI: Privacy-Preserving Student Performance Prediction
-------------------------------------------------------------------------
Main Training Entry Point: Initiates the 17-node Federated Learning simulation
using the 'EnhancedNet' model architecture.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fed_core import run_simulation

if __name__ == "__main__":
    run_simulation()
