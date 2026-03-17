# Fair-Fed-CI: Privacy-Preserving Student Performance Prediction via Federated Learning

This repository contains the official PyTorch implementation for the `EnhancedNet` federated learning framework. The project aims to predict student academic performance (e.g., final scores or risk of failure) while strictly preserving the data privacy of individual colleges/departments using Non-IID Federated Learning (FL).

## Key Features

- **Privacy-Preserving Architecture**: Utilizes the Flower (`flwr`) framework to decouple student data across 17 distinct college clients. Data never leaves the local institution.
- **EnhancedNet Model**: A custom neural network integrating **Feature Attention** and **Self-Attention** mechanisms, specifically designed to handle heterogeneous (Non-IID) educational data distributions.
- **Subpopulation Fairness**: Capable of drawing accurate predictions without introducing bias towards majority-sample colleges, ensuring algorithmic fairness across the student body.
- **Global Interpretability**: Built-in SHAP (SHapley Additive exPlanations) analysis on the aggregated global model to provide transparent and explainable AI insights for educational interventions.

---

## 1. Environment Setup

It is recommended to use an Anaconda environment (Python 3.9+). 

```bash
conda create -n fair_fed_ci python=3.9
conda activate fair_fed_ci

# Install core dependencies
pip install torch pandas numpy scikit-learn matplotlib shap flwr ray
```

---

## 2. Project Structure

```text
Fair-Fed-CI-v2/
├── data/
│   ├── raw_data.csv             # Original raw multi-college student records
│   └── processed_data_v2.csv    # Normalized & One-hot encoded feature set (110 dims)
├── src/
│   ├── data_core.py             # Data cleaning, feature engineering, and normalization
│   ├── model_core.py            # EnhancedNet & VanillaMLP PyTorch architectures
│   ├── fed_core.py              # Flower FL Client, Server logic, and evaluation metrics
│   ├── explainability.py        # SHAP global model interpretation
│   ├── plot_comparison.py       # Visualizes ablation & baseline performance
│   └── centralized_baseline.py  # Script for centralized (non-FL) theoretical upper bound
├── run_training.py              # Main entry point: Runs EnhancedNet FL simulation
├── run_ablation.py              # Secondary entry point: Runs Vanilla MLP FL simulation
├── evaluate_fairness.py         # Subpopulation analysis on the final global model
└── README.md                    # Project documentation
```

---

## 3. Data Flow & Execution Pipeline

### Step 3.1: Data Preprocessing
Before running any federated simulations, the raw data must be cleaned, categorical features one-hot encoded, and continuous features normalized.
```bash
python src/data_core.py
```
*(This generates `data/processed_data_v2.csv` with exactly 110 engineered features).*

### Step 3.2: Train the Global Federated Model (`EnhancedNet`)
Execute the main script to spawn 17 local college clients and coordinate 50 communication rounds of federated aggregation.
```bash
python run_training.py
```
*(This will output `training_history.csv` showing round-by-round MSE/RMSE/R2 metrics and saves the final aggregated weights to `global_model_weights.npz`).*

### Step 3.3: Global Model Interpretability (SHAP)
To understand *why* the global model makes its predictions, run the explainability script on the saved global weights.
```bash
python src/explainability.py
```
*(This generates `shap_summary.png`, highlighting which features (e.g., Failed Courses, Core Course Scores) contribute most heavily to the performance prediction).*

---

## 4. Academic Experiments & Baselines

To support rigorous academic evaluation, this repository includes complete scripts for ablation studies and fairness validations.

### A. Ablation Study (Vanilla MLP)
To prove the effectiveness of the Attention layers in `EnhancedNet`, run the same federated simulation using a standard MLP:
```bash
python run_ablation.py
```
*(Outputs `baseline_history.csv`).*

### B. Centralized Baseline (Upper Bound)
To calculate the "Privacy-Utility Trade-off", evaluate the performance if all data were illegally pooled together into one central server:
```bash
python src/centralized_baseline.py
```
*(Outputs `centralized_history.csv`).*

### C. Compare & Visualize
Generate comparative plots (RMSE & R2) comparing the three configurations:
```bash
python src/plot_comparison.py
```
*(Generates `final_comparison_rmse.png` & `final_comparison_r2.png`).*

### D. Fairness / Subpopulation Analysis
Verify that the federated model performs equitably across all 17 colleges, regardless of their individual sample sizes:
```bash
python evaluate_fairness.py
```
*(Generates `college_evaluation.csv` and `college_fairness_rmse.png`).*

---

## 5. Contact & Authors
*(Please insert your lab/personal contact information here before submission).*
