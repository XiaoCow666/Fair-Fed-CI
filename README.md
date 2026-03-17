# Fair-Fed-CI: Privacy-Preserving Student Performance Prediction via Federated Learning

<p align="center">
  <a href="#english">English</a> | <a href="#中文说明">中文说明</a>
</p>

---

<a name="english"></a>
## English

This repository contains the official PyTorch implementation for the `EnhancedNet` federated learning framework. The project aims to predict student academic performance (e.g., final scores or risk of failure) while strictly preserving the data privacy of individual colleges/departments using Non-IID Federated Learning (FL).

### Key Features

- **Privacy-Preserving Architecture**: Utilizes the Flower (`flwr`) framework to decouple student data across 17 distinct college clients. Data never leaves the local institution.
- **EnhancedNet Model**: A custom neural network integrating **Feature Attention** and **Self-Attention** mechanisms, specifically designed to handle heterogeneous (Non-IID) educational data distributions.
- **Subpopulation Fairness**: Capable of drawing accurate predictions without introducing bias towards majority-sample colleges, ensuring algorithmic fairness across the student body.
- **Global Interpretability**: Built-in SHAP (SHapley Additive exPlanations) analysis on the aggregated global model to provide transparent and explainable AI insights for educational interventions.

### 1. Environment Setup

It is recommended to use an Anaconda environment (Python 3.9+). 

```bash
conda create -n fair_fed_ci python=3.9
conda activate fair_fed_ci

# Install core dependencies
pip install torch pandas numpy scikit-learn matplotlib shap flwr ray
```

### 2. Project Structure

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

### 3. Data Flow & Execution Pipeline

1. **Data Preprocessing**: `python src/data_core.py` (Generates `processed_data_v2.csv`).
2. **FL Training**: `python run_training.py` (Trains EnhancedNet across 17 colleges).
3. **Interpretability**: `python src/explainability.py` (Generates SHAP summary).
4. **Fairness**: `python evaluate_fairness.py` (Evaluates per-college RMSE).

---

<a name="中文说明"></a>
## 中文说明

本仓库包含 `EnhancedNet` 联邦学习框架的官方 PyTorch 实现。本项目旨在预测学生的学业表现（如最终成绩或不及格风险），同时通过非独立同分布（Non-IID）联邦学习（FL）严格保护各学院/部门的数据隐私。

### 核心特性

- **隐私保护架构**：利用 Flower (`flwr`) 框架将学生数据解耦到 17 个不同的学院客户端。数据绝不出域，始终保留在本地机构。
- **EnhancedNet 模型**：一种自定义神经网络，集成了**特征注意力（Feature Attention）**和**自注意力（Self-Attention）**机制，专为处理异构（Non-IID）教育数据分布而设计。
- **子群体公平性**：能够在不对大样本学院产生偏见的情况下得出准确预测，确保整个学生群体的算法公平性。
- **全局可解释性**：对聚合后的全局模型进行内置的 SHAP（SHapley Additive exPlanations）分析，为教育干预提供透明、可解释的 AI 洞察。

### 1. 环境配置

建议使用 Anaconda 环境（Python 3.9+）。

```bash
conda create -n fair_fed_ci python=3.9
conda activate fair_fed_ci

# 安装核心依赖
pip install torch pandas numpy scikit-learn matplotlib shap flwr ray
```

### 2. 项目结构

```text
Fair-Fed-CI-v2/
├── data/
│   ├── raw_data.csv             # 原始多学院学生记录
│   └── processed_data_v2.csv    # 归一化和独热编码后的特征集 (110 维)
├── src/
│   ├── data_core.py             # 数据清洗、特征工程和归一化
│   ├── model_core.py            # EnhancedNet 和 VanillaMLP 模型架构
│   ├── fed_core.py              # Flower 联邦客户端、服务端逻辑及评估指标
│   ├── explainability.py        # SHAP 全局模型解释
│   ├── plot_comparison.py       # 对比消融实验和基线表现的可视化
│   └── centralized_baseline.py  # 中心化训练（非联邦）的理论上限脚本
├── run_training.py              # 主入口：运行 EnhancedNet 联邦模拟
├── run_ablation.py              # 辅助入口：运行 Vanilla MLP 联邦消融实验
├── evaluate_fairness.py         # 对最终全局模型进行子群体公平性分析
└── README.md                    # 项目文档
```

### 3. 数据流与执行管线

1. **数据预处理**：运行 `python src/data_core.py`（生成 `processed_data_v2.csv`）。
2. **联邦训练**：运行 `python run_training.py`（在 17 个学院间训练 EnhancedNet）。
3. **可解释性分析**：运行 `python src/explainability.py`（生成 SHAP 摘要图）。
4. **公平性评估**：运行 `python evaluate_fairness.py`（评估各学院独立的 RMSE 误差）。

---

## Contact & Authors
*(Please insert your lab/personal contact information here).*
