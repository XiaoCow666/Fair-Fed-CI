# 系统演进对比报告：Legacy (cfx) vs Fair-Fed-CI v2

## 1. 核心架构差异 (Architecture)

| 维度 | 旧版本 (Legacy / cfx) | 新版本 (Fair-Fed-CI v2) | 评价 |
| :--- | :--- | :--- | :--- |
| **训练模式** | **集中式 (Centralized)** | **联邦学习 (Federated Learning)** | **质的飞跃**。旧版需要把所有学生数据汇总到一个 CSV，存在隐私风险；新版数据不出学院，仅交换模型参数。 |
| **网络结构** | 普通 MLP (全连接网络) | **Split-Layer Attention MLP** | 新版引入了**分层架构**（共享层+私有层）和**注意力机制**。 |
| **任务类型** | 分类 (及格/不及格) | **回归 (具体分数预测)** | 新版能预测具体分数（如 78.5 分），比单纯的二分类更精准。 |

## 2. 可解释性 (Interpretability)

- **旧版本**:
    - 依赖外部工具 (LIME/SHAP)。
    - 属于“事后解释” (Post-hoc)，计算慢，且不是模型自带的逻辑。
    - 只能分析“整体”特征重要性。

- **新版本**:
    - **内生解释 (Intrinsic)**: 模型自带 Attention 层。
    - **实时性**: 预测的同时直接输出权重。
    - **个性化**: 能针对**每个学生**给出专属的归因分析（如“张三是因为高数挂了，李四是因为英语挂了”）。

## 3. 公平性 (Fairness)

- **旧版本**:
    - **无 (None)**。
    - 仅追求准确率，忽略了不同学院/群体间的差异。

- **新版本**:
    - **公平性约束 (Fairness Loss)**。
    - 在损失函数中显式加入了 `Variance(Group Errors)`，强迫模型在不同学院间保持“一碗水端平”。

## 4. 代码实现细节

### 4.1 数据处理
- **旧版 (`only.py`)**: 手动合并列（如把“程序设计基础”和“高级程序设计”硬编码合并）。
- **新版 (`data_core.py`)**: 自动化特征工程，动态提取 Top-N 课程，无需人工干预。

### 4.2 模型对比
- **旧版 (`NN.py`)**:
  ```python
  self.net = nn.Sequential(
      nn.Linear(input_size, 128),
      nn.ReLU(),
      ...
  )
  ```
- **新版 (`model_core.py`)**:
  ```python
  # 1. Attention (Explainability)
  self.feature_attention = FeatureAttention(input_dim)
  # 2. Encoder (Shared Knowledge)
  self.encoder = ...
  # 3. Head (Personalized)
  self.head = ...
  ```

## 5. 总结
旧版本 (`cfx`) 更像是一个**“数据挖掘作业”**，使用了传统的机器学习方法（SVM, RF, MLP）在集中数据上跑结果。

新版本 (`v2`) 是一个**“工业级联邦系统”**，它解决了：
1.  **数据孤岛** (通过联邦学习)。
2.  **黑盒问题** (通过 Attention)。
3.  **算法歧视** (通过 Fairness Loss)。
4.  **非独立同分布 (Non-IID)** (通过 Split-Layer)。

这是从“实验代码”到“前沿研究系统”的跨越。
