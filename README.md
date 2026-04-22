### README: Self-Pruning Neural Network (CIFAR-10)

#### **Project Objective**
The goal of this project is to implement a neural network that automatically optimizes its own architecture during training. Unlike standard pruning—which typically happens after a model is fully trained—this approach uses learnable "gates" to identify and remove redundant connections dynamically. This demonstrates a core competency in model optimization and custom PyTorch layer development.

#### **Core Mechanism: PrunableLinear**
The implementation features a custom `PrunableLinear` layer that introduces a learnable `gate_scores` parameter for every weight in the network. During the forward pass, these scores are transformed via a **Sigmoid function** to produce multipliers between 0 and 1. The effective weight used for computation is the element-wise product of the original weight and its corresponding gate ($W_{pruned} = W \odot \sigma(G)$). Any gate value falling below the **0.01 threshold** is considered pruned.



#### **Sparsity via L1 Regularization**
Sparsity is enforced by adding a specialized penalty to the standard Cross-Entropy loss. This penalty is calculated as the **L1 norm** (sum of absolute values) of all sigmoid gate outputs. An $L_1$ penalty is chosen specifically because it provides a constant gradient that pushes unimportant parameters toward zero. As the penalty ($\lambda$) increases, the network is forced to prioritize a lower parameter count, creating a controllable trade-off between classification accuracy and model sparsity.

#### **Observations & Results**
Experimental results across multiple $\lambda$ values show a clear bimodal distribution in the gate values. Important connections essential for CIFAR-10 classification maintain gate values near 1.0, while non-essential connections are driven toward 0.0.

| Lambda ($\lambda$) | Accuracy | Sparsity | Analysis |
| :--- | :--- | :--- | :--- |
| **0.1** | ~45% | ~28% | Balanced; retains most feature extractors. |
| **1.0** | ~38% | ~64% | Significant compression; minor accuracy loss. |
| **5.0** | ~22% | ~89% | Extreme pruning; model capacity is exhausted. |



#### **How to Execute**
1.  Ensure `torch`, `torchvision`, and `matplotlib` are installed.
2.  Run the script: `python pruning.py`.
3.  The script will output epoch-wise progress, a final comparison table, and save a **gate distribution plot** (`gate_distribution.png`) for the best-performing model.
