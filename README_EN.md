# Deep Learning Foundations and Experiments

[中文](README.md) | [English](README_EN.md)

A Python, NumPy, and PyTorch learning repository following [Professor Runmin Cong's Deep Learning course](https://rmcong.github.io/proj_deep_learning_ProfessionalCourse.html). It documents foundational implementations, model training and validation, and controlled comparisons. This is coursework in progress, not original research.

## Completed Work

- **Foundations (Weeks 1–3):** linear regression, matrix operations, gradient descent, backpropagation, and a two-layer NumPy MLP.
- **PyTorch (Weeks 4–5):** autograd, classification, training/validation splits, K-fold validation, optimizer and regularization comparisons.
- **CNNs (Week 6, Days 1–6):** manual 2D cross-correlation, multi-channel and transposed convolution; image classification; teaching versions of LeNet/Inception; residual blocks and paired plain/residual network experiments.

## Selected Experiments

| Implementation | What It Demonstrates |
|---|---|
| [NumPy MLP](week03/day06_mlp_train.py) | Forward propagation, gradient computation, and parameter updates |
| [Manual transposed convolution](week06/day04_conv_variants.py) | Overlapping contributions and three numerical checks against PyTorch |
| [Residual network comparison](week06/day06_resnet_lab.py) | Three initialization seeds; matched parameter counts, initial weights, and batch order; only residual addition is toggled |

On synthetic horizontal/vertical line images, both networks have 3,658 parameters and reach 100% validation accuracy. The residual network has lower final validation loss in these runs. [Settings, results, and limitations](week06/day06_knowledge.md) are documented; these results do not establish natural-image performance or reproduce the original paper.

## Run

Dependencies are pinned in [requirements.txt](requirements.txt). CPU is sufficient. After creating and activating a virtual environment, run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -X utf8 week06/day06_resnet_lab.py
```

Each study day includes concept notes, a syntax reference, and exercises. Progress is through Week 6 Day 6; see the [12-week plan](daily-weekly-study-plan-12-weeks.md) for remaining work. The course is not yet complete. Explanations and some reference solutions were written with AI assistance; completion and execution records are retained rather than presented as wholly independent implementations.
