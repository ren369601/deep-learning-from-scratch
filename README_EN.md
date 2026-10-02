# Transformer From Scratch

[中文](README.md) | [English](README_EN.md)

This is a long-term learning project that starts with Python and mathematical fundamentals, progresses through deep learning, and ultimately builds a Transformer from scratch.

The goal is not simply to call an existing Transformer API. The project focuses on understanding and implementing every essential component: matrix operations, gradient descent, backpropagation, Attention, Multi-Head Attention, the Encoder, the Decoder, and complete model training and inference.

## Goals

By completing this project, the learner should be able to:

- Understand the core principles of neural network training.
- Work confidently with NumPy arrays, PyTorch tensors, and matrix operations.
- Implement common neural network modules independently.
- Understand and implement Attention and Multi-Head Attention.
- Build a Transformer Encoder, Decoder, and complete model from scratch.
- Read and reproduce the core architecture from "Attention Is All You Need."
- Read deep learning papers and GitHub projects and debug models independently.

## Current Progress

| Stage | Topic | Status |
|---|---|---|
| Week 1 | Python fundamentals and linear models | Complete |
| Week 2 | NumPy, matrix operations, and gradient descent | Complete |
| Week 3 | Neurons, activation functions, and backpropagation | Up next |
| Week 4 | PyTorch fundamentals and MLPs | Not started |
| Week 5 | Training workflows and CNNs | Not started |
| Week 6 | RNNs, LSTMs, and sequence modeling | Not started |
| Week 7 | Attention | Not started |
| Week 8 | Multi-Head Attention and masking | Not started |
| Week 9 | Transformer Encoder | Not started |
| Week 10 | Transformer Decoder and complete model | Not started |
| Week 11 | Paper reading and reproduction | Not started |
| Week 12 | Training, experiments, and project cleanup | Not started |

See the [12-week study plan](daily-weekly-study-plan-12-weeks.md) for the detailed schedule. A more gradual [24-week study plan](deep-learning-24-week-plan.md) is also available.

## Completed Work

### Week 1: Python and Linear Models

- The linear model `y = wx + b` and multi-feature models.
- Python functions, lists, loops, and classes.
- Samples, batches, and batch prediction.
- Encapsulating a linear model in a class.
- Implementing MSE and searching candidate parameters.
- Building a plain-Python linear regression prototype.

Review material: [Week 1 Review](week01/review.md)

### Week 2: NumPy and Gradient Descent

- NumPy arrays, shape, ndim, and dtype.
- Reshape, transpose, and broadcasting.
- Dot products and matrix multiplication.
- Vectorized MSE with NumPy.
- Gradient descent, learning rates, and training loops.
- Multi-feature linear regression using `X.T @ errors`.

Review material: [Week 2 Review](week02/review.md)

## Learning Path

```text
Python / NumPy
        ↓
Mathematical Foundations and Gradient Descent
        ↓
Neural Networks and Backpropagation
        ↓
PyTorch and MLPs
        ↓
CNN / RNN / LSTM
        ↓
Attention
        ↓
Multi-Head Attention and Masking
        ↓
Transformer Encoder / Decoder
        ↓
Paper Reproduction and Complete Project
```

## Repository Structure

```text
.
├── README.md
├── README_EN.md
├── requirements.txt
├── daily-weekly-study-plan-12-weeks.md
├── deep-learning-24-week-plan.md
├── week01/
│   ├── dayXX_knowledge.md
│   ├── dayXX_python_syntax.md
│   ├── dayXX_*.py
│   ├── notes.md
│   └── review.md
└── week02/
    ├── dayXX_knowledge.md
    ├── dayXX_numpy_syntax.md
    ├── dayXX_*.py
    ├── linear_regression_numpy.py
    ├── notes.md
    └── review.md
```

Each week follows the same learning format:

```text
Concepts -> Syntax and Formulas -> Examples -> Exercises -> Homework -> Review
```

## Setup

Clone the repository:

```bash
git clone https://github.com/ren369601/deep-learning-from-scratch.git
cd deep-learning-from-scratch
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Examples

Run the Week 1 linear regression review:

```bash
python week01/day06_week01_review.py
```

Run the Week 2 NumPy multi-feature linear regression project:

```bash
python week02/linear_regression_numpy.py
```

Run the Week 2 integrated review:

```bash
python week02/day07_week02_review.py
```

## Learning Principles

- Study theory and code together.
- Analyze shapes before every matrix operation.
- Attempt each exercise independently before checking the explanation or examples.
- Go beyond making code run: explain what every step does.
- Review notes, run all code, and commit progress every week.

## Final Deliverable

The completed project will contain an implementation that does not rely on a prebuilt Transformer model API, including:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Padding and Causal Masks
- Positional Encoding
- Feed Forward Networks
- Layer Normalization and Residual Connections
- Transformer Encoder and Decoder
- Training loops and autoregressive inference
- Reading and reproduction notes for "Attention Is All You Need"

This repository is an active learning project and will continue to evolve.
