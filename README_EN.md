# Deep Learning Course Study Log

[中文](README.md) | [English](README_EN.md)

This repository documents a journey through deep learning from the fundamentals. The current focus is [Professor Runmin Cong's Deep Learning course](https://rmcong.github.io/proj_deep_learning_ProfessionalCourse.html): understand the course material, complete its experiments, and prepare to join the research group. The repository name is a legacy name; implementing a Transformer is not a graduation requirement for this course.

The daily format is concept notes, a syntax or formula reference, example code, small exercises, homework, and a weekly review. Exercises are attempted independently before review.

## Progress

| Week | Topic | Status |
|---|---|---|
| Week 1 | Python and linear models | Complete |
| Week 2 | NumPy, matrices, and gradient descent | Complete |
| Week 3 | Neural networks, losses, and backpropagation | Complete |
| Week 4 | PyTorch, autograd, MLPs, training, and validation | Complete |
| Week 5 | Foundations, optimization, regularization, and lab review | Days 1-7 code/review complete; Day 1 application analysis pending |
| Weeks 6-12 | CNNs, RNNs, unsupervised learning, frontiers, and capstone | Upcoming |

The [12-week study plan](daily-weekly-study-plan-12-weeks.md) is the single current schedule. Its first four weeks are preserved; the remaining weeks cover all 12 lectures and five labs listed on the course site. The final lab offers five project choices, of which one will be completed.

## Completed Exercises

- [Week 1](week01/): plain-Python linear models, multiple features and batches, MSE, and review.
- [Week 2](week02/): NumPy arrays and shapes, matrix multiplication, gradient descent, linear regression, and review.
- [Week 3](week03/): neurons, activation functions, losses, backpropagation, a two-layer NumPy MLP, and review.
- [Week 4](week04/): tensors, autograd, nn.Module, optimizers, a PyTorch MLP, XOR training/validation, and the [weekly review](week04/review.md).
- [Week 5](week05/): classification, framework labs, MLP/K-fold validation, optimization, regularization, and the [review log](week05/review.md). Days 5-7 include assistant-completed reference solutions; the personal Day 1 application analysis is pending.

A typical day contains dayXX_knowledge.md, a Python/NumPy/PyTorch syntax or formula note, and dayXX_*.py. Weekly reflections are kept in review.md.

## Course Roadmap

1. Fill remaining foundation gaps; optimization and regularization.
2. CNNs and the convolutional network lab.
3. RNNs and the sequence modeling lab.
4. Unsupervised learning.
5. Attention, generative models, deep reinforcement learning, graph neural networks, and the course's industry topic.
6. Complete one capstone lab and prepare a reproducible report for research-group entry.

## Setup

Use a Python virtual environment. Dependency versions are listed in [requirements.txt](requirements.txt). The exercises are designed to run on a CPU by default.

~~~powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python week05/day07_week05_review.py
~~~

On macOS/Linux, activate the environment with source .venv/bin/activate. Read the concept and syntax notes in a day folder before running its exercise.

## Completion Criteria

Explain the main models and learning objectives, run and analyze experiments independently, separate training from validation data, and submit one reproducible capstone report. The research direction after joining the group will follow the professor's guidance.
