# TensorFlow MNIST Baseline + Regularization

## Goal
Train a baseline neural network on MNIST and reduce overfitting using Dropout.

## Models
**Baseline**
- Flatten → Dense(128, ReLU) → Dense(10, Softmax)

**Improved**
- Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(10, Softmax)

## Results
- Baseline test accuracy: 0.9689000248908997
- Dropout test accuracy: 0.9742000102996826

## Plots
![Baseline Train vs Val](results/mnist_baseline_acc.png)
![Validation Comparison](results/mnist_compare_val_acc.png)

## What I learned
- How to build/train/evaluate models in TensorFlow/Keras
- How to detect overfitting using train vs validation curves
- How Dropout affects generalization
