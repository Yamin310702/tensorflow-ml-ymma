# TensorFlow MNIST Classifier (Baseline + Regularization)

## Goal
Train a neural network on MNIST and reduce overfitting using Dropout.

## Dataset
- MNIST handwritten digits
- 60,000 training images
- 10,000 test images

## Models
### Baseline Model
- Input → Flatten
- Dense(128, ReLU)
- Dense(10, Softmax)

### Improved Model
- Input → Flatten
- Dense(128, ReLU)
- Dropout(0.3)
- Dense(10, Softmax)

## Results
- Baseline test accuracy: **0.9695000052452087**
- Improved test accuracy: **0.973800003528595**

## Training Curves
### Baseline
![Baseline Training Curve](mnist_baseline_acc.png)

### Baseline vs Dropout
![Validation Accuracy Comparison](mnist_compare_val_acc.png)

## What I Learned
- How to build and train models using TensorFlow/Keras
- How to diagnose overfitting using train/validation curves
- How regularization (Dropout) improves generalization
- How to save trained models for reuse

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook

## Limitations
- The current models use fully connected (dense) layers and do not explicitly leverage the spatial structure of image data.
- Convolutional Neural Networks (CNNs), which are better suited for image tasks, would likely achieve higher accuracy.
- Hyperparameter tuning was limited to basic experimentation; more systematic approaches could further improve performance.

## Future Work
- Implement convolutional neural networks (CNNs) to better capture spatial features.
- Apply data augmentation techniques to improve robustness.
- Perform structured hyperparameter optimization and compare additional regularization methods.
