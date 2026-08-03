# System Architecture

## Overview

This document describes the overall architecture of the Chest X-Ray Disease Classification project, from dataset preparation to model inference and explainability.

The project follows a modular deep learning workflow designed to be reproducible, interpretable, and easy to understand.

---

# High-Level Architecture

```text
Stanford CheXpert Dataset
            │
            ▼
 Dataset Preprocessing
            │
            ▼
 Image Transformations
            │
            ▼
 PyTorch Dataset
            │
            ▼
 DataLoader
            │
            ▼
 DenseNet121
            │
            ▼
 Training
            │
            ▼
 Validation
            │
            ▼
 Performance Evaluation
            │
            ▼
 Grad-CAM Explainability
```

---

# Data Pipeline

The project begins with the Stanford CheXpert dataset.

The preprocessing pipeline performs:

* Dataset loading
* Label extraction
* Image resizing
* Tensor conversion
* Normalization
* Batch creation using DataLoader

The processed batches are then passed to the neural network during training and validation.

---

# Model Pipeline

The learning pipeline consists of:

1. Load pretrained DenseNet121.
2. Replace the original classification layer.
3. Configure the optimizer and loss function.
4. Train using transfer learning.
5. Validate after each epoch.
6. Save the best-performing checkpoint.

---

# Training Workflow

The model is trained using the following sequence:

```text
Training Images
      │
      ▼
Forward Pass
      │
      ▼
Prediction
      │
      ▼
Loss Calculation
      │
      ▼
Backpropagation
      │
      ▼
Optimizer Update
      │
      ▼
Next Batch
```

This process repeats until all batches in an epoch have been processed.

---

# Validation Workflow

After each epoch, the model is evaluated on the validation dataset.

The validation pipeline performs:

* Forward inference
* Probability prediction
* Thresholding
* Metric computation
* AUROC calculation
* Best model selection

No gradient updates are performed during validation.

---

# Evaluation Pipeline

The trained model is evaluated using multiple complementary metrics.

```text
Predicted Probabilities
          │
          ▼
 Decision Threshold
          │
          ▼
 Binary Predictions
          │
          ▼
Performance Metrics
```

The evaluation includes:

* AUROC
* Precision
* Recall
* F1-Score
* Accuracy
* ROC Curves
* Precision–Recall Curves
* Confusion Matrices

---

# Explainability Pipeline

Model interpretability is achieved using Grad-CAM.

```text
Input Image
      │
      ▼
DenseNet121
      │
      ▼
Feature Maps
      │
      ▼
Gradients
      │
      ▼
Grad-CAM Heatmap
      │
      ▼
Overlay on Chest X-ray
```

The resulting heatmaps provide visual explanations of the regions that influenced each prediction.

---

# End-to-End Workflow

The complete project workflow can be summarized as follows:

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Training
   │
   ▼
Validation
   │
   ▼
Evaluation
   │
   ▼
Visualization
   │
   ▼
Explainability
```

This modular workflow separates each stage of the project, making the implementation easier to understand, reproduce, and extend.

---

# Design Principles

The project architecture was designed with the following goals:

* Reproducibility
* Simplicity
* Modularity
* Explainability
* Readability
* Extensibility

Each stage of the workflow performs a clearly defined task while remaining independent of the others.

---

# Summary

The architecture combines data preprocessing, transfer learning, systematic evaluation, and explainable AI into a unified Computer Vision workflow.

This modular design makes the project suitable for learning, experimentation, and future extensions while maintaining a clean and reproducible implementation.
