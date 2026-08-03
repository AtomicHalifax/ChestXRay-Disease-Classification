# Training Documentation

## Overview

This document describes the training methodology used for the Chest X-Ray Disease Classification project.

The model was trained using transfer learning with DenseNet121 on the Stanford CheXpert dataset. The training pipeline was designed to be reproducible, efficient, and easy to understand while following modern deep learning practices.

---

# Training Environment

The project was developed using the following software stack.

| Component    | Version |
| ------------ | ------- |
| Python       | 3.11+   |
| PyTorch      | 2.11.0  |
| Torchvision  | 0.26.0  |
| NumPy        | 2.0.2   |
| Pandas       | 2.2.2   |
| Scikit-learn | 1.6.1   |
| Matplotlib   | 3.10.0  |

Training was performed using **Google Colab** with GPU acceleration.

---

# Dataset Preparation

The Stanford CheXpert dataset was downloaded separately and is **not included** in this repository.

Before training:

* Dataset paths were configured.
* Images were loaded from the training directory.
* Labels were extracted from the metadata CSV files.
* Missing values were handled during preprocessing.
* Five target diseases were selected for multi-label classification.

---

# Image Preprocessing

Each chest X-ray image underwent preprocessing before being passed to the model.

The preprocessing pipeline included:

* Image loading
* Resizing to **224 × 224**
* Conversion to tensor format
* Normalization using ImageNet statistics

These transformations ensure compatibility with the pretrained DenseNet121 backbone.

---

# Training Configuration

| Parameter     | Value             |
| ------------- | ----------------- |
| Architecture  | DenseNet121       |
| Framework     | PyTorch           |
| Image Size    | 224 × 224         |
| Batch Size    | 32                |
| Optimizer     | Adam              |
| Loss Function | BCEWithLogitsLoss |
| Epochs        | 10                |
| Training Time | ~4.5 Hours        |

---

# Training Workflow

The overall training process follows these steps:

1. Load the CheXpert dataset.
2. Apply preprocessing and image transformations.
3. Create PyTorch DataLoaders.
4. Initialize the pretrained DenseNet121 model.
5. Replace the classification head for five disease outputs.
6. Configure the optimizer and loss function.
7. Train the model for multiple epochs.
8. Validate after each epoch.
9. Compute validation metrics.
10. Save the best-performing model checkpoint.

---

# Model Checkpointing

To preserve the best-performing model, checkpointing was performed during training.

The checkpoint with the highest validation performance was saved as:

```text
best_densenet121.pth
```

This prevents losing the best model if later epochs do not improve performance.

---

# Validation Strategy

Model performance was evaluated after every training epoch using the validation dataset.

The validation process included:

* Forward inference
* Probability prediction
* Metric calculation
* AUROC computation
* Loss monitoring

The best-performing checkpoint was selected based on validation performance.

---

# Performance Monitoring

Throughout training, the following values were monitored:

* Training Loss
* Validation Loss
* Disease-wise AUROC
* Mean AUROC

Tracking these metrics made it possible to monitor convergence and compare model performance across epochs.

---

# Training Outcome

After **10 training epochs**, the DenseNet121 baseline achieved a **Mean AUROC of 0.8790** across the five selected thoracic diseases.

The trained model was then evaluated using additional metrics including Precision, Recall, F1-score, Accuracy, ROC Curves, Precision–Recall Curves, Confusion Matrices, and Grad-CAM visualizations.

---

# Reproducibility

The training workflow has been designed to be reproducible.

To reproduce the results:

1. Download the CheXpert dataset.
2. Install the project dependencies.
3. Update the dataset paths in the notebook.
4. Execute the notebook from top to bottom.

This reproduces the complete training and evaluation pipeline described throughout the repository.

---

# Summary

The training pipeline combines transfer learning, efficient preprocessing, systematic validation, and automatic checkpointing to establish a strong baseline for multi-label chest X-ray disease classification. The resulting model demonstrates competitive performance while remaining easy to reproduce and extend for future experimentation.
