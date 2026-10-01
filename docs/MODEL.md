# Model Documentation

## Overview

This project uses **DenseNet121** as the backbone architecture for multi-label chest X-ray disease classification.

DenseNet121 is a convolutional neural network introduced in the paper **"Densely Connected Convolutional Networks"** by Huang et al. It is widely used in medical image analysis because of its efficient feature reuse, strong gradient propagation, and competitive performance on image classification tasks.

For this project, a pretrained DenseNet121 model was fine-tuned using transfer learning to classify five thoracic diseases from frontal chest X-ray images.

---

# Why DenseNet121?

Several convolutional neural network architectures are commonly used for medical imaging, including ResNet, EfficientNet, and Vision Transformers. DenseNet121 was selected for this project because it provides a strong balance between accuracy, computational efficiency, and training stability.

Key advantages include:

* Efficient feature reuse through dense connections.
* Improved gradient flow during training.
* Fewer parameters than many comparable architectures.
* Strong performance on medical imaging benchmarks.
* Well-supported within the PyTorch ecosystem.

These characteristics make DenseNet121 a reliable baseline for chest X-ray disease classification.

---

# Transfer Learning

Training a deep convolutional neural network entirely from scratch requires a very large amount of labeled data and computational resources.

Instead, this project uses **transfer learning**, where DenseNet121 is initialized with weights pretrained on ImageNet before being fine-tuned for the chest X-ray classification task.

Transfer learning enables:

* Faster convergence.
* Better feature extraction.
* Reduced training time.
* Improved performance on limited medical datasets.

---

# Model Architecture

The pretrained DenseNet121 backbone is retained for feature extraction.

The original ImageNet classification head is replaced with a custom fully connected layer containing **five output neurons**, corresponding to the selected thoracic diseases.

Each output neuron independently predicts the probability of a disease being present in the input image.

This design enables **multi-label classification**, allowing multiple diseases to be predicted simultaneously.

---

# Input Pipeline

The model receives chest X-ray images after preprocessing.

Input specifications:

| Property       | Value            |
| -------------- | ---------------- |
| Image Size     | 224 × 224 pixels |
| Color Channels | 3 (RGB)          |
| Input Format   | Tensor           |
| Batch Size     | 32               |

Image preprocessing and transformations are performed before training and validation.

---

# Loss Function

This project uses **Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss)**.

Unlike standard cross-entropy loss, BCEWithLogitsLoss treats each disease prediction independently, making it well suited for multi-label classification problems.

Each disease is assigned an independent probability between 0 and 1.

---

# Optimizer

Training is performed using the **Adam** optimizer.

Adam combines adaptive learning rates with momentum, providing efficient optimization and stable convergence during fine-tuning.

---

# Output Layer

The final classification layer contains five neurons representing:

* Atelectasis
* Cardiomegaly
* Consolidation
* Edema
* Pleural Effusion

Each neuron outputs an independent prediction score for its corresponding disease.

---

# Training Summary

| Parameter           | Value             |
| ------------------- | ----------------- |
| Framework           | PyTorch 2.11.0    |
| Architecture        | DenseNet121       |
| Input Size          | 224 × 224         |
| Classification Type | Multi-label       |
| Loss Function       | BCEWithLogitsLoss |
| Optimizer           | Adam              |
| Batch Size          | 32                |
| Epochs              | 10                |
| Training Time       | ~4.5 Hours        |

---

# Model Strengths

The selected architecture offers several advantages:

* Efficient transfer learning.
* Strong baseline performance.
* Good computational efficiency.
* Suitable for medical imaging tasks.
* Well-established architecture with extensive community support.

---

# Model Limitations

Although DenseNet121 performs well, several limitations should be considered:

* Performance depends on dataset quality and label distribution.
* Predictions are sensitive to the selected decision threshold.
* The model should not be interpreted as a clinical diagnostic tool.
* Additional architectures may achieve improved performance under different training settings.

---

# Future Improvements

Potential future model improvements include:

* Compare with ResNet, EfficientNet, and Vision Transformers.
* Hyperparameter optimization.
* Ensemble learning.
* Model calibration.
* Higher-resolution input images.
* Additional explainability techniques beyond Grad-CAM.

---

# Summary

DenseNet121 provides a strong and efficient baseline for multi-label chest X-ray disease classification. Combined with transfer learning, comprehensive evaluation, and Grad-CAM explainability, the model forms the core of this Medical AI project and demonstrates a practical end-to-end deep learning workflow using PyTorch.
