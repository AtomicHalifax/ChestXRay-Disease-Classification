# Project Limitations

## Overview

While the DenseNet121 model achieved strong performance on the selected thoracic diseases, this project has several limitations that should be considered when interpreting the results.

The project is intended as a **Computer Vision and Medical AI portfolio project** and should not be used for clinical decision-making.

---

# Dataset Scope

The model was trained using the **Stanford CheXpert** dataset.

Although CheXpert contains labels for multiple thoracic observations, this project focuses on only **five target diseases**:

* Atelectasis
* Cardiomegaly
* Consolidation
* Edema
* Pleural Effusion

The model does not predict the remaining CheXpert labels.

---

# Validation Dataset

Performance was evaluated using the provided validation dataset.

The reported metrics reflect performance on this dataset only and should not be interpreted as a measure of general clinical performance across different hospitals, imaging devices, or patient populations.

---

# Baseline Architecture

This project uses **DenseNet121** as a strong transfer learning baseline.

Other modern architectures, such as ConvNeXt, EfficientNet, Vision Transformers (ViT), or Swin Transformers, were not evaluated and may achieve different performance characteristics.

---

# Image Resolution

All chest X-ray images were resized to **224 × 224** pixels before training and inference.

While this resolution is commonly used for transfer learning, resizing may reduce fine anatomical details that could be useful for detecting subtle abnormalities.

---

# Multi-label Classification Challenges

Chest X-rays may contain multiple coexisting diseases.

The complexity of overlapping thoracic abnormalities can make multi-label classification more challenging than single-label classification and may influence model performance.

---

# Threshold Selection

Binary predictions are generated using a fixed decision threshold.

Different thresholds may improve precision or recall depending on the intended application, and no threshold optimization was performed as part of this project.

---

# External Validation

The model was not evaluated on external datasets.

Performance on chest X-rays acquired from different institutions or imaging protocols may differ from the results reported in this repository.

---

# Clinical Use

This project is designed for:

* Learning
* Research
* Portfolio demonstration
* Computer Vision experimentation

It is **not** intended for:

* Clinical diagnosis
* Medical treatment
* Patient care
* Emergency decision-making

Any medical decisions should always be made by qualified healthcare professionals.

---

# Future Improvements

Potential improvements include:

* Training on additional thoracic disease labels.
* Evaluating more recent deep learning architectures.
* Hyperparameter optimization.
* External validation on independent datasets.
* Higher input image resolutions.
* Threshold optimization for each disease.
* Calibration of prediction probabilities.
* Prospective evaluation in real-world clinical settings.

---

# Summary

Despite these limitations, the project demonstrates a complete and reproducible deep learning workflow for multi-label chest X-ray disease classification.

The implementation provides a strong DenseNet121 baseline with comprehensive evaluation and Grad-CAM explainability, serving as a foundation for future experimentation and extension.
