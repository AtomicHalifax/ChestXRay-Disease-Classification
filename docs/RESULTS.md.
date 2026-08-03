# Results Documentation

## Overview

This document summarizes the performance of the DenseNet121 model on the selected five thoracic diseases using the Stanford CheXpert dataset.

The objective of this project was to establish a strong baseline for multi-label chest X-ray disease classification while providing comprehensive evaluation and model interpretability.

---

# Experiment Summary

| Property            | Value             |
| ------------------- | ----------------- |
| Model               | DenseNet121       |
| Framework           | PyTorch           |
| Dataset             | Stanford CheXpert |
| Training Images     | 223,414           |
| Validation Images   | 234               |
| Classification Type | Multi-label       |
| Training Epochs     | 10                |
| Training Time       | ~4.5 Hours        |
| Primary Metric      | Mean AUROC        |

---

# Overall Performance

## Mean AUROC

**0.8790**

The DenseNet121 baseline achieved a **Mean AUROC of 0.8790**, demonstrating strong performance across the selected thoracic diseases and establishing a reliable baseline for future experimentation.

---

# Disease-wise Results

| Disease          | Precision | Recall | F1-Score | Accuracy |      AUROC |
| ---------------- | --------: | -----: | -------: | -------: | ---------: |
| Atelectasis      |    0.6170 | 0.7250 |   0.6667 |   0.7521 | **0.8176** |
| Cardiomegaly     |    0.7407 | 0.5882 |   0.6557 |   0.8205 | **0.8547** |
| Consolidation    |    0.4333 | 0.7879 |   0.5591 |   0.8248 | **0.8973** |
| Edema            |    0.5065 | 0.8667 |   0.6393 |   0.8120 | **0.9061** |
| Pleural Effusion |    0.8033 | 0.7313 |   0.7656 |   0.8718 | **0.9194** |

---

# Performance Highlights

Several notable observations emerged from the evaluation:

* Mean AUROC reached **0.8790** across all five target diseases.
* **Pleural Effusion** achieved the highest AUROC (**0.9194**).
* **Edema** also demonstrated excellent performance with an AUROC above **0.90**.
* All five diseases achieved AUROC values greater than **0.81**.
* The DenseNet121 baseline consistently performed well despite the complexity of the multi-label classification task.

---

# Visual Evaluation

To provide a comprehensive understanding of model performance, the repository includes multiple visual analyses.

Available visualizations include:

* AUROC Comparison Chart
* ROC Curves
* Precision–Recall Curves
* Confusion Matrices
* Sample Predictions
* Misclassified Images
* Grad-CAM Visualizations

These visualizations complement the numerical metrics and provide additional insight into the model's behavior.

---

# Grad-CAM Findings

Grad-CAM visualizations indicate that the model generally focuses on clinically relevant regions of the chest radiographs when making predictions.

These activation maps improve interpretability by highlighting image regions that contribute most strongly to the model's decisions and provide additional confidence in the evaluation process.

---

# Error Analysis Summary

Although the DenseNet121 baseline achieved strong overall performance, several challenges remain:

* Some diseases exhibit lower precision than others.
* Performance varies depending on disease prevalence.
* Prediction confidence is influenced by the selected decision threshold.
* Certain misclassified cases remain visually challenging even for deep learning models.

These findings identify opportunities for future improvements and further experimentation.

---

# Key Takeaways

The experimental results demonstrate that:

* DenseNet121 provides a strong baseline for multi-label chest X-ray disease classification.
* Transfer learning is effective for medical imaging tasks.
* AUROC offers a reliable threshold-independent evaluation metric.
* Grad-CAM enhances transparency by improving model interpretability.
* Comprehensive evaluation provides a deeper understanding of model strengths and limitations.

---

# Conclusion

The DenseNet121 baseline achieved competitive performance on the selected thoracic diseases while maintaining a reproducible and interpretable workflow.

Combined with extensive evaluation, visualization, and explainability, the results establish a solid foundation for future research, experimentation, and extension of this Medical AI project.
