# 🩻 Chest X-Ray Disease Classification using DenseNet121

> **A deep learning Computer Vision and Medical AI project for multi-label chest X-ray disease classification using the Stanford CheXpert dataset with Grad-CAM explainability and comprehensive evaluation.**


![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![DenseNet121](https://img.shields.io/badge/Model-DenseNet121-blue?style=for-the-badge)
![Medical AI](https://img.shields.io/badge/Domain-Medical_AI-success?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer-Vision-purple?style=for-the-badge)
![Grad-CAM](https://img.shields.io/badge/Explainability-Grad--CAM-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/1c646a7c-247d-4e5e-910a-8df9b88fa4a8" />


- 📘 [Target Diseases](docs/DISEASES.md) — Clinical overview of the five thoracic diseases and model performance.
---

# 📖 Overview

Chest X-rays are among the most frequently used medical imaging techniques for diagnosing thoracic diseases. However, interpreting large volumes of radiographs requires significant clinical expertise and can be time-intensive.

This project explores **multi-label chest X-ray disease classification** using **DenseNet121** trained on the **Stanford CheXpert** dataset. The model predicts five clinically relevant thoracic diseases while emphasizing not only predictive performance but also model interpretability through **Grad-CAM** visualizations.

The repository demonstrates an end-to-end deep learning workflow, including data preprocessing, transfer learning, training, evaluation, visualization, and explainability. It is designed as a portfolio-quality Computer Vision project showcasing practical Medical AI development with PyTorch.

---

# 🚀 Project Highlights

| Category               | Details                                        |
| ---------------------- | ---------------------------------------------- |
| **Domain**             | Medical AI / Computer Vision                   |
| **Task**               | Multi-label Chest X-ray Disease Classification |
| **Dataset**            | Stanford CheXpert (~11.5 GB)                   |
| **Training Images**    | 223,414                                        |
| **Validation Images**  | 234                                            |
| **Target Diseases**    | 5                                              |
| **Model Architecture** | DenseNet121                                    |
| **Framework**          | PyTorch                                        |
| **Training Time**      | ~4.5 Hours                                     |
| **Training Epochs**    | 10                                             |
| **Mean AUROC**         | **0.8790**                                     |
| **Explainability**     | Grad-CAM                                       |

---

# ✨ Why You'll Find This Repository Useful

Whether you're learning deep learning or building Medical AI applications, this repository provides a practical reference for modern Computer Vision workflows.

It includes:

* End-to-end multi-label image classification using PyTorch
* Transfer learning with DenseNet121
* Training on a real-world medical imaging dataset
* Comprehensive model evaluation using multiple metrics
* AUROC, ROC Curves, and Precision–Recall Curves
* Confusion matrix analysis
* Grad-CAM explainability
* Sample prediction visualizations
* Clean and well-documented implementation suitable for learning and portfolio development

---

# 📚 Table of Contents

* Overview
* Project Highlights
* Why This Project?
* Dataset Overview
* Target Diseases
* Model Architecture
* Training Configuration
* Repository Structure
* Installation
* Dataset Download
* Results
* Evaluation Metrics
* Explainability (Grad-CAM)
* Sample Predictions
* Future Work
* References
* License
* Citation

# 🎯 Why This Project?

Medical imaging is one of the most impactful applications of artificial intelligence, with chest X-rays serving as one of the most widely used diagnostic tools for detecting thoracic diseases. Developing reliable computer vision models for chest X-ray interpretation has the potential to support clinicians by improving workflow efficiency and assisting with early disease detection.

This project was built to explore the practical application of deep learning for multi-label chest X-ray disease classification using the Stanford CheXpert dataset. Rather than focusing solely on predictive performance, the project emphasizes reproducibility, comprehensive evaluation, and explainability through Grad-CAM visualizations.

The repository demonstrates an end-to-end Computer Vision workflow using PyTorch, making it a practical learning resource and a portfolio-quality Medical AI project.

---

# 📂 Dataset Overview

This project uses the **Stanford CheXpert** dataset, one of the largest publicly available chest X-ray datasets designed for multi-label thoracic disease classification.

| Property                | Value                |
| ----------------------- | -------------------- |
| **Dataset**             | Stanford CheXpert    |
| **Dataset Size**        | ~11.5 GB             |
| **Training Images**     | 223,414              |
| **Validation Images**   | 234                  |
| **Classification Type** | Multi-label          |
| **Image Modality**      | Frontal Chest X-rays |

### Dataset Sources

* **Original Stanford Dataset:** https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
* **Kaggle Mirror:** https://www.kaggle.com/datasets/ashery/chexpert

> **Note:** The dataset is **not included** in this repository because of its size (~11.5 GB) and the licensing terms of the original dataset.

---

# 🫁 Target Diseases

The model predicts five clinically relevant thoracic diseases.

| Disease              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **Atelectasis**      | Partial collapse of lung tissue that reduces airflow.                                       |
| **Cardiomegaly**     | Enlargement of the heart visible on chest radiographs.                                      |
| **Consolidation**    | Lung tissue filled with inflammatory material or fluid, commonly associated with infection. |
| **Edema**            | Excess fluid accumulation within the lungs.                                                 |
| **Pleural Effusion** | Fluid buildup between the lungs and the chest wall.                                         |

---

# 🧠 Why These Five Diseases?

The CheXpert dataset contains multiple thoracic findings with varying frequencies and uncertainty labels. To create a focused and reproducible benchmark, this project concentrates on five clinically relevant diseases that provide a balanced evaluation of multi-label classification while keeping the implementation computationally practical.

These diseases represent a diverse set of thoracic abnormalities and allow the project to demonstrate transfer learning, multi-label prediction, and comprehensive evaluation within a manageable scope.

---

# 🔄 Workflow Overview

The overall pipeline implemented in this project is shown below.

```text
Stanford CheXpert Dataset
            │
            ▼
Data Preprocessing
            │
            ▼
Image Transformations
            │
            ▼
DenseNet121 (Transfer Learning)
            │
            ▼
Model Training
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

*A visual workflow diagram is included in the project presentation and documentation.*

---

# 📖 Learning Outcomes

This repository demonstrates practical concepts in modern Computer Vision and Medical AI, including:

* Multi-label image classification
* Transfer learning with DenseNet121
* Medical image preprocessing
* Training deep learning models using PyTorch
* AUROC-based model evaluation
* ROC and Precision–Recall curve analysis
* Confusion matrix interpretation
* Threshold selection and performance trade-offs
* Grad-CAM explainability
* End-to-end Computer Vision project development

# 🏗️ Model Architecture

This project uses **DenseNet121**, a convolutional neural network architecture that introduces dense connectivity between layers to encourage feature reuse and improve gradient flow. Rather than learning redundant representations, each layer receives feature maps from all preceding layers, making DenseNet both parameter-efficient and well-suited for transfer learning tasks.

A pretrained DenseNet121 backbone was fine-tuned for multi-label chest X-ray disease classification by replacing the original classification head with a custom output layer containing five neurons, corresponding to the selected thoracic diseases.

---

# ⚙️ Training Configuration

The model was trained using transfer learning with PyTorch. Binary Cross-Entropy with Logits was used as the loss function to support independent prediction of multiple diseases within a single chest X-ray image.

| Parameter               | Value                     |
| ----------------------- | ------------------------- |
| **Framework**           | PyTorch 2.11.0            |
| **Model**               | DenseNet121               |
| **Input Image Size**    | 224 × 224                 |
| **Classification Type** | Multi-label               |
| **Loss Function**       | BCEWithLogitsLoss         |
| **Optimizer**           | Adam                      |
| **Batch Size**          | 32                        |
| **Training Epochs**     | 10                        |
| **Training Time**       | ~4.5 Hours                |
| **Device**              | NVIDIA GPU (Google Colab) |

---

# 🔄 Training Pipeline

The complete training workflow consists of the following stages:

1. Download and prepare the Stanford CheXpert dataset.
2. Apply image preprocessing and resizing.
3. Perform data transformations suitable for transfer learning.
4. Initialize the pretrained DenseNet121 model.
5. Replace the classification head for five disease predictions.
6. Train using Binary Cross-Entropy with Logits loss.
7. Validate the model after each epoch.
8. Save the best-performing model checkpoint.
9. Perform comprehensive evaluation on the validation set.
10. Generate Grad-CAM visualizations for model interpretability.

---

# 📈 Evaluation Methodology

Model performance was assessed using multiple complementary metrics to provide a balanced evaluation of classification quality.

The evaluation pipeline includes:

* Area Under the Receiver Operating Characteristic Curve (AUROC)
* Precision
* Recall
* F1-Score
* Accuracy
* Receiver Operating Characteristic (ROC) Curves
* Precision–Recall Curves
* Confusion Matrices
* Threshold Analysis
* Error Analysis
* Grad-CAM Explainability

Among these metrics, **Mean AUROC** serves as the primary evaluation metric because it provides a threshold-independent measure of the model's ability to distinguish between positive and negative cases across all target diseases.

---

# 💡 Explainable AI

Interpretability plays a critical role in Medical AI applications. To better understand the model's predictions, this project incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM highlights the image regions that contribute most strongly to each prediction, enabling visual inspection of the model's attention and providing additional confidence in its decision-making process.

These visualizations complement the quantitative evaluation metrics and help identify both successful predictions and challenging failure cases.

---

# 🎯 Reproducibility

The project has been designed with reproducibility in mind. The repository includes:

* Complete training notebook
* Dependency requirements
* Dataset download instructions
* Training configuration
* Evaluation pipeline
* Performance metrics
* Visualizations and explainability examples

This enables users to reproduce the workflow, experiment with alternative architectures, or extend the project for additional research and learning purposes.

# 📊 Results

The DenseNet121 model was trained for **10 epochs** on the Stanford CheXpert dataset using transfer learning. Performance was evaluated on the validation set using multiple classification metrics, with **Mean AUROC** serving as the primary evaluation metric.

## 🏆 Performance Summary

| Metric                |       Value |
| --------------------- | ----------: |
| **Model**             | DenseNet121 |
| **Training Images**   |     223,414 |
| **Validation Images** |         234 |
| **Target Diseases**   |           5 |
| **Training Time**     |  ~4.5 Hours |
| **Training Epochs**   |          10 |
| **Batch Size**        |          32 |
| **Mean AUROC**        |  **0.8790** |

---

## 📈 Disease-wise Performance

| Disease          | Precision | Recall | F1-Score | Accuracy |      AUROC |
| ---------------- | --------: | -----: | -------: | -------: | ---------: |
| Atelectasis      |    0.6170 | 0.7250 |   0.6667 |   0.7521 | **0.8176** |
| Cardiomegaly     |    0.7407 | 0.5882 |   0.6557 |   0.8205 | **0.8547** |
| Consolidation    |    0.4333 | 0.7879 |   0.5591 |   0.8248 | **0.8973** |
| Edema            |    0.5065 | 0.8667 |   0.6393 |   0.8120 | **0.9061** |
| Pleural Effusion |    0.8033 | 0.7313 |   0.7656 |   0.8718 | **0.9194** |

---

# 📉 Evaluation Visualizations

The repository includes a comprehensive evaluation pipeline with multiple visualizations to analyze model performance from different perspectives.

### Included Visualizations

* ROC Curves
* AUROC Comparison
* Precision–Recall Curves
* Confusion Matrices
* Sample Predictions
* Misclassified Image Analysis
* Grad-CAM Explainability

> Figures are available in the `images/` directory and are discussed throughout the accompanying documentation.

---

# 🔥 Grad-CAM Explainability

To improve transparency, Grad-CAM is used to visualize the image regions that contribute most strongly to each prediction.

These visual explanations help verify that the model focuses on clinically relevant regions of the chest radiograph instead of relying on unrelated image features.

Grad-CAM serves as an additional validation tool alongside quantitative evaluation metrics.

---

# ⚠️ Error Analysis

Understanding model failures is as important as measuring successful predictions.

This project includes an error analysis workflow that examines:

* False Positive predictions
* False Negative predictions
* Misclassified chest X-rays
* Disease-specific performance differences
* Threshold-dependent prediction behavior

Analyzing these cases provides insight into the strengths and limitations of the baseline DenseNet121 model and highlights opportunities for future improvement.

---

# 🎯 Key Findings

* Mean AUROC of **0.8790** across five thoracic diseases.
* Strongest performance achieved for **Pleural Effusion (0.9194 AUROC)**.
* All five diseases achieved an AUROC above **0.81**.
* Grad-CAM visualizations improved model interpretability.
* Comprehensive evaluation demonstrates the effectiveness of DenseNet121 as a strong baseline for multi-label chest X-ray disease classification.

# 🚀 Getting Started

## Prerequisites

Before running the notebook, ensure the following are installed:

* Python 3.11+
* PyTorch 2.11.0
* Torchvision 0.26.0
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* OpenCV
* Pillow
* tqdm
* grad-cam

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Download

This project uses the **Stanford CheXpert** dataset.

The dataset is **not included** in this repository because of its size (~11.5 GB) and licensing requirements.

Please download the dataset from one of the following official sources:

* **Stanford CheXpert:** https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
* **Kaggle Mirror:** https://www.kaggle.com/datasets/ashery/chexpert

After downloading, update the dataset paths inside the notebook before training or evaluation.

---

# 📽️ Project Resources

This repository includes additional resources to complement the notebook and documentation.

### Presentation

A presentation summarizing the project, methodology, model architecture, evaluation, and key findings.

### Technical Documentation

Detailed documentation covering:

* Project motivation
* Dataset
* Model architecture
* Training methodology
* Evaluation strategy
* Experimental results

### Project Poster

A one-page visual summary highlighting the motivation, workflow, architecture, performance, and conclusions.

---

# 🎯 Project Scope

This repository is intended as a **Computer Vision and Medical AI portfolio project**.

The objective is to demonstrate an end-to-end deep learning workflow for multi-label chest X-ray disease classification, including dataset preparation, transfer learning, evaluation, and explainability.

Cloud deployment and MLOps are intentionally outside the scope of this project and will be explored in future repositories.

---

# 🔮 Future Improvements

Potential directions for extending this project include:

* Train on the complete CheXpert label set.
* Evaluate Vision Transformer architectures.
* Hyperparameter optimization.
* Cross-validation experiments.
* External dataset evaluation.
* Model calibration.
* Lightweight inference optimization.
* Comparison with additional CNN backbones.
* Clinical deployment considerations.

---

# 📚 References

* Stanford CheXpert Dataset
* CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison
* Densely Connected Convolutional Networks (DenseNet)
* Grad-CAM: Visual Explanations from Deep Networks
* PyTorch Documentation
* Torchvision Documentation

---

# 📄 License

This project is released under the **MIT License**.

---

# 🙏 Acknowledgements

Special thanks to the **Stanford Center for Artificial Intelligence in Medicine & Imaging (AIMI)** for making the CheXpert dataset publicly available.

Additional thanks to the **PyTorch** open-source community for providing the tools and libraries used throughout this project.

---

# ⭐ Support

If you found this repository useful:

* ⭐ Star the repository
* 🍴 Fork the project
* 💡 Share feedback or suggestions
* 🤝 Connect with me on LinkedIn

Contributions, discussions, and constructive feedback are always welcome.

