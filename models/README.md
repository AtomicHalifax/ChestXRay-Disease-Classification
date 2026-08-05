# Pre-trained Model

This directory contains the pre-trained model weights for the Chest X-Ray Disease Classification project.

## Available Model

**best_densenet121.pth**

### Model Details

* Architecture: DenseNet121
* Framework: PyTorch
* Task: Multi-label Chest X-ray Disease Classification
* Target Diseases: 5
* Training Epochs: 10
* Input Resolution: 224 × 224
* Loss Function: BCEWithLogitsLoss
* Optimizer: Adam

### Performance

| Metric     |      Value |
| ---------- | ---------: |
| Mean AUROC | **0.8790** |

The model corresponds to the best validation checkpoint obtained during training and is provided for inference, evaluation, and reproducibility.

> **Note:** This model is intended for research, educational, and portfolio purposes only. It is **not** approved for clinical diagnosis or medical decision-making.
