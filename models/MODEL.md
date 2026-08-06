# Pre-trained Model

## Overview

The pre-trained DenseNet121 model for this project is hosted on **Hugging Face Hub** due to GitHub's file size limitations.

## Model Repository

**Hugging Face**

https://huggingface.co/AtomicHalifax/ChestXRay-DenseNet121

## Model Information

| Property        | Value                                          |
| --------------- | ---------------------------------------------- |
| Architecture    | DenseNet121                                    |
| Framework       | PyTorch                                        |
| Task            | Multi-label Chest X-ray Disease Classification |
| Dataset         | Stanford CheXpert                              |
| Target Diseases | 5                                              |
| Training Epochs | 10                                             |
| Training Time   | ~4.5 Hours                                     |
| Mean AUROC      | **0.8790**                                     |

## Usage

Download the model weights from Hugging Face and place them inside the `models/` directory before running inference.

```text
models/
└── best_densenet121.pth
```

The inference notebook automatically loads the model from this location.

## Disclaimer

The model is provided for **research, educational, and portfolio purposes only**. It is **not intended for clinical diagnosis or medical decision-making**.
