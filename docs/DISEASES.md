# Target Diseases

## Overview

The DenseNet121 model in this project is trained to perform **multi-label classification** of five clinically important thoracic diseases using chest X-ray images from the Stanford CheXpert dataset.

Unlike single-label classification, a patient may present with **multiple diseases simultaneously**, allowing the model to predict more than one condition for a single chest X-ray.

The selected diseases are among the most commonly encountered thoracic abnormalities in clinical practice and represent a diverse set of pulmonary and cardiac conditions.

---

# Disease Summary

| Disease          | Description                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| Atelectasis      | Partial or complete collapse of lung tissue, reducing the amount of air in the affected region.               |
| Cardiomegaly     | Enlargement of the heart, often associated with underlying cardiovascular disease.                            |
| Consolidation    | Filling of the lung air spaces with fluid, inflammatory cells, or other material, commonly seen in pneumonia. |
| Edema            | Accumulation of excess fluid within the lung tissue, frequently associated with heart failure.                |
| Pleural Effusion | Collection of excess fluid in the pleural space surrounding the lungs.                                        |

---

# 1. Atelectasis

### Overview

Atelectasis refers to the partial or complete collapse of lung tissue, leading to reduced air volume in part of the lung.

### Clinical Importance

* Reduces oxygen exchange.
* Common after surgery.
* May result from airway obstruction or infection.
* Early detection helps prevent respiratory complications.

### Typical Chest X-ray Findings

* Increased opacity.
* Loss of lung volume.
* Displacement of nearby anatomical structures.

### Model Performance

| Metric    |      Value |
| --------- | ---------: |
| Precision | **0.6170** |
| Recall    | **0.7250** |
| F1-Score  | **0.6667** |
| Accuracy  | **0.7521** |
| AUROC     | **0.8176** |

<img width="2000" height="1646" alt="image" src="https://github.com/user-attachments/assets/471b051a-6b43-46d2-9f04-d948523b1973" />

---

# 2. Cardiomegaly

### Overview

Cardiomegaly is an enlargement of the heart and often indicates underlying cardiovascular disease rather than being a disease itself.

### Clinical Importance

* May indicate heart failure.
* Can be associated with hypertension.
* May reflect structural heart disease.

### Typical Chest X-ray Findings

* Enlarged cardiac silhouette.
* Increased cardiothoracic ratio.

### Model Performance

| Metric    |      Value |
| --------- | ---------: |
| Precision | **0.7407** |
| Recall    | **0.5882** |
| F1-Score  | **0.6557** |
| Accuracy  | **0.8205** |
| AUROC     | **0.8547** |

<img width="926" height="483" alt="image" src="https://github.com/user-attachments/assets/0626e445-5eb3-480d-8064-e66902245150" />

---

# 3. Consolidation

### Overview

Consolidation occurs when the normal air-filled spaces of the lungs become filled with fluid, inflammatory cells, or other material.

### Clinical Importance

* Frequently associated with pneumonia.
* May result from infection or inflammation.
* Important indicator of pulmonary disease.

### Typical Chest X-ray Findings

* Dense white regions within the lungs.
* Localized areas of increased opacity.

### Model Performance

| Metric    |      Value |
| --------- | ---------: |
| Precision | **0.4333** |
| Recall    | **0.7879** |
| F1-Score  | **0.5591** |
| Accuracy  | **0.8248** |
| AUROC     | **0.8973** |

<img width="2000" height="2022" alt="image" src="https://github.com/user-attachments/assets/edb6eb59-cbd6-4e65-a529-542c5909e1fb" />

---

# 4. Edema

### Overview

Pulmonary edema is the accumulation of excess fluid within the lungs, making breathing difficult.

### Clinical Importance

* Often caused by heart failure.
* Can become life-threatening if untreated.
* Requires prompt medical attention.

### Typical Chest X-ray Findings

* Diffuse bilateral opacities.
* Fluid accumulation within lung tissue.

### Model Performance

| Metric    |      Value |
| --------- | ---------: |
| Precision | **0.5065** |
| Recall    | **0.8667** |
| F1-Score  | **0.6393** |
| Accuracy  | **0.8120** |
| AUROC     | **0.9061** |

<img width="991" height="758" alt="image" src="https://github.com/user-attachments/assets/87b4cffd-bca9-4ca3-a60f-b61788a56bb3" />

---

# 5. Pleural Effusion

### Overview

Pleural effusion is the accumulation of excess fluid in the pleural space surrounding the lungs.

### Clinical Importance

* May occur due to infection, heart failure, or malignancy.
* Can compress lung tissue and impair breathing.

### Typical Chest X-ray Findings

* Blunting of the costophrenic angle.
* Fluid layering at the lung base.
* Increased opacity in the lower chest.

### Model Performance

| Metric    |      Value |
| --------- | ---------: |
| Precision | **0.8033** |
| Recall    | **0.7313** |
| F1-Score  | **0.7656** |
| Accuracy  | **0.8718** |
| AUROC     | **0.9194** |

---

# Overall Model Performance

| Metric            |           Value |
| ----------------- | --------------: |
| Training Images   |     **223,414** |
| Validation Images |         **234** |
| Target Diseases   |           **5** |
| Architecture      | **DenseNet121** |
| Training Epochs   |          **10** |
| Training Time     |  **~4.5 Hours** |
| Mean AUROC        |      **0.8790** |

<img width="461" height="433" alt="image" src="https://github.com/user-attachments/assets/6c860145-af87-4b6c-991a-f2e5012355b7" />

---

# Key Observations

* The model achieved a **Mean AUROC of 0.8790** across the five selected thoracic diseases.
* **Pleural Effusion** achieved the highest AUROC (**0.9194**), indicating the strongest discrimination among the target diseases.
* **Edema** also achieved an AUROC above **0.90**, demonstrating strong performance.
* All five diseases achieved AUROC values greater than **0.81**, establishing a solid baseline for multi-label chest X-ray disease classification.

---

# Medical Disclaimer

This project was developed for **research, educational, and portfolio purposes only**.

The model is **not intended for clinical diagnosis, treatment planning, or medical decision-making**. Any predictions generated by the model should not be interpreted as medical advice and must not replace evaluation by qualified healthcare professionals.
