# Sign Language Action Recognition Using an LSTM Network (Weighted Accuracy ≈ 0.84)

This project implements a real-time sign language action recognition system using a Long Short-Term Memory (LSTM) neural network.  
The model is trained to classify five American Sign Language (ASL) gestures: `hello`, `thanks`, `iloveyou`, `yes`, and `no`.

The system achieves **0.83 overall accuracy** and a **weighted F1-score of 0.84** using a 80/20 train-test split.  
Temporal dependencies are modeled using stacked LSTM layers, and spatial features are extracted from keypoints detected via the MediaPipe Holistic model.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![Keras](https://img.shields.io/badge/Keras-API-red?style=flat&logo=keras)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange?style=flat&logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Keypoints-yellow?style=flat)
![F1 Weighted](https://img.shields.io/badge/F1%20Score%20(Weighted)-0.84-brightgreen?style=flat)

---

## Demo Screen Shot

![image](https://github.com/user-attachments/assets/86e6a36b-a921-4b63-9d2e-3b677ae46fae)

---

## Evaluation

**Overall Accuracy**: **83%**  

**Macro F1-Score**: **0.80**

**Weighted F1-Score**: **0.84**

<img width="389" alt="image" src="https://github.com/user-attachments/assets/e7a9294f-0258-452f-b723-efd55ff74724" />

### Strengths
- **"thanks"** gesture is classified with perfect performance (Precision, Recall, F1-score = 1.00), suggesting strong feature representation and consistent training.
- **"hello"** and **"no"** both demonstrate high F1-scores (≥ 0.86), indicating reliable recognition across varied samples.
- Overall **macro and weighted F1-scores** of 0.80 and 0.84 reflect solid performance across most classes.

### Weaknesses
- **"iloveyou"** has the lowest F1-score (0.57), likely due to:
  - Fewer examples in the dataset (support = 3).
  - High intra-class variation or visual overlap with other gestures.
- **"yes"** exhibits moderate performance (F1-score = 0.67), which may result from ambiguous or inconsistent keypoint patterns.
- Class imbalance may be affecting generalization; additional data collection or augmentation is recommended for underperforming classes.

---

## Model Architecture

**Type:** Sequential LSTM Neural Network  
**Framework:** Keras with TensorFlow backend

```
Input: (30 frames × 1662 features)

LSTM Layer 1   ── LSTM(64 units)     ── Activation: tanh     ── return_sequences=True  
LSTM Layer 2   ── LSTM(128 units)    ── Activation: tanh     ── return_sequences=True  

LSTM Layer 3   ── LSTM(64 units)     ── Activation: tanh     ── return_sequences=False  

Dense Layer 1  ── Dense(64 units)    ── Activation: ReLU  

Dense Layer 2  ── Dense(32 units)    ── Activation: ReLU  

Output Layer   ── Dense(5 units)     ── Activation: Softmax

Loss Function:  Categorical Crossentropy  
Optimizer:      Adam

```

---

## Key Features

- Real time, webcam based gesture recognition
- Extracts 1662-dim keypoint vectors using MediaPipe Holistic
- Sequential LSTM layers for temporal pattern recognition
- Sentence overlay showing recent predictions
- Probability indicators for visual feedback
- TensorBoard logging support

---

## Known Issues and Future Improvements

**Low Performance on Certain Gestures:**  
The model struggles with gestures such as `yes` and `iloveyou` due to:

- High inter-user variability in signing
- Subtle, fast movements or finger configurations
- Small sample size for these classes

**Proposed Fixes:**

- **Collect more diverse examples** for underperforming signs
- **Data augmentation**: simulate jitter, hand rotation, or occlusion
- **Apply temporal smoothing** like moving averages or Kalman filters
- **Class weighting** or **SMOTE** to balance the training distribution
- **Attention mechanisms** or **transformer encoders** to capture key frames
- **Frame-wise data analysis** to identify high-information keypoints

---

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn

---
