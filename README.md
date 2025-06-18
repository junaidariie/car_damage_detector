# Car Damage Detector 🚗

A **Deep Learning-based Car Damage Detection App** that identifies different types of car damage from images using a ResNet50 model. Built with PyTorch and Streamlit, this app offers a user-friendly interface for fast and accurate predictions.

### 🌐 Live Demo

👉 [Streamlit App](https://cardamagedetector-e3smtwusr8p8kmhywkq7zy.streamlit.app/)

![App Screenshot](https://github.com/user-attachments/assets/47bc2549-4662-4dcc-b807-07db30678a32)

---

## 🧠 Model Overview

The model classifies images into the following six categories:

* `F_Breakage`
* `F_Crushed`
* `F_Normal`
* `R_Breakage`
* `R_Crushed`
* `R_Normal`

### 📊 Model Progression & Accuracy

| Model Description                       | Accuracy (%) |
| --------------------------------------- | ------------ |
| CNN (Custom from scratch)               | 47           |
| Transfer Learning: EfficientNet         | 71           |
| Transfer Learning: EfficientNet B4      | 65           |
| **Transfer Learning: ResNet50** (Final) | **80**       |

### ✅ Final Validation Results (ResNet50)

```
              precision    recall  f1-score   support

  F_Breakage       0.88      0.83      0.86       100
   F_Crushed       0.77      0.72      0.74        82
    F_Normal       0.82      0.92      0.87        89
  R_Breakage       0.71      0.86      0.78        64
   R_Crushed       0.72      0.70      0.71        66
    R_Normal       0.88      0.71      0.79        59

    accuracy                           0.80       460
   macro avg       0.80      0.79      0.79       460
weighted avg       0.80      0.80      0.80       460
```

---

## 🚀 Features

* 🔍 Detects different types of **car damage** (front/rear, breakage/crushed/normal).
* 🧠 Powered by **ResNet50** and **PyTorch**.
* 🎯 Achieves **\~80% validation accuracy** on real-world data.
* 🖼️ Easy-to-use **Streamlit web interface**.
* 💾 Model stored as `save_model.pth`.

---
