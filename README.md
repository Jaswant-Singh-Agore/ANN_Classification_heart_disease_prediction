Heart Disease Prediction using Artificial Neural Networks (ANN)

### End-to-End Deep Learning Project | Python | TensorFlow | Scikit-learn | Modular Architecture

---

## Overview

This project predicts the likelihood of **heart disease** in patients using an **Artificial Neural Network (ANN)** built with TensorFlow and Keras.  
It demonstrates an **end-to-end machine learning pipeline** — from raw data ingestion to model evaluation — built in a **modular and production-ready format**.

---

## Project Highlights

- **End-to-End ML Workflow:** Raw data → Preprocessing → Training → Evaluation  
- **Modular Design:** Each step isolated in its own script (clean, reusable, and scalable)  
- **Custom Logging & Exception Handling:** Centralized error tracking and logging  
- **Configurable Pipeline:** Full control using a single command via `main.py`  
- **Reproducible Environment:** Easy setup via `requirements.txt`  

---

## Folder Structure
ANN_Heart_Disease/
│
├── src/
│ ├── data_loader.py # Handles dataset loading
│ ├── data_preprocessing.py # Cleans, encodes, splits data
│ ├── model.py # ANN model architecture
│ ├── model_train.py # Training pipeline
│ ├── evaluate.py # Model performance evaluation
│ ├── logger.py # Centralized logging setup
│ ├── exception.py # Custom exception handling
│ ├── run_preprocessing.py # Runs preprocessing pipeline
│ ├── run_training.py # Runs model training pipeline
│ ├── run_evaluate.py # Runs evaluation pipeline
│ └── inference.py # Predicts on new unseen data
│
├── notebooks/ # Jupyter notebooks for EDA
├── data/ # (ignored in Git) raw & processed data
├── artifacts/ # (ignored in Git) trained models & figures
│
├── main.py # One-click pipeline runner
├── requirements.txt # Python dependencies
├── LICENSE # License file
├── .gitignore # Files ignored by Git
└── README.md # You’re reading it :)


---

## Model Architecture

| Layer | Type | Units | Activation |
|-------|------|--------|-------------|
| 1 | Dense | 64 | ReLU |
| 2 | Dense | 32 | ReLU |
| 3 | Dense | 1 | Sigmoid |

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, AUC

- ## Actual Evaluation Results
| Metric | Score |
|---------|--------|
| Accuracy | 0.87 |
| AUC | **0.8929** |
| Precision | 0.88 |
| Recall | 0.84 |


---

## Tech Stack

| Category | Tools Used |
|-----------|------------|
| Programming | Python 3.10 |
| Libraries | TensorFlow, Keras, NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Environment | Conda / pip |
| Logging & Error Handling | Custom logger + CustomException |
| Version Control | Git & GitHub |

---




