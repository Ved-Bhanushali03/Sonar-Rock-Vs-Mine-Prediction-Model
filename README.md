# SONAR Underwater Object Detection (Rock vs. Mine)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)

An end-to-end Machine Learning pipeline utilizing Supervised Learning to classify underwater objects detected by SONAR signals as either a **Rock (R)** or a **Metal Cylinder/Explosive Mine (M)**. This predictive architecture is tailored for defensive maritime transit systems (such as autonomous submarines) to dynamically navigate or mitigate sub-surface operational hazards.

---

## 📌 Project Overview
Naval vessels and submarines rely heavily on Sound Navigation and Ranging (SONAR) data to map terrains and identify underwater entities. This system automates the processing of raw frequency response signatures bounced off objects. 

By analyzing specific signal patterns across 60 discrete frequency bands, the model executes a binary classification task to determine whether an anomaly is a harmless geological formation (Rock) or an explosive threat (Mine).

### Key Architectural Flow:
1. **Data Ingestion & In-Memory DataFraming** via `Pandas`
2. **Statistical Profiling & Mean Matrix Separation**
3. **Feature-Label Isolation**
4. **Stratified Train-Test Matrix Splitting** (Maintaining target baseline ratio)
5. **Discriminative Supervised Training** via `Logistic Regression`
6. **Inference Pipeline Integration** (Single-instance operational prediction)

---

## 📊 Dataset Profile
The architecture processes structured `.csv` frequency matrix configurations consisting of:
* **Instances:** 208 recorded signal responses present.
* **Features:** 60 continuous numeric attributes representing energy integration across normalized frequency bands (0.0 to 1.0).
* **Target Classification:** Binary structural labels (`M` = Mine, `R` = Rock).
* **Class Balance:** 111 Mine profiles, 97 Rock profiles (optimized, balanced class weights).

---

## 🛠 Tech Stack & Core Dependencies
The runtime environment operates purely using foundational python mathematical and scientific computing infrastructure:
* **Core Runtime:** Python 3.8+ (Cloud-ready on Google Colaboratory / Jupyter Engine)
* **Data Manipulation:** `pandas`
* **Numerical Processing:** `numpy`
* **Model Pipeline:** `scikit-learn` (`linear_model.LogisticRegression`, `model_selection.train_test_split`, `metrics.accuracy_score`)

## 📈 Performance Benchmarks
Training Metrics: ~83.42% Accurate Classification.

Testing Evaluation Matrix: ~76.19% Generalization Performance.

The narrow variance between the training execution benchmarks and validation arrays demonstrates low model overfitting, rendering the architecture reliable for out-of-sample data points.
---
## 🚀 Execution & Implementation Pipeline
### 1. Repository Setup & Installation
Clone this repository locally or upload the script/dataset straight to your target workspace:
```bash
git clone https://github.com/Ved-Bhanushali03/Sonar-Rock-Vs-Mine-Prediction-Model.git
cd sonar-rock-mine-prediction
