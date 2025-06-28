# 🐟 Fish Market Data Analysis

A data analysis project that explores the Fish Market dataset to understand relationships between fish species and physical features like length, height, and width — and how they affect weight using linear regression.

---

## 📂 Dataset

- Source: [Kaggle – Fish Market Dataset](https://www.kaggle.com/datasets/vipullrathod/fish-market)
- Contains physical measurements of different fish species (Length1, Length2, Height, Width, etc.) and their weight.

---

## 📊 Objectives

1. Visualize relationships between fish species and their features.
2. Add synthetic noise to the dataset and study its impact on model performance.
3. Train Linear Regression models on both original and noisy data.
4. Evaluate model accuracy using **MSE** and **R²** scores.

---

## 📈 Techniques Used

- 📌 Pandas, NumPy for data handling
- 📌 Matplotlib and Altair for visualization
- 📌 Scikit-learn for regression models and performance evaluation
- 📌 Added noise to features to analyze model robustness

---

## 🛠️ Installation

Install required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn altair opendatasets
