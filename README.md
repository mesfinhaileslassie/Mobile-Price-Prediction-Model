# Mobile-Price-Prediction-Model

A machine learning project that predicts mobile phone price ranges based on specifications like RAM, battery power, camera quality, and more.

## 🎯 Project Overview

This project classifies mobile phones into **4 price ranges** (0 = Low Cost, 1 = Medium Cost, 2 = High Cost, 3 = Very High Cost) based on their technical specifications. The model achieves **96.5% accuracy** using Logistic Regression.

### 📊 Dataset Features

| Feature | Description | Range |
|---------|-------------|-------|
| `battery_power` | Battery capacity in mAh | 501 - 1998 |
| `blue` | Bluetooth support | 0 (No), 1 (Yes) |
| `clock_speed` | Processor speed in GHz | 0.5 - 3.0 |
| `dual_sim` | Dual SIM support | 0, 1 |
| `fc` | Front camera in MP | 0 - 19 |
| `four_g` | 4G support | 0, 1 |
| `int_memory` | Internal memory in GB | 2 - 64 |
| `m_dep` | Mobile depth in cm | 0.1 - 1.0 |
| `mobile_wt` | Weight in grams | 80 - 200 |
| `n_cores` | Number of processor cores | 1 - 8 |
| `pc` | Primary camera in MP | 0 - 20 |
| `px_height` | Pixel resolution height | 0 - 1960 |
| `px_width` | Pixel resolution width | 500 - 1998 |
| `ram` | RAM in MB | 256 - 3998 |
| `sc_h` | Screen height in cm | 5 - 19 |
| `sc_w` | Screen width in cm | 0 - 18 |
| `talk_time` | Battery life in hours | 2 - 20 |
| `three_g` | 3G support | 0, 1 |
| `touch_screen` | Touch screen support | 0, 1 |
| `wifi` | WiFi support | 0, 1 |

**Target Variable:** `price_range` (0, 1, 2, 3) - Balanced dataset with 500 samples per class.

## 📁 Project Structure
Mobile Price classification/
│
├── data/
│ ├── train.csv # Training data (2000 samples)
│ ├── test.csv # Test data (1000 samples)
│ └── submission.csv # Final predictions
│
├── models/
│ ├── logistic_regression_model.pkl # Trained model
│ └── standard_scaler.pkl # Fitted scaler
│
├── mobile_price_model.py # Main Python script
├── requirements.txt # Dependencies
└── README.md # Project documentation

text

## 🚀 Results Summary

### 🏆 Best Model: Logistic Regression

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **96.50%** |
| Training Accuracy | 97.69% |
| Overfitting Gap | 1.19% |
| Training Time | 0.014 seconds |

### 📈 Model Comparison

| Model | Validation Accuracy | Training Time |
|-------|-------------------|---------------|
| **Logistic Regression** | **96.50%** | 0.014s |
| Gradient Boosting | 91.25% | 1.27s |
| Support Vector Classifier | 89.50% | 0.18s |
| Random Forest | 88.00% | 0.22s |
| Decision Tree | 83.00% | 0.008s |
| K-Nearest Neighbors | 50.00% | 1.34s |

### 📊 Feature Importance (Top 5)

| Rank | Feature | Importance (Coefficient Magnitude) |
|------|---------|-----------------------------------|
| 1 | **ram** | 3.82 |
| 2 | **battery_power** | 1.31 |
| 3 | **px_height** | 0.89 |
| 4 | **px_width** | 0.76 |
| 5 | **mobile_wt** | 0.61 |

**Key Insight:** RAM is overwhelmingly the most important predictor of mobile price range.
