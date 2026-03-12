| Package            | Purpose                         | Install Command                |
| ------------------ | ------------------------------- | ------------------------------ |
| `streamlit`        | Web dashboard                   | `pip install streamlit`        |
| `pandas`           | CSV reading & data manipulation | `pip install pandas`           |
| `joblib`           | Save/load ML models             | `pip install joblib`           |
| `matplotlib`       | Charts (pie/bar)                | `pip install matplotlib`       |
| `scikit-learn`     | ML models & preprocessing       | `pip install scikit-learn`     |
| `imbalanced-learn` | Handle imbalanced data (SMOTE)  | `pip install imbalanced-learn` |


PAG TAMAD NA TALAGA
pip install streamlit pandas joblib matplotlib scikit-learn imbalanced-learn

# Streamlit dashboard
import streamlit as st

# Data handling
import pandas as pd
import joblib

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt

# Optional
import os  # for file checks

#PAG I TRAIN NA WAX ON WAX OFF ECHOZ
python train.py

#PARA MA RUN Streamlit
python -m streamlit run app.py