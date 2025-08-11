# src/utils.py

import os
import joblib
import warnings
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

def parse_years_and_months(age: str) -> int:
    """Convert 'X Years and Y Months' to total months."""
    if isinstance(age, str):
        try:
            years = int(age.split(' Years')[0]) if 'Years' in age else 0
            months = int(age.split('and')[-1].split(' Months')[0].strip()) if 'Months' in age else 0
            return years * 12 + months
        except Exception:
            return 0
    return 0

def map_loan_type(value: str, loan_map: dict) -> str:
    """Map raw loan types to standardized categories."""
    value_lower = str(value).lower()
    for keyword, category in loan_map.items():
        if re.search(keyword, value_lower):
            return category
    return 'Other'

def save_preprocessor(preprocessor, path: str):
    """Saves a preprocessor (e.g., scaler, encoder) to a specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")

def load_preprocessor(path: str):
    """Loads a preprocessor from a specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor file not found at {path}")
    return joblib.load(path)

def save_model(model, path: str):
    """Saves a trained model to a specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path: str):
    """Loads a trained model from a specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

def plot_boxplots(df: pd.DataFrame, title: str = "Boxplots of Numerical Features"):
    """Plots boxplots for all numerical columns in a DataFrame."""
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = len(numerical_columns)
    rows = (num_cols // 3) + (1 if num_cols % 3 else 0)

    plt.figure(figsize=(15, rows * 4)) # Adjusted height
    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(rows, 3, i)
        sns.boxplot(x=df[column])
        plt.title(f"{column}\nMedian: {df[column].median():.1f}")
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16) # Add a main title
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix of Numerical Features"):
    """Plots a correlation matrix heatmap for numerical columns."""
    numerical_columns = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numerical_columns.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                square=True,
                linewidths=1,
                cbar_kws={"shrink": .5})

    plt.title(title, pad=20, size=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\nHighly correlated features (|correlation| > 0.5):")
    high_corr = np.where(np.abs(correlation_matrix) > 0.5)
    for i, j in zip(*high_corr):
        if i != j:
            print(f"{correlation_matrix.index[i]} -> {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.2f}")

def plot_categorical_distributions(df: pd.DataFrame):
    """Plots distributions for Credit Score, Type of Loan, and Delay Flag."""
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    sns.histplot(df['Credit_Score'], ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution of Credit Score')

    sns.countplot(x='Type_of_Loan', data=df, ax=axes[1], palette='pastel')
    axes[1].set_title('Distribution of Type of Loan')
    axes[1].tick_params(axis='x', rotation=45)

    sns.countplot(x='Delay_Flag', data=df, ax=axes[2], palette='pastel')
    axes[2].set_title('Distribution of Delay Flag')
    axes[2].set_xticklabels(['No Delay', 'Delay'])

    plt.tight_layout()
    plt.show()