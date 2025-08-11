# main.py

import pandas as pd
import warnings
import numpy as np
import sys
import os

# Add src to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_and_combine_data
from data_cleaner import (
    clean_missing_values,
    perform_format_conversions,
    handle_outliers,
    apply_eda_based_scaling_filters
)
from feature_engineer import (
    create_new_features,
    perform_eda,
    scale_and_encode_features
)
from model_trainer import (
    split_and_prepare_data,
    train_model,
    evaluate_model
)
from config import TARGET_COLUMN, CREDIT_SCORE_CATEGORIES

warnings.filterwarnings("ignore", category=FutureWarning)

def run_pipeline():
    """
    Orchestrates the entire credit scoring classification pipeline.
    """
    print("Starting Credit Scoring Classification Pipeline...\n")

    # 1. Data Preparation & Exploration
    df = load_and_combine_data()

    # 2. Data Cleaning & Preprocessing
    df = clean_missing_values(df)
    df = perform_format_conversions(df)
    df = handle_outliers(df)
    df = apply_eda_based_scaling_filters(df)

    # 3. Feature Engineering
    df = create_new_features(df)
    perform_eda(df.copy()) # Pass a copy to EDA functions to avoid modifying original df for subsequent steps
    df = scale_and_encode_features(df) # This handles numerical scaling

    # 4. Model Development
    # Pass the 'df' to split_and_prepare_data. It will handle the train/test split,
    # categorical encoding, and SMOTE internally.
    X_train_split, y_train_split, X_val_split, y_val_split, X_test_final, y_test_true_labels = split_and_prepare_data(df)

    # Train the final RandomForestClassifier model
    rf_classifier_model = train_model(X_train_split, y_train_split)

    # Evaluate the model
    evaluate_model(rf_classifier_model, X_val_split, y_val_split)

    print("\nPipeline execution complete!")

    # Example of using the trained model on the actual test set (where Credit_Score was null)
    # Note: X_test_final contains the features for the data points where Credit_Score was NaN.
    # The true labels (y_test_true_labels) for this set are still NaN.
    # You would typically make predictions here and submit them or save them.
    print("\n--- Making Predictions on Original Test Data (where Credit_Score was NaN) ---")
    predictions_on_null_credit_score = rf_classifier_model.predict(X_test_final)
    # Convert numerical predictions back to categorical labels
    # Load the ordinal encoder used for Credit_Score
    from src.utils import load_preprocessor
    from config import ORDINAL_ENCODER_PATH
    try:
        credit_score_encoder = load_preprocessor(ORDINAL_ENCODER_PATH)
        predicted_credit_scores_labels = credit_score_encoder.inverse_transform(np.round(predictions_on_null_credit_score).reshape(-1, 1))
        print(f"Sample predictions (numeric): {predictions_on_null_credit_score[:5]}")
        print(f"Sample predictions (labels): {predicted_credit_scores_labels[:5].flatten().tolist()}")
        print(f"Distribution of predicted labels: {pd.Series(predicted_credit_scores_labels.flatten()).value_counts()}")
    except FileNotFoundError:
        print(f"Warning: Ordinal encoder not found at {ORDINAL_ENCODER_PATH}. Cannot inverse transform predictions.")
        print(f"Sample predictions (numeric): {predictions_on_null_credit_score[:5]}")


if __name__ == "__main__":
    run_pipeline()