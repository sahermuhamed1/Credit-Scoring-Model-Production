# Example for prediction with saved model (e.g., in a separate predict.py script)
import pandas as pd
import numpy as np
import os
from src.utils import load_model, load_preprocessor
from config import RF_MODEL_PATH, SCALER_PATH, ORDINAL_ENCODER_PATH, LABEL_ENCODER_PATH, PAYMENT_BEHAVIOUR_ENCODER_PATH

def predict_new_data(new_data_df: pd.DataFrame):
    # Load preprocessors and model
    scaler = load_preprocessor(SCALER_PATH)
    credit_score_encoder = load_preprocessor(ORDINAL_ENCODER_PATH)
    label_encoder_occupation = load_preprocessor(LABEL_ENCODER_PATH) # Assuming this was saved as label_encoder.pkl
    label_encoder_type_of_loan = load_preprocessor(LABEL_ENCODER_PATH.replace('.pkl', '_type_of_loan.pkl'))
    label_encoder_payment_min_amount = load_preprocessor(LABEL_ENCODER_PATH.replace('.pkl', '_payment_min_amount.pkl'))
    label_encoder_num_of_loan = load_preprocessor(LABEL_ENCODER_PATH.replace('.pkl', '_num_of_loan.pkl'))
    payment_behaviour_encoder = load_preprocessor(PAYMENT_BEHAVIOUR_ENCODER_PATH)

    model = load_model(RF_MODEL_PATH)

    # Apply the same preprocessing steps as the training pipeline to new_data_df
    # This is a simplified example; a robust prediction function would encapsulate
    # the entire data preparation pipeline for inference.
    from src.data_cleaner import (
        clean_missing_values,
        perform_format_conversions,
        handle_outliers,
        apply_eda_based_scaling_filters
    )
    from src.feature_engineer import (
        create_new_features,
        scale_and_encode_features
    )
    from config import FEATURES, TARGET_COLUMN, OCCUPATION_COL, TYPE_OF_LOAN_COL, PAYMENT_OF_MIN_AMOUNT_COL, NUM_OF_LOAN_COL, PAYMENT_BEHAVIOUR_COL

    # IMPORTANT: For real-world deployment, you'd typically save and load
    # a `Pipeline` object that encapsulates all preprocessing.
    # For this modular conversion, we're applying functions sequentially.
    # Be mindful of what columns are dropped/transformed.

    # Start with a copy of the new data for safe modification
    processed_new_data = new_data_df.copy()

    # Step 1: Initial Renaming (if new data also has 'Total_EMI_per_month')
    processed_new_data.rename(columns={'Total_EMI_per_month': 'Monthly_EMI'}, inplace=True, errors='ignore')

    # Step 2: Mimic cleaning (this will be tricky without the original full dataset context,
    # especially for KNNImputer which needs fit on the training data)
    # For prediction, you should have pre-fitted imputer/scaler.
    # For simplicity, we'll run the cleaning steps, but highlight the limitation.
    # A better approach: Save the *fitted* imputer from training.

    # Manual imputation for simplicity in this example (not ideal for real use)
    processed_new_data = clean_missing_values(processed_new_data) # This will re-fit KNNImputer, which is bad for prediction
    processed_new_data = perform_format_conversions(processed_new_data)
    processed_new_data = handle_outliers(processed_new_data)
    processed_new_data = apply_eda_based_scaling_filters(processed_new_data)
    processed_new_data = create_new_features(processed_new_data) # Creates RFM and derived features


    # Apply saved encoders and scaler
    # Categorical encoding
    processed_new_data[OCCUPATION_COL] = label_encoder_occupation.transform(processed_new_data[OCCUPATION_COL])
    processed_new_data[TYPE_OF_LOAN_COL] = label_encoder_type_of_loan.transform(processed_new_data[TYPE_OF_LOAN_COL])
    processed_new_data[PAYMENT_OF_MIN_AMOUNT_COL] = label_encoder_payment_min_amount.transform(processed_new_data[PAYMENT_OF_MIN_AMOUNT_COL])
    processed_new_data[NUM_OF_LOAN_COL] = label_encoder_num_of_loan.transform(processed_new_data[NUM_OF_LOAN_COL])
    processed_new_data[PAYMENT_BEHAVIOUR_COL] = payment_behaviour_encoder.transform(processed_new_data[[PAYMENT_BEHAVIOUR_COL]])


    # Numerical scaling
    numerical_cols_for_scaling = [col for col in processed_new_data.select_dtypes(include=np.number).columns if col in FEATURES and col != TARGET_COLUMN]
    processed_new_data[numerical_cols_for_scaling] = scaler.transform(processed_new_data[numerical_cols_for_scaling])

    # Ensure feature order and presence
    X_predict = processed_new_data[FEATURES].copy()
    # Fill any NaNs that might have been introduced during feature engineering or that exist in new data
    for col in X_predict.columns:
        X_predict[col] = pd.to_numeric(X_predict[col], errors='coerce').fillna(X_predict[col].median())


    predictions_numeric = model.predict(X_predict)
    predictions_rounded = np.round(predictions_numeric).astype(int)
    predictions_labels = credit_score_encoder.inverse_transform(predictions_rounded.reshape(-1, 1))

    return predictions_labels.flatten().tolist(), predictions_numeric

# Example Usage:
# Assuming you have a `sample_data.csv` similar to your raw test.csv format but with new entries
# Create a dummy sample_data.csv for demonstration
sample_data = {
    'ID': [100001, 100002],
    'Customer_ID': ['CUS_999', 'CUS_888'],
    'Month': ['January', 'February'],
    'Name': ['John Doe', 'Jane Smith'],
    'Age': [35, 42],
    'SSN': ['abc', 'def'], # Dummy, should be dropped
    'Occupation': ['Engineer', 'Teacher'],
    'Annual_Income': ['75000$', '60,000'],
    'Num_Bank_Accounts': [2, 1],
    'Num_Credit_Card': [3, 2],
    'Interest_Rate': [12.5, 10.0],
    'Num_of_Loan': [2, 1],
    'Type_of_Loan': ['Auto Loan, Personal Loan', 'Mortgage Loan'],
    'Delay_from_due_date': [5, 0],
    'Num_of_Delayed_Payment': [1, 0],
    'Credit_Mix': ['Good', 'Standard'], # Dummy, should be dropped
    'Outstanding_Debt': ['2500_dollars', '1000_'],
    'Credit_Utilization_Ratio': [0.45, 0.20],
    'Credit_History_Age': ['5 Years and 3 Months', '10 Years and 0 Months'],
    'Payment_of_Min_Amount': ['No', 'Yes'],
    'Total_EMI_per_month': [300, 200],
    'Amount_invested_monthly': [500, 300],
    'Payment_Behaviour': ['Low_spent_Medium_value_payments', 'High_spent_Small_value_payments'],
    'Monthly_Inhand_Salary': ['6000', '4500'],
    'Monthly_Balance': [2000, 1500],
    'Credit_Score': [np.nan, np.nan] # These are the ones we want to predict
}
sample_df = pd.DataFrame(sample_data)

if __name__ == '__main__':
    # Run the full pipeline first to train and save models/preprocessors
    run_pipeline()

    print("\n--- Testing Prediction on Sample Data ---")
    # This calls a simplified `predict_new_data` function that would need to replicate the
    # *exact* preprocessing logic. For production, save a sklearn.pipeline.Pipeline.
    predicted_labels, predicted_scores = predict_new_data(sample_df)

    print("\nSample Data with Predictions:")
    sample_df['Predicted_Credit_Score_Label'] = predicted_labels
    sample_df['Predicted_Credit_Score_Numeric'] = predicted_scores
    print(sample_df[['Customer_ID', 'Age', 'Annual_Income', 'Outstanding_Debt',
                     'Predicted_Credit_Score_Label', 'Predicted_Credit_Score_Numeric']])