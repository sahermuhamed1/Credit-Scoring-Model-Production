# src/feature_engineer.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from src.utils import map_loan_type, plot_correlation_matrix, plot_categorical_distributions, save_preprocessor
from config import (
    TARGET_COLUMN, CUSTOMER_ID_COL, MONTH_COL, OUTSTANDING_DEBT_COL,
    PAYMENT_BEHAVIOUR_COL, LOAN_TYPE_MAPPING, CREDIT_HISTORY_AGE_COL,
    NUM_BANK_ACCOUNTS_COL, NUM_CREDIT_CARD_COL, ANNUAL_INCOME_COL,
    NUM_OF_DELAYED_PAYMENT_COL, TOTAL_NUM_ACCOUNTS_COL,
    DEBT_PER_ACCOUNT_COL, DEBT_TO_INCOME_RATIO_COL, DELAYED_PAYMENTS_PER_ACCOUNT_COL,
    CREDIT_SCORE_CATEGORIES, PAYMENT_BEHAVIOUR_CATEGORIES,
    OCCUPATION_COL, TYPE_OF_LOAN_COL, PAYMENT_OF_MIN_AMOUNT_COL,
    SCALER_PATH, ORDINAL_ENCODER_PATH, LABEL_ENCODER_PATH,
    PAYMENT_BEHAVIOUR_ENCODER_PATH # New encoder path
)

def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates RFM (Recency, Frequency, Monetary) and other derived features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    print("\n--- Creating New Features ---")

    # Drop 'col' column if it exists (was a temporary column from cleaning)
    if 'col' in df.columns:
        df.drop(columns=['col'], inplace=True)
        print("Dropped temporary 'col' column.")

    # Move target column to end (for cleaner viewing, not strictly necessary for model)
    if TARGET_COLUMN in df.columns:
        credit_score = df.pop(TARGET_COLUMN)
        df[TARGET_COLUMN] = credit_score
        print(f"Moved '{TARGET_COLUMN}' to the end of the DataFrame.")

    # RFM Features
    df['Frequency'] = df[CUSTOMER_ID_COL].map(df[CUSTOMER_ID_COL].value_counts())
    print(f"Created '{'Frequency'}' feature.")

    # Encode months as integers for Recency calculation
    month_mapping_to_int = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                            'May': 5, 'June': 6, 'July': 7,
                            'August': 8, 'September': 9, 'October': 10,
                            'November': 11, 'December': 12}
    df['Month_Encoded'] = df[MONTH_COL].map(month_mapping_to_int)

    # Get the latest month for each Customer_ID, assume data starts from month 9 (Sept)
    df['Recency'] = df.groupby(CUSTOMER_ID_COL)['Month_Encoded'].transform('max') - 9
    print(f"Created '{'Recency'}' feature.")

    df['Monetary'] = df.groupby(CUSTOMER_ID_COL)[OUTSTANDING_DEBT_COL].transform('last')
    print(f"Created '{'Monetary'}' feature.")

    # Decode month column back to original string for potential EDA or future use
    month_mapping_to_str = {v: k for k, v in month_mapping_to_int.items()}
    df[MONTH_COL] = df['Month_Encoded'].map(month_mapping_to_str)
    df.drop(columns=['Month_Encoded'], inplace=True) # Drop the temporary encoded month column
    print(f"Decoded '{MONTH_COL}' back to string and dropped temporary 'Month_Encoded'.")

    # Clean Payment_Behaviour
    df[PAYMENT_BEHAVIOUR_COL].replace({'!@9#%8': 'Medium_spent_Moderate_value_payments'}, inplace=True)
    print(f"Cleaned '{PAYMENT_BEHAVIOUR_COL}'.")

    # Map Type_of_Loan
    df[TYPE_OF_LOAN_COL] = df[TYPE_OF_LOAN_COL].apply(lambda x: map_loan_type(x, LOAN_TYPE_MAPPING))
    print(f"Mapped '{TYPE_OF_LOAN_COL}' to standardized categories.")

    # Create additional derived features
    # Handle division by zero for Total_Num_Accounts if it can be zero after cleaning
    df[TOTAL_NUM_ACCOUNTS_COL] = df[NUM_BANK_ACCOUNTS_COL] + df[NUM_CREDIT_CARD_COL]
    df[TOTAL_NUM_ACCOUNTS_COL] = df[TOTAL_NUM_ACCOUNTS_COL].replace(0, 1) # Avoid division by zero
    df[DEBT_PER_ACCOUNT_COL] = df[OUTSTANDING_DEBT_COL] / df[TOTAL_NUM_ACCOUNTS_COL]
    print(f"Created '{TOTAL_NUM_ACCOUNTS_COL}' and '{DEBT_PER_ACCOUNT_COL}'.")

    # Handle division by zero for Annual_Income
    df[ANNUAL_INCOME_COL] = df[ANNUAL_INCOME_COL].replace(0, 1) # Avoid division by zero
    df[DEBT_TO_INCOME_RATIO_COL] = df[OUTSTANDING_DEBT_COL] / df[ANNUAL_INCOME_COL]
    print(f"Created '{DEBT_TO_INCOME_RATIO_COL}'.")

    df[DELAYED_PAYMENTS_PER_ACCOUNT_COL] = df[NUM_OF_DELAYED_PAYMENT_COL] / df[TOTAL_NUM_ACCOUNTS_COL]
    print(f"Created '{DELAYED_PAYMENTS_PER_ACCOUNT_COL}'.")

    return df

def perform_eda(df: pd.DataFrame):
    """
    Performs Exploratory Data Analysis (EDA) visualizations.
    """
    print("\n--- Performing EDA ---")
    plot_correlation_matrix(df.copy())
    plot_categorical_distributions(df.copy())

def scale_and_encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numerical features and encodes categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with scaled and encoded features.
    """
    print("\n--- Scaling & Encoding Features ---")

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(
        [col for col in [TARGET_COLUMN, CUSTOMER_ID_COL] if col in df.columns], errors='ignore'
    )
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"Numerical columns for scaling: {numerical_cols.tolist()}")
    print(f"Categorical columns for encoding: {categorical_cols}")

    # Scale numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    save_preprocessor(scaler, SCALER_PATH)
    print(f"Scaled numerical columns using StandardScaler and saved to {SCALER_PATH}.")

    # Encoding 'Credit_Score' for training data
    # This should be applied only to the 'train' part of the df later
    # For now, we'll keep it as is, or handle it during split if `df` still contains NaN in `Credit_Score`
    # The original notebook applied this after split, which is better.
    # We will keep the categorical columns for now and encode them after the train/test split.
    # The model training step will handle the specific encoding for train/test.

    return df