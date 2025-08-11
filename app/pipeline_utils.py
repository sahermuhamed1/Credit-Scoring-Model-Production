# app/pipeline_utils.py (with added debug prints)

import pandas as pd
import numpy as np
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COLUMNS_TO_DROP, NAME_COL, TYPE_OF_LOAN_COL, CREDIT_HISTORY_AGE_COL,
    OUTSTANDING_DEBT_COL, ANNUAL_INCOME_COL, AGE_COL, NUM_OF_LOAN_COL,
    NUMERICAL_IMPUTE_COLS, POSITIVE_CLIP_COLS, NUM_BANK_ACCOUNTS_COL,
    NUM_CREDIT_CARD_COL, DELAY_FROM_DUE_DATE_COL,
    AGE_UPPER_BOUND, NUM_BANK_ACCOUNTS_UPPER_BOUND, OUTSTANDING_DEBT_UPPER_BOUND,
    AMOUNT_INVESTED_MONTHLY_COL, MONTHLY_BALANCE_COL, MONTHLY_EMI_COL,
    NUM_OF_DELAYED_PAYMENT_COL,
    AMOUNT_INVESTED_MONTHLY_QUANTILE_CLIP, DELAY_FROM_DUE_DATE_UPPER_BOUND,
    AGE_EDA_UPPER, MONTHLY_INHAND_SALARY_EDA_UPPER, MONTHLY_EMI_UPPER_BOUND,
    NUM_CREDIT_CARD_EDA_LOWER, NUM_CREDIT_CARD_EDA_UPPER,
    INTEREST_RATE_EDA_UPPER, NUM_OF_LOAN_EDA_UPPER,
    DELAY_FROM_DUE_DATE_EDA_UPPER, OUTSTANDING_DEBT_EDA_UPPER,
    MONTHLY_EMI_EDA_UPPER_1, AMOUNT_INVESTED_MONTHLY_EDA_LOWER,
    AMOUNT_INVESTED_MONTHLY_EDA_UPPER, MONTHLY_EMI_EDA_UPPER_2,
    LOAN_TYPE_MAPPING, PAYMENT_BEHAVIOUR_COL,
    CUSTOMER_ID_COL, MONTH_COL, OCCUPATION_COL,
    TOTAL_NUM_ACCOUNTS_COL, DEBT_PER_ACCOUNT_COL, DEBT_TO_INCOME_RATIO_COL,
    DELAYED_PAYMENTS_PER_ACCOUNT_COL, DELAY_FLAG_COL, FREQUENCY_COL, RECENCY_COL, MONETARY_COL,
    RF_MODEL_PATH, SCALER_PATH, ORDINAL_ENCODER_PATH, PAYMENT_BEHAVIOUR_ENCODER_PATH,
    LABEL_ENCODER_OCCUPATION_PATH, LABEL_ENCODER_TYPE_OF_LOAN_PATH,
    LABEL_ENCODER_PAYMENT_MIN_AMOUNT_PATH, PAYMENT_OF_MIN_AMOUNT_COL,
    FEATURES, CREDIT_SCORE_CATEGORIES, MONTHLY_INHAND_SALARY_COL
)
from src.utils import parse_years_and_months, map_loan_type, load_model, load_preprocessor

rf_model = None
scaler = None
credit_score_encoder = None
payment_behaviour_encoder = None
le_occupation = None
le_type_of_loan = None
le_payment_min_amount = None
le_num_of_loan = None

numerical_medians = {
    MONTHLY_INHAND_SALARY_COL: 4000.0,
    NUM_OF_DELAYED_PAYMENT_COL: 0.0,
    AMOUNT_INVESTED_MONTHLY_COL: 100.0,
    MONTHLY_BALANCE_COL: 500.0,
    ANNUAL_INCOME_COL: 60000.0,
    OUTSTANDING_DEBT_COL: 5000.0,
    AGE_COL: 30.0,
    NUM_OF_LOAN_COL: 2.0,
    'Interest_Rate': 15.0,
    'Changed_Credit_Limit': 10.0,
    'Credit_Utilization_Ratio': 0.3,
    'Num_Bank_Accounts': 2.0,
    'Num_Credit_Card': 3.0,
    'Delay_from_due_date': 0.0,
    MONTHLY_EMI_COL: 200.0,
    'Credit_History_Age_Months': 60.0,
    'Total_Num_Accounts': 5.0,
    'Debt_Per_Account': 1000.0,
    'Debt_to_Income_Ratio': 0.08,
    'Delayed_Payments_Per_Account': 0.0,
    'Delay_Flag': 0.0,
    'Frequency': 1.0,
    'Recency': 0.0,
    'Monetary': 5000.0,
}

EXPECTED_RAW_INPUT_COLS = [
    'ID', 'Customer_ID', 'Month', 'Name', 'Age', 'SSN', 'Occupation', 'Annual_Income',
    'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
    'Interest_Rate', 'Num_of_Loan', 'Type_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Credit_Mix', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
    'Credit_History_Age', 'Payment_of_Min_Amount', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance', 'Credit_Score'
]


def load_all_preprocessors_and_model():
    global rf_model, scaler, credit_score_encoder, payment_behaviour_encoder, \
           le_occupation, le_type_of_loan, le_payment_min_amount, le_num_of_loan
    try:
        rf_model = load_model(RF_MODEL_PATH)
        scaler = load_preprocessor(SCALER_PATH)
        credit_score_encoder = load_preprocessor(ORDINAL_ENCODER_PATH)
        payment_behaviour_encoder = load_preprocessor(PAYMENT_BEHAVIOUR_ENCODER_PATH)
        le_occupation = load_preprocessor(LABEL_ENCODER_OCCUPATION_PATH)
        le_type_of_loan = load_preprocessor(LABEL_ENCODER_TYPE_OF_LOAN_PATH)
        le_payment_min_amount = load_preprocessor(LABEL_ENCODER_PAYMENT_MIN_AMOUNT_PATH)
        # le_num_of_loan is no longer loaded/used for encoding
        
        print("All models and preprocessors loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading a file: {e}. Please ensure all model and preprocessor files exist in the '{os.path.basename(os.path.dirname(RF_MODEL_PATH))}' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        sys.exit(1)


def safe_extract_and_impute_int_inference(value, default_median):
    if pd.isna(value):
        return default_median
    try:
        match = re.search(r'(\d+)', str(value))
        if match:
            return int(match.group(1))
        else:
            return default_median
    except (ValueError, TypeError):
        return default_median

def preprocess_single_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    print(f"\n--- DEBUG START ---")
    print(f"Input data keys: {data.keys()}")
    print(f"Config FEATURES length: {len(FEATURES)}")
    print(f"Config FEATURES: {FEATURES}")

    for col in EXPECTED_RAW_INPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[EXPECTED_RAW_INPUT_COLS]
    print(f"After initial raw column alignment, df.columns: {df.columns.tolist()}")

    df.rename(columns={'Total_EMI_per_month': MONTHLY_EMI_COL}, inplace=True)
    print(f"After EMI rename, df.columns: {df.columns.tolist()}")

    df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], inplace=True, errors='ignore')
    print(f"After dropping initial cols, df.columns: {df.columns.tolist()}")


    # --- Cleaning Missing Values ---
    df[NAME_COL] = df[NAME_COL].fillna('Unknown')
    df[TYPE_OF_LOAN_COL] = df[TYPE_OF_LOAN_COL].fillna('Unknown')
    df[CREDIT_HISTORY_AGE_COL] = df[CREDIT_HISTORY_AGE_COL].fillna('Unknown')

    for col in NUMERICAL_IMPUTE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(numerical_medians.get(col, 0.0))

    df.loc[:, ANNUAL_INCOME_COL] = df[ANNUAL_INCOME_COL].astype(str).str.replace(r'[^0-9.]', '', regex=True)
    df.loc[:, ANNUAL_INCOME_COL] = pd.to_numeric(df[ANNUAL_INCOME_COL], errors='coerce').fillna(numerical_medians.get(ANNUAL_INCOME_COL, 0.0))

    df.loc[:, AGE_COL] = df[AGE_COL].apply(lambda x: safe_extract_and_impute_int_inference(x, numerical_medians.get(AGE_COL, 30.0)))
    df.loc[:, NUM_OF_LOAN_COL] = df[NUM_OF_LOAN_COL].apply(lambda x: safe_extract_and_impute_int_inference(x, numerical_medians.get(NUM_OF_LOAN_COL, 2.0)))
    df.loc[:, NUM_OF_LOAN_COL] = pd.to_numeric(df[NUM_OF_LOAN_COL], errors='coerce').fillna(numerical_medians.get(NUM_OF_LOAN_COL, 2.0))


    df.loc[:, OUTSTANDING_DEBT_COL] = df[OUTSTANDING_DEBT_COL].astype(str).str.strip().str.strip('_')
    df.loc[:, OUTSTANDING_DEBT_COL] = pd.to_numeric(df[OUTSTANDING_DEBT_COL], errors='coerce').fillna(numerical_medians.get(OUTSTANDING_DEBT_COL, 0.0))

    print(f"After cleaning values, df.columns: {df.columns.tolist()}")
    print(f"Current df.shape: {df.shape}")

    # --- Format Conversions & Initial Clipping ---
    df['Credit_History_Age_Months'] = df[CREDIT_HISTORY_AGE_COL].apply(parse_years_and_months)
    df['Credit_History_Age_Months'] = df['Credit_History_Age_Months'].fillna(numerical_medians.get('Credit_History_Age_Months', 0.0))

    df.loc[:, AGE_COL] = df[AGE_COL].clip(lower=14, upper=AGE_UPPER_BOUND)
    df.loc[:, MONTHLY_BALANCE_COL] = df[MONTHLY_BALANCE_COL].clip(lower=-1e5, upper=1e5)

    for column in POSITIVE_CLIP_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(numerical_medians.get(column, 0.0))
            df[column] = df[column].clip(lower=0)
    print(f"After format conversions/initial clipping, df.columns: {df.columns.tolist()}")
    print(f"Current df.shape: {df.shape}")

    # --- Handle Outliers ---
    df.loc[:, AGE_COL] = df[AGE_COL].clip(upper=AGE_UPPER_BOUND)
    df.loc[:, NUM_BANK_ACCOUNTS_COL] = df[NUM_BANK_ACCOUNTS_COL].clip(upper=NUM_BANK_ACCOUNTS_UPPER_BOUND)
    df.loc[:, OUTSTANDING_DEBT_COL] = df[OUTSTANDING_DEBT_COL].clip(upper=OUTSTANDING_DEBT_UPPER_BOUND)

    if AMOUNT_INVESTED_MONTHLY_COL in df.columns:
        df.loc[:, AMOUNT_INVESTED_MONTHLY_COL] = df[AMOUNT_INVESTED_MONTHLY_COL].apply(lambda x: x if x > 0 else 1e-6)
        df.loc[:, AMOUNT_INVESTED_MONTHLY_COL] = np.log1p(df[AMOUNT_INVESTED_MONTHLY_COL])
        df.loc[:, AMOUNT_INVESTED_MONTHLY_COL] = df[AMOUNT_INVESTED_MONTHLY_COL].clip(upper=np.log1p(7000))

    if NUM_CREDIT_CARD_COL in df.columns:
        df.loc[:, NUM_CREDIT_CARD_COL] = pd.to_numeric(df[NUM_CREDIT_CARD_COL], errors='coerce').fillna(numerical_medians.get(NUM_CREDIT_CARD_COL, 0.0)).clip(lower=0)
        df.loc[:, NUM_CREDIT_CARD_COL] = np.sqrt(df[NUM_CREDIT_CARD_COL])

    if DELAY_FROM_DUE_DATE_COL in df.columns:
        df.loc[:, DELAY_FROM_DUE_DATE_COL] = df[DELAY_FROM_DUE_DATE_COL].clip(upper=DELAY_FROM_DUE_DATE_UPPER_BOUND)
        df.loc[:, DELAY_FLAG_COL] = np.where(df[DELAY_FROM_DUE_DATE_COL] > 0, 1, 0)
    else:
        df.loc[:, DELAY_FLAG_COL] = numerical_medians.get(DELAY_FLAG_COL, 0.0)

    if MONTHLY_BALANCE_COL in df.columns:
        df.loc[:, MONTHLY_BALANCE_COL] = df[MONTHLY_BALANCE_COL].clip(lower=0)

    if MONTHLY_EMI_COL in df.columns:
        df.loc[:, MONTHLY_EMI_COL] = df[MONTHLY_EMI_COL].clip(upper=MONTHLY_EMI_UPPER_BOUND)

    additional_outlier_cols = ['Num_Bank_Accounts', 'Interest_Rate', 'Annual_Income',
                               'Num_of_Delayed_Payment', 'Monthly_EMI', 'Num_Credit_Card',
                               'Changed_Credit_Limit', 'Credit_Utilization_Ratio']
    for column in additional_outlier_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(numerical_medians.get(column, 0.0))
            if column == 'Num_Bank_Accounts': df[column] = df[column].clip(upper=NUM_BANK_ACCOUNTS_UPPER_BOUND)
            elif column == 'Interest_Rate': df[column] = df[column].clip(upper=INTEREST_RATE_EDA_UPPER)
            elif column == 'Annual_Income': df[column] = df[column].clip(upper=numerical_medians.get(ANNUAL_INCOME_COL, 2000000.0) * 5)
            elif column == 'Num_of_Delayed_Payment': df[column] = df[column].clip(upper=DELAY_FROM_DUE_DATE_UPPER_BOUND)
            elif column == 'Monthly_EMI': df[column] = df[column].clip(upper=MONTHLY_EMI_UPPER_BOUND)
            elif column == 'Num_Credit_Card': df[column] = df[column].clip(upper=NUM_CREDIT_CARD_EDA_UPPER)
            elif column == 'Changed_Credit_Limit': df[column] = df[column].clip(upper=500.0)
            elif column == 'Credit_Utilization_Ratio': df[column] = df[column].clip(upper=1.0)
    print(f"After outlier handling, df.columns: {df.columns.tolist()}")
    print(f"Current df.shape: {df.shape}")

    # --- Apply EDA-based Scaling Filters ---
    if AGE_COL in df.columns: df.loc[:, AGE_COL] = df[AGE_COL].clip(upper=AGE_EDA_UPPER)
    if MONTHLY_INHAND_SALARY_COL in df.columns: df.loc[:, MONTHLY_INHAND_SALARY_COL] = df[MONTHLY_INHAND_SALARY_COL].clip(upper=MONTHLY_INHAND_SALARY_EDA_UPPER)
    if NUM_CREDIT_CARD_COL in df.columns:
        df.loc[:, NUM_CREDIT_CARD_COL] = df[NUM_CREDIT_CARD_COL].clip(lower=NUM_CREDIT_CARD_EDA_LOWER, upper=NUM_CREDIT_CARD_EDA_UPPER)
    if 'Interest_Rate' in df.columns: df.loc[:, 'Interest_Rate'] = df['Interest_Rate'].clip(upper=INTEREST_RATE_EDA_UPPER)
    if NUM_OF_LOAN_COL in df.columns: df.loc[:, NUM_OF_LOAN_COL] = df[NUM_OF_LOAN_COL].clip(upper=NUM_OF_LOAN_EDA_UPPER)
    if DELAY_FROM_DUE_DATE_COL in df.columns: df.loc[:, DELAY_FROM_DUE_DATE_COL] = df[DELAY_FROM_DUE_DATE_COL].clip(upper=DELAY_FROM_DUE_DATE_EDA_UPPER)
    if OUTSTANDING_DEBT_COL in df.columns: df.loc[:, OUTSTANDING_DEBT_COL] = df[OUTSTANDING_DEBT_COL].clip(upper=OUTSTANDING_DEBT_EDA_UPPER)
    if MONTHLY_EMI_COL in df.columns:
        df.loc[:, MONTHLY_EMI_COL] = df[MONTHLY_EMI_COL].clip(upper=min(MONTHLY_EMI_EDA_UPPER_1, MONTHLY_EMI_EDA_UPPER_2))
    if AMOUNT_INVESTED_MONTHLY_COL in df.columns:
        df.loc[:, AMOUNT_INVESTED_MONTHLY_COL] = df[AMOUNT_INVESTED_MONTHLY_COL].clip(lower=AMOUNT_INVESTED_MONTHLY_EDA_LOWER, upper=AMOUNT_INVESTED_MONTHLY_EDA_UPPER)
    print(f"After EDA-based filters, df.columns: {df.columns.tolist()}")
    print(f"Current df.shape: {df.shape}")

    # --- Feature Engineering ---
    df[FREQUENCY_COL] = numerical_medians.get(FREQUENCY_COL, 1.0)
    month_mapping_to_int = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                            'May': 5, 'June': 6, 'July': 7,
                            'August': 8, 'September': 9, 'October': 10,
                            'November': 11, 'December': 12}
    if MONTH_COL in df.columns and pd.notna(df.loc[0, MONTH_COL]):
        current_month_encoded = month_mapping_to_int.get(df.loc[0, MONTH_COL], 9)
        df[RECENCY_COL] = max(0, current_month_encoded - 9)
    else:
        df[RECENCY_COL] = numerical_medians.get(RECENCY_COL, 0.0)

    df[MONETARY_COL] = df[OUTSTANDING_DEBT_COL]
    df[MONETARY_COL] = df[MONETARY_COL].fillna(numerical_medians.get(MONETARY_COL, 0.0))

    if PAYMENT_BEHAVIOUR_COL in df.columns:
        df[PAYMENT_BEHAVIOUR_COL].replace({'!@9#%8': 'Medium_spent_Moderate_value_payments'}, inplace=True)
        df[PAYMENT_BEHAVIOUR_COL] = df[PAYMENT_BEHAVIOUR_COL].astype(str).fillna('Unknown')
    else:
        df[PAYMENT_BEHAVIOUR_COL] = 'Unknown'

    if TYPE_OF_LOAN_COL in df.columns:
        df[TYPE_OF_LOAN_COL] = df[TYPE_OF_LOAN_COL].apply(lambda x: map_loan_type(x, LOAN_TYPE_MAPPING))
        df[TYPE_OF_LOAN_COL] = df[TYPE_OF_LOAN_COL].astype(str).fillna('Unknown')
    else:
        df[TYPE_OF_LOAN_COL] = 'Unknown'

    df[TOTAL_NUM_ACCOUNTS_COL] = df[NUM_BANK_ACCOUNTS_COL] + df[NUM_CREDIT_CARD_COL]
    df[TOTAL_NUM_ACCOUNTS_COL] = df[TOTAL_NUM_ACCOUNTS_COL].replace(0, numerical_medians.get(TOTAL_NUM_ACCOUNTS_COL, 1.0))

    df[DEBT_PER_ACCOUNT_COL] = df[OUTSTANDING_DEBT_COL] / df[TOTAL_NUM_ACCOUNTS_COL]
    df[DEBT_PER_ACCOUNT_COL] = df[DEBT_PER_ACCOUNT_COL].fillna(numerical_medians.get(DEBT_PER_ACCOUNT_COL, 0.0))

    df[ANNUAL_INCOME_COL] = df[ANNUAL_INCOME_COL].replace(0, numerical_medians.get(ANNUAL_INCOME_COL, 1.0))
    df[DEBT_TO_INCOME_RATIO_COL] = df[OUTSTANDING_DEBT_COL] / df[ANNUAL_INCOME_COL]
    df[DEBT_TO_INCOME_RATIO_COL] = df[DEBT_TO_INCOME_RATIO_COL].fillna(numerical_medians.get(DEBT_TO_INCOME_RATIO_COL, 0.0))

    df[DELAYED_PAYMENTS_PER_ACCOUNT_COL] = df[NUM_OF_DELAYED_PAYMENT_COL] / df[TOTAL_NUM_ACCOUNTS_COL]
    df[DELAYED_PAYMENTS_PER_ACCOUNT_COL] = df[DELAYED_PAYMENTS_PER_ACCOUNT_COL].fillna(numerical_medians.get(DELAYED_PAYMENTS_PER_ACCOUNT_COL, 0.0))

    print(f"After Feature Engineering, df.columns: {df.columns.tolist()}")
    print(f"Current df.shape: {df.shape}")

    # --- Categorical Encoding ---
    if OCCUPATION_COL in df.columns:
        df[OCCUPATION_COL] = df[OCCUPATION_COL].astype(str).fillna('Unknown')
        df[OCCUPATION_COL] = le_occupation.transform(df[OCCUPATION_COL])

    if PAYMENT_BEHAVIOUR_COL in df.columns:
        df[PAYMENT_BEHAVIOUR_COL] = df[PAYMENT_BEHAVIOUR_COL].astype(str).fillna('Unknown')
        df[PAYMENT_BEHAVIOUR_COL] = payment_behaviour_encoder.transform(df[[PAYMENT_BEHAVIOUR_COL]])

    if TYPE_OF_LOAN_COL in df.columns:
        df[TYPE_OF_LOAN_COL] = df[TYPE_OF_LOAN_COL].astype(str).fillna('Unknown')
        df[TYPE_OF_LOAN_COL] = le_type_of_loan.transform(df[TYPE_OF_LOAN_COL])

    if PAYMENT_OF_MIN_AMOUNT_COL in df.columns:
        df[PAYMENT_OF_MIN_AMOUNT_COL] = df[PAYMENT_OF_MIN_AMOUNT_COL].astype(str).fillna('Unknown')
        df[PAYMENT_OF_MIN_AMOUNT_COL] = le_payment_min_amount.transform(df[PAYMENT_OF_MIN_AMOUNT_COL])
    print(f"After Categorical Encoding, df.columns: {df.columns.tolist()}")
    print(f"Current df.shape: {df.shape}")

    # --- Create the final DataFrame for model prediction ---
    final_processed_df = pd.DataFrame(index=df.index)

    # Use a set for quick lookup of existing columns in df
    df_cols_set = set(df.columns)

    for feature in FEATURES:
        if feature in df_cols_set:
            final_processed_df[feature] = pd.to_numeric(df[feature], errors='coerce')
            final_processed_df[feature] = final_processed_df[feature].fillna(numerical_medians.get(feature, 0.0))
        else:
            print(f"DEBUG: Feature '{feature}' from FEATURES list not found in df. Adding with default value.")
            final_processed_df[feature] = numerical_medians.get(feature, 0.0)

    # Final check for dtypes before scaling
    for col in final_processed_df.columns:
        if final_processed_df[col].dtype == 'object':
            print(f"ERROR: Column '{col}' is still object type before final numeric conversion. Attempting conversion.")
            final_processed_df[col] = pd.to_numeric(final_processed_df[col], errors='coerce').fillna(numerical_medians.get(col, 0.0))
        # Ensure it's float64, as StandardScaler typically expects floats
        final_processed_df[col] = final_processed_df[col].astype(np.float64)

    print(f"Before final scaling, final_processed_df.columns: {final_processed_df.columns.tolist()}")
    print(f"Before final scaling, final_processed_df.shape: {final_processed_df.shape}")
    print(f"Expected FEATURES list count: {len(FEATURES)}")


    # --- Numerical scaling on the *final* feature set ---
    if scaler:
        try:
            # Ensure the order is exactly as in FEATURES
            scaled_array = scaler.transform(final_processed_df[FEATURES])
            final_processed_df[FEATURES] = scaled_array
        except Exception as e:
            print(f"CRITICAL ERROR during scaling: {e}")
            print(f"Features in final_processed_df (being passed to scaler): {final_processed_df.columns.tolist()}")
            print(f"Shape of df being passed to scaler: {final_processed_df[FEATURES].shape}")
            print(f"Features expected by scaler (from config.FEATURES): {FEATURES}")
            raise

    print(f"--- DEBUG END ---")
    return final_processed_df[FEATURES]

load_all_preprocessors_and_model()