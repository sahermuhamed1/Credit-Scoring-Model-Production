# src/data_cleaner.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from src.utils import parse_years_and_months, plot_boxplots
from config import (
    COLUMNS_TO_DROP, NAME_COL, TYPE_OF_LOAN_COL, CREDIT_HISTORY_AGE_COL,
    OUTSTANDING_DEBT_COL, ANNUAL_INCOME_COL, AGE_COL, NUM_OF_LOAN_COL,
    NUMERICAL_IMPUTE_COLS, POSITIVE_CLIP_COLS,NUM_BANK_ACCOUNTS_COL,
    NUM_CREDIT_CARD_COL, DELAY_FROM_DUE_DATE_COL,
    AGE_UPPER_BOUND, NUM_BANK_ACCOUNTS_UPPER_BOUND, OUTSTANDING_DEBT_UPPER_BOUND,
    AMOUNT_INVESTED_MONTHLY_COL, MONTHLY_BALANCE_COL, MONTHLY_EMI_COL,
    NUM_CREDIT_CARD_COL, DELAY_FROM_DUE_DATE_COL, 
    AMOUNT_INVESTED_MONTHLY_QUANTILE_CLIP,DELAY_FROM_DUE_DATE_UPPER_BOUND,
    AGE_EDA_UPPER, MONTHLY_INHAND_SALARY_EDA_UPPER, MONTHLY_EMI_UPPER_BOUND,
    NUM_CREDIT_CARD_EDA_LOWER, NUM_CREDIT_CARD_EDA_UPPER,
    INTEREST_RATE_EDA_UPPER, NUM_OF_LOAN_EDA_UPPER,
    DELAY_FROM_DUE_DATE_EDA_UPPER, OUTSTANDING_DEBT_EDA_UPPER,
    MONTHLY_EMI_EDA_UPPER_1, AMOUNT_INVESTED_MONTHLY_EDA_LOWER,
    AMOUNT_INVESTED_MONTHLY_EDA_UPPER, MONTHLY_EMI_EDA_UPPER_2
)

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame.
    Fills categorical NaNs and imputes numerical NaNs using KNNImputer.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    print("\n--- Cleaning Missing Values ---")

    # Drop unnecessary columns
    df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')
    print(f"Dropped columns: {COLUMNS_TO_DROP}. Current columns: {df.columns.tolist()}")

    # 1. Fill missing categorical values
    df[NAME_COL] = df[NAME_COL].fillna('Unknown')
    df[TYPE_OF_LOAN_COL] = df[TYPE_OF_LOAN_COL].fillna('Unknown')
    df[CREDIT_HISTORY_AGE_COL] = df[CREDIT_HISTORY_AGE_COL].fillna('Unknown')
    print("Filled categorical missing values with 'Unknown'.")

    # 2. Clean and convert numerical columns for KNN Imputation
    temp_df_for_imputation = df[NUMERICAL_IMPUTE_COLS].astype(str).copy()
    for col in NUMERICAL_IMPUTE_COLS:
        temp_df_for_imputation[col] = temp_df_for_imputation[col].replace(r'[^0-9.-]', '', regex=True)
        temp_df_for_imputation[col] = pd.to_numeric(temp_df_for_imputation[col], errors='coerce')

    imputer = KNNImputer(n_neighbors=5)
    df[NUMERICAL_IMPUTE_COLS] = imputer.fit_transform(temp_df_for_imputation)
    print("Imputed numerical missing values using KNNImputer.")

    # 3. Clean and convert Annual_Income
    df.loc[:, ANNUAL_INCOME_COL] = df[ANNUAL_INCOME_COL].astype(str).str.replace(r'[^0-9.]', '', regex=True)
    df.loc[:, ANNUAL_INCOME_COL] = pd.to_numeric(df[ANNUAL_INCOME_COL], errors='coerce').fillna(0)
    print(f"Cleaned and converted '{ANNUAL_INCOME_COL}'.")

    # 4. Clean Age and Num_of_Loan - REVISED
    def safe_extract_and_impute_int(series, col_name):
        # Convert the series to string, extract digits, then convert to numeric, coercing errors to NaN
        extracted_numeric = pd.to_numeric(series.astype(str).str.extract(r'(\d+)').iloc[:, 0], errors='coerce')

        # Calculate median from the *numeric-coerced* series.
        # If the series is entirely NaN after coercion, median() will return NaN.
        median_val = extracted_numeric.median()

        # If median_val is NaN (meaning all values were non-numeric/missing), use a hardcoded default (e.g., 0)
        # or consider if a more complex imputation strategy is needed for these specific columns.
        if pd.isna(median_val):
            print(f"WARNING: '{col_name}' column became all NaN after initial digit extraction. Defaulting fill value to 0.")
            median_val = 0 # Fallback if no numeric values could be extracted at all

        # Fill NaNs in the extracted_numeric series with the calculated median_val
        # Then convert to integer type
        return extracted_numeric.fillna(median_val).astype(int)

    # Apply the revised function
    df.loc[:, AGE_COL] = safe_extract_and_impute_int(df[AGE_COL], AGE_COL)
    df.loc[:, NUM_OF_LOAN_COL] = safe_extract_and_impute_int(df[NUM_OF_LOAN_COL], NUM_OF_LOAN_COL)

    print(f"Cleaned '{AGE_COL}' and '{NUM_OF_LOAN_COL}'.")
    print(f"DEBUG: {AGE_COL} min/max after cleaning: {df[AGE_COL].min()}/{df[AGE_COL].max()}")
    print(f"DEBUG: {NUM_OF_LOAN_COL} min/max after cleaning: {df[NUM_OF_LOAN_COL].min()}/{df[NUM_OF_LOAN_COL].max()}")


    # 5. Clean Outstanding_Debt (strip leading/trailing underscores and convert)
    df.loc[:, OUTSTANDING_DEBT_COL] = df[OUTSTANDING_DEBT_COL].astype(str).str.strip().str.strip('_')
    df.loc[:, OUTSTANDING_DEBT_COL] = pd.to_numeric(df[OUTSTANDING_DEBT_COL], errors='coerce').fillna(0)
    print(f"Cleaned and converted '{OUTSTANDING_DEBT_COL}'.")

    print("\nMissing values after initial cleaning:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    return df

def perform_format_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs format conversions and initial clipping for numerical columns.
    ... (rest of the function is the same as previous correction) ...
    """
    print("\n--- Performing Format Conversions ---")

    df['Credit_History_Age_Months'] = df[CREDIT_HISTORY_AGE_COL].apply(parse_years_and_months)
    print(f"Converted '{CREDIT_HISTORY_AGE_COL}' to '{'Credit_History_Age_Months'}'.")

    df.loc[:, AGE_COL] = df[AGE_COL].clip(lower=14, upper=AGE_UPPER_BOUND)
    df.loc[:, MONTHLY_BALANCE_COL] = df[MONTHLY_BALANCE_COL].clip(lower=-1e5, upper=1e5)
    print(f"Clipped '{AGE_COL}' and '{MONTHLY_BALANCE_COL}'.")

    cols_to_process_for_clip = [col for col in POSITIVE_CLIP_COLS if col in df.columns]

    initial_rows = df.shape[0]

    for column in cols_to_process_for_clip:
        numeric_col = pd.to_numeric(df[column], errors='coerce')

        if not numeric_col.empty and not numeric_col.isnull().all():
            percentile_threshold = 0.98
            q_val = numeric_col.quantile(percentile_threshold)
            print(f"DEBUG: {column} 98th percentile: {q_val}")
            if pd.notnull(q_val) and q_val > 0:
                df = df[df[column] <= q_val]
            else:
                print(f"WARNING: Skipping {column} clipping as 98th percentile is not positive or NaN: {q_val}")
        else:
            print(f"WARNING: Skipping {column} clipping as column is empty or all NaN: {numeric_col.isnull().all()}")

    print(f"Removed {initial_rows - df.shape[0]} rows by clipping top 2% for positive columns.")

    print("\nDataFrame info after format conversions:")
    df.info()
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles outliers using various clipping and transformation techniques.
    ... (rest of the function is the same as previous correction) ...
    """
    print("\n--- Handling Outliers ---")

    if df.empty:
        print("DataFrame is empty after previous steps. Skipping outlier handling.")
        return df

    print("Boxplots before outlier handling:")
    plot_boxplots(df.copy(), title="Boxplots Before Outlier Handling")

    df.loc[:, AGE_COL] = df[AGE_COL].apply(lambda x: AGE_UPPER_BOUND if x >= AGE_UPPER_BOUND else x)

    df.loc[:, NUM_BANK_ACCOUNTS_COL] = df[NUM_BANK_ACCOUNTS_COL].clip(upper=NUM_BANK_ACCOUNTS_UPPER_BOUND)

    df.loc[:, OUTSTANDING_DEBT_COL] = df[OUTSTANDING_DEBT_COL].clip(upper=OUTSTANDING_DEBT_UPPER_BOUND)

    if AMOUNT_INVESTED_MONTHLY_COL in df.columns and not df[AMOUNT_INVESTED_MONTHLY_COL].empty:
        df.loc[:, AMOUNT_INVESTED_MONTHLY_COL] = df[AMOUNT_INVESTED_MONTHLY_COL].apply(lambda x: x if x > 0 else 1e-6)
        df.loc[:, AMOUNT_INVESTED_MONTHLY_COL] = np.log1p(df[AMOUNT_INVESTED_MONTHLY_COL])
        if not df[AMOUNT_INVESTED_MONTHLY_COL].empty:
            df.loc[:, AMOUNT_INVESTED_MONTHLY_COL] = df[AMOUNT_INVESTED_MONTHLY_COL].clip(upper=df[AMOUNT_INVESTED_MONTHLY_COL].quantile(AMOUNT_INVESTED_MONTHLY_QUANTILE_CLIP))

    if NUM_CREDIT_CARD_COL in df.columns and not df[NUM_CREDIT_CARD_COL].empty:
        df.loc[:, NUM_CREDIT_CARD_COL] = np.sqrt(df[NUM_CREDIT_CARD_COL])

    if DELAY_FROM_DUE_DATE_COL in df.columns and not df[DELAY_FROM_DUE_DATE_COL].empty:
        df.loc[:, DELAY_FROM_DUE_DATE_COL] = df[DELAY_FROM_DUE_DATE_COL].clip(upper=DELAY_FROM_DUE_DATE_UPPER_BOUND)
        df.loc[:, 'Delay_Flag'] = np.where(df[DELAY_FROM_DUE_DATE_COL] > 0, 1, 0)

    if MONTHLY_BALANCE_COL in df.columns and not df[MONTHLY_BALANCE_COL].empty:
        df.loc[:, MONTHLY_BALANCE_COL] = df[MONTHLY_BALANCE_COL].clip(lower=0)

    if MONTHLY_EMI_COL in df.columns and not df[MONTHLY_EMI_COL].empty:
        df.loc[:, MONTHLY_EMI_COL] = df[MONTHLY_EMI_COL].clip(upper=MONTHLY_EMI_UPPER_BOUND)

    print("Descriptive statistics after initial outlier handling:")
    if not df.empty:
        print(df.describe().T)
    else:
        print("DataFrame is empty, no statistics to show.")

    additional_outlier_cols = ['Num_Bank_Accounts', 'Interest_Rate', 'Annual_Income',
                               'Num_of_Delayed_Payment', 'Monthly_EMI', 'Num_Credit_Card']
    cols_to_process_additional = [col for col in additional_outlier_cols if col in df.columns]

    initial_rows = df.shape[0]
    for column in cols_to_process_additional:
        numeric_col = pd.to_numeric(df[column], errors='coerce')
        if not numeric_col.empty and not numeric_col.isnull().all():
            percentile_threshold = 0.98
            q_val = numeric_col.quantile(percentile_threshold)
            print(f"DEBUG (Additional): {column} 98th percentile: {q_val}")
            if pd.notnull(q_val):
                df = df[df[column] <= q_val]
            else:
                print(f"WARNING: Skipping additional {column} clipping as 98th percentile is NaN: {q_val}")
        else:
            print(f"WARNING: Skipping additional {column} clipping as column is empty or all NaN: {numeric_col.isnull().all()}")

    print(f"Removed {initial_rows - df.shape[0]} rows by additional 2% percentile clipping.")

    print("\nDescriptive statistics after additional outlier handling:")
    if not df.empty:
        print(df.describe().T)
    else:
        print("DataFrame is empty, no statistics to show.")

    print("Boxplots after outlier handling:")
    if not df.empty:
        plot_boxplots(df.copy(), title="Boxplots After Outlier Handling")
    else:
        print("DataFrame is empty, cannot plot boxplots.")
    return df

def apply_eda_based_scaling_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies filters based on insights from EDA to remove extreme values.
    ... (rest of the function is the same as previous correction) ...
    """
    print("\n--- Applying EDA-based Scaling Filters ---")
    initial_rows = df.shape[0]

    if df.empty:
        print("DataFrame is empty after previous steps. Skipping EDA-based filters.")
        return df

    if AGE_COL in df.columns and not df[AGE_COL].empty:
        df = df[df[AGE_COL] <= AGE_EDA_UPPER]
    if 'Monthly_Inhand_Salary' in df.columns and not df['Monthly_Inhand_Salary'].empty:
        df = df[df['Monthly_Inhand_Salary'] <= MONTHLY_INHAND_SALARY_EDA_UPPER]
    if NUM_CREDIT_CARD_COL in df.columns and not df[NUM_CREDIT_CARD_COL].empty:
        df = df[(df[NUM_CREDIT_CARD_COL] >= NUM_CREDIT_CARD_EDA_LOWER) & (df[NUM_CREDIT_CARD_COL] <= NUM_CREDIT_CARD_EDA_UPPER)]
    if 'Interest_Rate' in df.columns and not df['Interest_Rate'].empty:
        df = df[df['Interest_Rate'] <= INTEREST_RATE_EDA_UPPER]
    if NUM_OF_LOAN_COL in df.columns and not df[NUM_OF_LOAN_COL].empty:
        df = df[df[NUM_OF_LOAN_COL] <= NUM_OF_LOAN_EDA_UPPER]
    if DELAY_FROM_DUE_DATE_COL in df.columns and not df[DELAY_FROM_DUE_DATE_COL].empty:
        df = df[df[DELAY_FROM_DUE_DATE_COL] <= DELAY_FROM_DUE_DATE_EDA_UPPER]
    if OUTSTANDING_DEBT_COL in df.columns and not df[OUTSTANDING_DEBT_COL].empty:
        df = df[df[OUTSTANDING_DEBT_COL] <= OUTSTANDING_DEBT_EDA_UPPER]
    if MONTHLY_EMI_COL in df.columns and not df[MONTHLY_EMI_COL].empty:
        df = df[df[MONTHLY_EMI_COL] <= MONTHLY_EMI_EDA_UPPER_1]
    if AMOUNT_INVESTED_MONTHLY_COL in df.columns and not df[AMOUNT_INVESTED_MONTHLY_COL].empty:
        df = df[(df[AMOUNT_INVESTED_MONTHLY_COL] >= AMOUNT_INVESTED_MONTHLY_EDA_LOWER) & (df[AMOUNT_INVESTED_MONTHLY_COL] <= AMOUNT_INVESTED_MONTHLY_EDA_UPPER)]
    if MONTHLY_EMI_COL in df.columns and not df[MONTHLY_EMI_COL].empty:
        df = df[df[MONTHLY_EMI_COL] <= MONTHLY_EMI_EDA_UPPER_2]

    print(f"Removed {initial_rows - df.shape[0]} rows by applying EDA-based filters.")
    return df