# src/data_loader.py

import pandas as pd
from config import RAW_DATA_PATH_TRAIN, RAW_DATA_PATH_TEST, MONTHLY_INHAND_SALARY_COL, TOTAL_EMI_COL, MONTHLY_EMI_COL

def load_and_combine_data() -> pd.DataFrame:
    """
    Loads train and test datasets, combines them, and performs initial renames.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    try:
        train_df = pd.read_csv(RAW_DATA_PATH_TRAIN)
        test_df = pd.read_csv(RAW_DATA_PATH_TEST)
        df = pd.concat([train_df, test_df], ignore_index=True)

        # Renaming columns
        df.rename(columns={TOTAL_EMI_COL: MONTHLY_EMI_COL}, inplace=True)

        print(f"Data loaded. Combined shape: {df.shape}")
        print(f"Columns after initial rename: {df.columns.tolist()}")
        return df
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please ensure '{RAW_DATA_PATH_TRAIN}' and '{RAW_DATA_PATH_TEST}' exist.")
        raise e
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        raise e