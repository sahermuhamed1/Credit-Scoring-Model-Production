
# config.py

import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # This needs adjustment if config.py is inside src

# Correct BASE_DIR if config.py is in the root and src is a subfolder
# Assuming config.py is in the root 'credit_scoring_project' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
RAW_DATA_PATH_TRAIN = os.path.join(BASE_DIR, 'dataset', 'data', 'raw', 'train.csv')
RAW_DATA_PATH_TEST = os.path.join(BASE_DIR, 'dataset', 'data', 'raw', 'test.csv')

# Model and preprocessor save paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RF_MODEL_PATH = os.path.join(MODELS_DIR, 'rf_classifier_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
ORDINAL_ENCODER_PATH = os.path.join(MODELS_DIR, 'ordinal_encoder.pkl')
LABEL_ENCODER_NUM_OF_LOAN_PATH = os.path.join(MODELS_DIR, 'label_encoder_num_of_loan.pkl')
LABEL_ENCODER_PAYMENT_MIN_AMOUNT_PATH = os.path.join(MODELS_DIR, 'label_encoder_payment_min_amount.pkl')
LABEL_ENCODER_TYPE_OF_LOAN_PATH = os.path.join(MODELS_DIR, 'label_encoder_type_of_loan.pkl')
LABEL_ENCODER_OCCUPATION_PATH = os.path.join(MODELS_DIR, 'label_encoder_occupation.pkl')
PAYMENT_BEHAVIOUR_ENCODER_PATH = os.path.join(MODELS_DIR, 'payment_behaviour_encoder.pkl') # New encoder path

# Column names
TARGET_COLUMN = 'Credit_Score'
MONTHLY_INHAND_SALARY_COL = 'Monthly_Inhand_Salary'
TOTAL_EMI_COL = 'Total_EMI_per_month' # Original name
MONTHLY_EMI_COL = 'Monthly_EMI'       # Renamed name
OUTSTANDING_DEBT_COL = 'Outstanding_Debt'
CREDIT_HISTORY_AGE_COL = 'Credit_History_Age'
PAYMENT_OF_MIN_AMOUNT_COL = 'Payment_of_Min_Amount'
TYPE_OF_LOAN_COL = 'Type_of_Loan'
NUM_OF_LOAN_COL = 'Num_of_Loan'
AGE_COL = 'Age'
ANNUAL_INCOME_COL = 'Annual_Income'
NUM_BANK_ACCOUNTS_COL = 'Num_Bank_Accounts'
NUM_CREDIT_CARD_COL = 'Num_Credit_Card'
DELAY_FROM_DUE_DATE_COL = 'Delay_from_due_date'
NUM_OF_DELAYED_PAYMENT_COL = 'Num_of_Delayed_Payment'
AMOUNT_INVESTED_MONTHLY_COL = 'Amount_invested_monthly'
MONTHLY_BALANCE_COL = 'Monthly_Balance'
NAME_COL = 'Name'
CUSTOMER_ID_COL = 'Customer_ID'
MONTH_COL = 'Month'
PAYMENT_BEHAVIOUR_COL = 'Payment_Behaviour'
OCCUPATION_COL = 'Occupation'
CREDIT_UTILIZATION_RATIO_COL = 'Credit_Utilization_Ratio'
CHANGED_CREDIT_LIMIT_COL = 'Changed_Credit_Limit'
INTEREST_RATE_COL = 'Interest_Rate'

# Derived feature names
CREDIT_HISTORY_AGE_MONTHS_COL = 'Credit_History_Age_Months'
DELAY_FLAG_COL = 'Delay_Flag'
FREQUENCY_COL = 'Frequency'
RECENCY_COL = 'Recency'
MONETARY_COL = 'Monetary'
TOTAL_NUM_ACCOUNTS_COL = 'Total_Num_Accounts'
DEBT_PER_ACCOUNT_COL = 'Debt_Per_Account'
DEBT_TO_INCOME_RATIO_COL = 'Debt_to_Income_Ratio'
DELAYED_PAYMENTS_PER_ACCOUNT_COL = 'Delayed_Payments_Per_Account'


# Columns to drop initially
COLUMNS_TO_DROP = ['ID', 'Num_Credit_Inquiries', 'Credit_Mix']

# Numerical columns for KNN Imputer and scaling
NUMERICAL_IMPUTE_COLS = [
    MONTHLY_INHAND_SALARY_COL,
    NUM_OF_DELAYED_PAYMENT_COL,
    AMOUNT_INVESTED_MONTHLY_COL,
    MONTHLY_BALANCE_COL
]

# Columns to clip to positive values
POSITIVE_CLIP_COLS = [
    NUM_BANK_ACCOUNTS_COL,
    NUM_CREDIT_CARD_COL,
    NUM_OF_LOAN_COL,
    DELAY_FROM_DUE_DATE_COL,
    NUM_OF_DELAYED_PAYMENT_COL,
    MONTHLY_EMI_COL,
    AMOUNT_INVESTED_MONTHLY_COL
]

# Outlier handling thresholds
AGE_UPPER_BOUND = 80
NUM_BANK_ACCOUNTS_UPPER_BOUND = 10
OUTSTANDING_DEBT_UPPER_BOUND = 4500
AMOUNT_INVESTED_MONTHLY_QUANTILE_CLIP = 0.99
DELAY_FROM_DUE_DATE_UPPER_BOUND = 90
MONTHLY_EMI_UPPER_BOUND = 550

# EDA based scaling thresholds
AGE_EDA_UPPER = 55
MONTHLY_INHAND_SALARY_EDA_UPPER = 8000
NUM_CREDIT_CARD_EDA_LOWER = 1
NUM_CREDIT_CARD_EDA_UPPER = 3
INTEREST_RATE_EDA_UPPER = 50.00
NUM_OF_LOAN_EDA_UPPER = 12
DELAY_FROM_DUE_DATE_EDA_UPPER = 50
OUTSTANDING_DEBT_EDA_UPPER = 3700
MONTHLY_EMI_EDA_UPPER_1 = 350
AMOUNT_INVESTED_MONTHLY_EDA_LOWER = 2
AMOUNT_INVESTED_MONTHLY_EDA_UPPER = 7
MONTHLY_EMI_EDA_UPPER_2 = 250


# Loan type mapping for cleaning
LOAN_TYPE_MAPPING = {
    'student': 'Student',
    'not specified': 'Unknown',
    'unknown': 'Unknown',
    'personal': 'Personal',
    'home equity': 'Home Equity',
    'mortgage': 'Mortgage',
    'debt consolidation': 'Debt Consolidation',
    'credit-builder': 'Credit-Builder',
    'auto': 'Auto Loan',
    'payday': 'Payday'
}

# Encoding categories
CREDIT_SCORE_CATEGORIES = ['Poor', 'Standard', 'Good']
PAYMENT_BEHAVIOUR_CATEGORIES = [
    'Low_spent_Small_value_payments',
    'High_spent_Medium_value_payments',
    'Low_spent_Medium_value_payments',
    'High_spent_Large_value_payments',
    'High_spent_Small_value_payments',
    'Low_spent_Large_value_payments',
    'Medium_spent_Moderate_value_payments'
]

# Features for model training
FEATURES = [
    ANNUAL_INCOME_COL,                  # 1
    NUM_BANK_ACCOUNTS_COL,              # 2
    NUM_CREDIT_CARD_COL,                # 3
    INTEREST_RATE_COL,                  # 4
    NUM_OF_LOAN_COL,                    # 5
    DELAY_FROM_DUE_DATE_COL,            # 6
    NUM_OF_DELAYED_PAYMENT_COL,         # 7
    CHANGED_CREDIT_LIMIT_COL,           # 8
    OUTSTANDING_DEBT_COL,               # 9
    CREDIT_UTILIZATION_RATIO_COL,       # 10
    MONTHLY_EMI_COL,                    # 11
    CREDIT_HISTORY_AGE_MONTHS_COL,      # 12
    TOTAL_NUM_ACCOUNTS_COL,             # 13
    DEBT_PER_ACCOUNT_COL,               # 14
    DEBT_TO_INCOME_RATIO_COL,           # 15
    DELAYED_PAYMENTS_PER_ACCOUNT_COL,   # 16
    DELAY_FLAG_COL,                     # 17
    FREQUENCY_COL,                      # 18
    RECENCY_COL,                        # 19
    MONETARY_COL,                       # 20
    MONTHLY_INHAND_SALARY_COL,          # 21
    AMOUNT_INVESTED_MONTHLY_COL,        # 22
    MONTHLY_BALANCE_COL                 # 23
]

# Model parameters
SMOTE_RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5
TRAIN_TEST_SPLIT_RATIO = 0.1
MODEL_RANDOM_STATE = 77
RF_N_ESTIMATORS = 500
XGB_N_ESTIMATORS = 3500
XGB_LEARNING_RATE = 0.05
CATBOOST_ITERATIONS = 1000
CATBOOST_LEARNING_RATE = 0.1
CATBOOST_DEPTH = 6
CATBOOST_RANDOM_SEED = 42

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)