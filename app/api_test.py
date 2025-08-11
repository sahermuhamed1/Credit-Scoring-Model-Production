import requests
import json
import os
import sys

# Add the project root to the Python path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import column names from config for building test data
from config import (
    CUSTOMER_ID_COL, MONTH_COL, NAME_COL,
    AGE_COL, OCCUPATION_COL, ANNUAL_INCOME_COL,
    MONTHLY_INHAND_SALARY_COL, NUM_BANK_ACCOUNTS_COL, NUM_CREDIT_CARD_COL,
    INTEREST_RATE_COL, NUM_OF_LOAN_COL, TYPE_OF_LOAN_COL, DELAY_FROM_DUE_DATE_COL,
    NUM_OF_DELAYED_PAYMENT_COL, CHANGED_CREDIT_LIMIT_COL, OUTSTANDING_DEBT_COL,
    CREDIT_UTILIZATION_RATIO_COL, CREDIT_HISTORY_AGE_COL, PAYMENT_OF_MIN_AMOUNT_COL,
    MONTHLY_EMI_COL, AMOUNT_INVESTED_MONTHLY_COL, PAYMENT_BEHAVIOUR_COL, MONTHLY_BALANCE_COL
)

# Define the API endpoint
API_URL = "http://127.0.0.1:5000/predict"

# 1. Example Valid Test Data (fill with representative values)
# These values should mimic the data types and formats from your original dataset.
# Use values that make sense for a customer.
test_data_1 = {
    CUSTOMER_ID_COL: "C000001",
    NAME_COL: "Alice Smith",
    MONTH_COL: "January",
    AGE_COL: 35,
    OCCUPATION_COL: "Engineer",
    ANNUAL_INCOME_COL: 75000.0,
    MONTHLY_INHAND_SALARY_COL: 6000.0,
    NUM_BANK_ACCOUNTS_COL: 3,
    NUM_CREDIT_CARD_COL: 4,
    INTEREST_RATE_COL: 12.5,
    NUM_OF_LOAN_COL: 3,
    TYPE_OF_LOAN_COL: "Personal Loan",
    DELAY_FROM_DUE_DATE_COL: 2,
    NUM_OF_DELAYED_PAYMENT_COL: 0,
    CHANGED_CREDIT_LIMIT_COL: 5.0,
    OUTSTANDING_DEBT_COL: 12000.0,
    CREDIT_UTILIZATION_RATIO_COL: 0.4,
    CREDIT_HISTORY_AGE_COL: "8 Years and 3 Months",
    PAYMENT_OF_MIN_AMOUNT_COL: "Yes",
    MONTHLY_EMI_COL: 400.0,
    AMOUNT_INVESTED_MONTHLY_COL: 200.0,
    PAYMENT_BEHAVIOUR_COL: "High_spent_Medium_value_payments",
    MONTHLY_BALANCE_COL: 1500.0,
    # Include other potentially expected but dropped columns from raw input if needed,
    # though the Flask app's preprocessing should handle their absence gracefully.
    "ID": 1001,
    "SSN": "XXX-XX-1234",
    "Num_Credit_Inquiries": 2,
    "Credit_Mix": "Good"
}

# 2. Example Test Data with Missing Fields (to test robustness)
test_data_2 = {
    CUSTOMER_ID_COL: "C000002",
    NAME_COL: "Bob Johnson",
    AGE_COL: 50,
    ANNUAL_INCOME_COL: 40000.0,
    NUM_BANK_ACCOUNTS_COL: 1,
    NUM_CREDIT_CARD_COL: 1,
    OUTSTANDING_DEBT_COL: 20000.0,
    # Intentionally missing many fields to test defaults
    # Monthly_Inhand_Salary, Occupation, Type_of_Loan, etc., will use defaults
}

# 3. Example Test Data with potentially problematic values (e.g., very high debt)
test_data_3 = {
    CUSTOMER_ID_COL: "C000003",
    NAME_COL: "Charlie Brown",
    MONTH_COL: "December",
    AGE_COL: 25,
    OCCUPATION_COL: "Journalist",
    ANNUAL_INCOME_COL: 10000.0,
    MONTHLY_INHAND_SALARY_COL: 800.0,
    NUM_BANK_ACCOUNTS_COL: 1,
    NUM_CREDIT_CARD_COL: 1,
    INTEREST_RATE_COL: 30.0,
    NUM_OF_LOAN_COL: 5,
    TYPE_OF_LOAN_COL: "Student Loan",
    DELAY_FROM_DUE_DATE_COL: 60, # Significant delay
    NUM_OF_DELAYED_PAYMENT_COL: 5, # Multiple delays
    CHANGED_CREDIT_LIMIT_COL: 0.0,
    OUTSTANDING_DEBT_COL: 30000.0, # High debt for income
    CREDIT_UTILIZATION_RATIO_COL: 0.9, # High utilization
    CREDIT_HISTORY_AGE_COL: "1 Year and 0 Months", # Short history
    PAYMENT_OF_MIN_AMOUNT_COL: "No", # Not paying minimum
    MONTHLY_EMI_COL: 800.0, # High EMI
    AMOUNT_INVESTED_MONTHLY_COL: 0.0, # No investment
    PAYMENT_BEHAVIOUR_COL: "Low_spent_Small_value_payments",
    MONTHLY_BALANCE_COL: -500.0, # Negative balance
}


def test_api_endpoint(data, description=""):
    print(f"\n--- Testing API with: {description} ---")
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response_json = response.json()

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response_json, indent=2)}")

        if response.status_code == 200 and response_json.get("status") == "success":
            print(f"SUCCESS! Predicted Credit Score: {response_json.get('predicted_credit_score')}")
        else:
            print(f"FAILURE! Error: {response_json.get('message', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to Flask app at {API_URL}. Is the Flask app running?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_api_endpoint(test_data_1, "Normal Customer Data (Good Profile)")
    test_api_endpoint(test_data_2, "Customer Data with Missing Fields (Testing Defaults)")
    test_api_endpoint(test_data_3, "Customer Data with Potentially Bad Profile (Testing Edge Cases)")