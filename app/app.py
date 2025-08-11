# app/app.py

from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the Python path to import config and src.utils
# This assumes app.py is inside the 'app' directory which is directly under the project root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import (
    CREDIT_SCORE_CATEGORIES, FEATURES,
    CUSTOMER_ID_COL, MONTH_COL, NAME_COL,
    AGE_COL, OCCUPATION_COL, ANNUAL_INCOME_COL,
    MONTHLY_INHAND_SALARY_COL, NUM_BANK_ACCOUNTS_COL, NUM_CREDIT_CARD_COL,
    INTEREST_RATE_COL, NUM_OF_LOAN_COL, TYPE_OF_LOAN_COL, DELAY_FROM_DUE_DATE_COL,
    NUM_OF_DELAYED_PAYMENT_COL, CHANGED_CREDIT_LIMIT_COL, OUTSTANDING_DEBT_COL,LABEL_ENCODER_OCCUPATION_PATH,
    CREDIT_UTILIZATION_RATIO_COL, CREDIT_HISTORY_AGE_COL, PAYMENT_OF_MIN_AMOUNT_COL,
    MONTHLY_EMI_COL, AMOUNT_INVESTED_MONTHLY_COL, PAYMENT_BEHAVIOUR_COL, MONTHLY_BALANCE_COL
)
from src.utils import load_preprocessor # Only load utils for the inverse transform
from pipeline_utils import preprocess_single_input, rf_model, credit_score_encoder # Import preprocessors and model from pipeline_utils

app = Flask(__name__)

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests.
    Expects JSON input with customer data.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Create a dictionary with all expected input fields,
    # mapping from form names to config names.
    # Provide default values or convert to proper types where necessary.
    input_data = {
        CUSTOMER_ID_COL: data.get(CUSTOMER_ID_COL, 'C0000_new'), # Default for new customer
        NAME_COL: data.get(NAME_COL, 'Unknown'),
        MONTH_COL: data.get(MONTH_COL, 'January'), # Default or last month in data
        AGE_COL: int(data.get(AGE_COL, 30)),
        OCCUPATION_COL: data.get(OCCUPATION_COL, 'Unknown'),
        ANNUAL_INCOME_COL: float(data.get(ANNUAL_INCOME_COL, 50000)),
        MONTHLY_INHAND_SALARY_COL: float(data.get(MONTHLY_INHAND_SALARY_COL, 4000)),
        NUM_BANK_ACCOUNTS_COL: int(data.get(NUM_BANK_ACCOUNTS_COL, 2)),
        NUM_CREDIT_CARD_COL: int(data.get(NUM_CREDIT_CARD_COL, 3)),
        INTEREST_RATE_COL: float(data.get(INTEREST_RATE_COL, 15.0)),
        NUM_OF_LOAN_COL: int(data.get(NUM_OF_LOAN_COL, 2)),
        TYPE_OF_LOAN_COL: data.get(TYPE_OF_LOAN_COL, 'Personal'),
        DELAY_FROM_DUE_DATE_COL: int(data.get(DELAY_FROM_DUE_DATE_COL, 0)),
        NUM_OF_DELAYED_PAYMENT_COL: int(data.get(NUM_OF_DELAYED_PAYMENT_COL, 0)),
        CHANGED_CREDIT_LIMIT_COL: float(data.get(CHANGED_CREDIT_LIMIT_COL, 10.0)),
        OUTSTANDING_DEBT_COL: float(data.get(OUTSTANDING_DEBT_COL, 5000.0)),
        CREDIT_UTILIZATION_RATIO_COL: float(data.get(CREDIT_UTILIZATION_RATIO_COL, 0.3)),
        CREDIT_HISTORY_AGE_COL: data.get(CREDIT_HISTORY_AGE_COL, '5 Years and 5 Months'),
        PAYMENT_OF_MIN_AMOUNT_COL: data.get(PAYMENT_OF_MIN_AMOUNT_COL, 'Yes'),
        MONTHLY_EMI_COL: float(data.get(MONTHLY_EMI_COL, 200.0)),
        AMOUNT_INVESTED_MONTHLY_COL: float(data.get(AMOUNT_INVESTED_MONTHLY_COL, 100.0)),
        PAYMENT_BEHAVIOUR_COL: data.get(PAYMENT_BEHAVIOUR_COL, 'Low_spent_Small_value_payments'),
        MONTHLY_BALANCE_COL: float(data.get(MONTHLY_BALANCE_COL, 500.0)),
        # Placeholder for columns that are dropped by the pipeline but might be in raw data.
        # They will be ignored by preprocess_single_input.
        'ID': data.get('ID', 0),
        'SSN': data.get('SSN', 'XXX-XX-XXXX'),
        'Num_Credit_Inquiries': data.get('Num_Credit_Inquiries', 5),
        'Credit_Mix': data.get('Credit_Mix', 'Good')
    }

    try:
        # Preprocess the input data
        processed_data = preprocess_single_input(input_data)

        # Ensure processed_data contains the exact features the model expects
        # and in the correct order. The `FEATURES` list from config should define this.
        # If any features are missing, ensure they are added with a default value (e.g., 0)
        # to match the model's input shape.
        final_input_for_model = processed_data[FEATURES]

        # Make prediction
        prediction_numeric = rf_model.predict(final_input_for_model)[0]

        # Inverse transform the numeric prediction to original label
        prediction_label = credit_score_encoder.inverse_transform([[np.round(prediction_numeric)]])[0][0]

        return jsonify({
            "status": "success",
            "predicted_credit_score": prediction_label
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Ensure model and preprocessors are loaded when the app starts
    # This call is already in pipeline_utils, but can be explicitly called here too for clarity
    # if you remove it from pipeline_utils.
    # pipeline_utils.load_all_preprocessors_and_model() # Already called when pipeline_utils is imported

    app.run(debug=True) # debug=True for development, set to False for production