# Credit Scoring Project - Production Inference API

## Overview

This project provides an end-to-end Machine Learning pipeline for predicting a customer's credit score (Poor, Standard, Good) based on various financial and behavioral data. It includes data preprocessing, model training, evaluation, and deployment as a Flask API.

## Project Setup
Prerequisites: Python 3.8+ and pip.
```Bash
# 1. Clone the repository
git clone https://github.com/your-username/Credit-Scoring-Model-Production.git
cd Credit-Scoring-Model-Production

# 2. Create and activate a virtual environment
python -m venv venv

# 3. Install dependencies
pip install -r requirements.txt
```

## Project Structure
credit_scoring_project/
``` markdown
├── app/                          # Flask Web Application for API deployment
│   ├── app.py                    # Main Flask application file
|   ├── api_test.py               # Test script for API endpoints
│   ├── pipeline_utils.py         # Adapted preprocessing logic for inference
│   ├── templates/                # HTML templates for the web interface (if used)
│   │   └── index.html            # Simple form for manual testing
│   └── static/                   # Static assets (CSS, JS)
│       └── style.css
├── config.py                     # Centralized configuration for paths, column names, thresholds, etc.
├── dataset/                      # Data storage
│   └── data/
│       └── raw/
│           ├── train.csv         # Raw training data
│           └── test.csv          # Raw test data (for prediction)
├── models/                       # Trained model and preprocessor artifacts (ignored by Git)
│   ├── rf_classifier_model.pkl   # Trained Random Forest model
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── ordinal_encoder.pkl       # Fitted OrdinalEncoder for Credit_Score
│   ├── payment_behaviour_encoder.pkl # Fitted OrdinalEncoder for Payment_Behaviour
│   ├── label_encoder_occupation.pkl  # Fitted LabelEncoder for Occupation
│   ├── label_encoder_type_of_loan.pkl # Fitted LabelEncoder for Type_of_Loan
│   └── label_encoder_payment_min_amount.pkl # Fitted LabelEncoder for Payment_of_Min_Amount
├── src/                          # Source code for the ML pipeline components
│   ├── data_cleaner.py           # Functions for data cleaning and preprocessing
│   ├── data_loader.py            # Functions for loading and combining raw data
│   ├── feature_engineer.py       # Functions for creating new features and scaling
│   ├── model_trainer.py          # Functions for model training, evaluation, and handling imbalances
│   └── utils.py                  # Utility functions (file I/O, parsing, plotting)
├── main.py                       # Orchestrates the entire ML pipeline (training, evaluation)
├── test_api.py                   # Script to test the deployed Flask API
├── .gitignore                    # Specifies files/directories to be ignored by Git
├── requirements.txt              # List of Python dependencies
└── README.md                     # Project README (this file)
```

## How to Train the Model Locally

This step executes the entire ML pipeline: data loading, cleaning, feature engineering, model training, and evaluation. It's crucial because it generates and saves the rf_classifier_model.pkl and all scaler.pkl/*_encoder.pkl files in the models/ directory, which the Flask API relies on.
*Run from the project root directory:*
```Bash
python main.py
```
Allow this script to complete running. You'll see progress messages and evaluation results in your terminal.


## How to Run the Flask Application
The Flask application (app/app.py) serves as your credit scoring prediction API.
Ensure the model and preprocessors are trained and saved (see previous step).
*Navigate to the app directory:*
```Bash
cd app
```
*Start the Flask development server:*
```Bash
python app.py
```
The API will typically be accessible at http://127.0.0.1:5000/. Keep this terminal window open as long as you want the API to be running.

## How to Use the Flask API
Your Flask application exposes a single prediction endpoint:
- URL: http://127.0.0.1:5000/predict
- Method: POST
- Content-Type: application/json
The API expects a JSON object in the request body containing customer features.
**Example Request Body (JSON):**
```json
{
    "Customer_ID": "C000001",
    "Name": "Alice Smith",
    "Month": "January",
    "Age": 35,
    "Occupation": "Engineer",
    "Annual_Income": 75000.0,
    "Monthly_Inhand_Salary": 6000.0,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 4,
    "Interest_Rate": 12.5,
    "Num_of_Loan": 3,
    "Type_of_Loan": "Personal Loan",
    "Delay_from_due_date": 2,
    "Num_of_Delayed_Payment": 0,
    "Changed_Credit_Limit": 5.0,
    "Outstanding_Debt": 12000.0,
    "Credit_Utilization_Ratio": 0.4,
    "Credit_History_Age": "8 Years and 3 Months",
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 400.0,
    "Amount_invested_monthly": 200.0,
    "Payment_Behaviour": "High_spent_Medium_value_payments",
    "Monthly_Balance": 1500.0
}
```
**Example Response (JSON):**
```json
{
    "status": "success",
    "predicted_credit_score": "Standard"
}
```

## How to Test the API
A convenient script test_api.py is included to demonstrate API interaction.
1. Ensure the Flask API is running in one terminal.
2. Open a NEW terminal window.
3. **Navigate to the project root directory:**
```Bash
cd Credit-Scoring-Model-Production
```
4. **Run the test script:**
```Bash
python test_api.py
```
This script will send sample requests to your running API and print the responses to the console.

## Contanct Information
For any questions or issues, please contact me at sahermuhamed176@gmail.com. I'll be happy to assist.

