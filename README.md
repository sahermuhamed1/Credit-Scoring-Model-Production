# Credit Scoring Model - Production Inference API

A production-ready Flask API for credit scoring classification using a pre-trained Random Forest model.

## Project Structure

```
├── app.py                  # Flask API entry point
├── model/
│   ├── __init__.py
│   ├── preprocessing.py    # Data preprocessing functions
│   └── prediction.py       # Model loading and prediction
├── models/                 # Pretrained artifacts
│   ├── rf_classifier_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── ordinal_encoder.pkl
├── requirements.txt
├── README.md
└── dataset/
    └── data/raw/          # Training data
```

## Quick Start

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the API

```bash
python app.py
```

The API will start on `http://127.0.0.1:5000`

## API Usage

### Predict Credit Score

**Endpoint:** `POST /predict`

**Content-Type:** `application/json`

**Example Request:**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{
       "Annual_Income": 50000,
       "Num_Bank_Accounts": 2,
       "Num_Credit_Card": 1,
       "Interest_Rate": 15.5,
       "Num_of_Loan": 1,
       "Delay_from_due_date": 0,
       "Num_of_Delayed_Payment": 0,
       "Changed_Credit_Limit": 1000,
       "Outstanding_Debt": 1500,
       "Credit_Utilization_Ratio": 0.3,
       "Monthly_EMI": 200,
       "Credit_History_Age": "2 Years and 6 Months",
       "Monthly_Inhand_Salary": 4000,
       "Amount_invested_monthly": 500,
       "Monthly_Balance": 1000
     }' \
     http://127.0.0.1:5000/predict
```

**Example Response:**
```json
{
  "predicted_class": "Standard",
  "probability": 0.85,
  "all_probabilities": {
    "Poor": 0.05,
    "Standard": 0.85,
    "Good": 0.10
  }
}
```

### Health Check

**Endpoint:** `GET /health`

Returns the API health status.

## Features

- **Preprocessing**: Handles missing values, outliers, and feature engineering
- **Scaling**: Applies StandardScaler transformation
- **Encoding**: Handles categorical variables with saved encoders
- **Error Handling**: Validates input and provides meaningful error messages
- **Logging**: Minimal logging for debugging

## Model Information

- **Algorithm**: Random Forest Classifier
- **Classes**: Poor, Standard, Good
- **Accuracy**: ~90% on test set
- **Features**: 20 engineered features including RFM metrics

## Development

### Adding New Features

1. Update `preprocessing.py` with new preprocessing steps
2. Retrain the model and save new artifacts to `models/`
3. Update API documentation

### Testing

Run the API locally and test with sample data:
```bash
python -m pytest tests/  # if tests are added
