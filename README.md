# Credit Scoring Model Production

This project provides a machine learning solution for credit scoring, focusing on the deployment of tree-based models using Flask and FastAPI.

## Model Usage

- **Tree Models**: The project utilizes tree-based algorithms (e.g., Decision Trees, Random Forests, Gradient Boosting) for predicting creditworthiness. These models are chosen for their interpretability and strong performance on tabular financial data.

## Evaluation Metrics

- **Precision/Recall**: The model evaluation emphasizes both precision and recall. 
  - **Precision** measures the proportion of predicted positives that are actually positive.
  - **Recall** measures the proportion of actual positives that are correctly identified.

## Business Context: Importance of False Negatives

- In loan approval, **False Negatives** (predicting a good applicant as bad) are critical. They represent lost business opportunities, as potentially creditworthy customers are denied loans. Minimizing false negatives is important to maximize revenue while managing risk.

## Credit Score Classification

- The model classifies applicants into three categories:
  - **Poor**: High risk of default.
  - **Standard**: Moderate risk.
  - **Good**: Low risk, likely to repay.

## Deployment

- The trained models are deployed as REST APIs using:
  - **Flask**: For lightweight, quick deployments.
  - **FastAPI**: For high-performance, asynchronous API serving.

Both frameworks allow integration with web or mobile applications for real-time credit scoring.

## Usage

1. Train the model using provided scripts.
2. Start the API server (Flask or FastAPI).
3. Send applicant data to the API endpoint to receive a credit score classification.

---
# Credit-Scoring-Model-Production
