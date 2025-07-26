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

## Workflow

- **1. Load the dataset** : The dataset is loaded from a CSV file containing applicant data.
- **2. Preprocess and clean the data**: The data is cleaned and transformed to ensure it is suitable for model training.
- **3. Feature engineering for RFM**: Features are created based on Recency, Frequency, and Monetary value to enhance model performance.
- **4. Train the model**: The model is trained using the processed data, focusing on minimizing false negatives.
- **5. Evaluate the model**: The model is evaluated using precision and recall metrics to ensure it meets business requirements.
- **6. Deploy the model**: The trained model is deployed as a REST API using Flask or FastAPI, allowing for easy integration with applications.
- **7. Hyperparameter tuning**: The model is fine-tuned to optimize performance, focusing on reducing false negatives while maintaining a balance with precision and recall.
- **8. Model inference**: The deployed API can be used to classify new applicants based on their data, providing real-time credit scoring.
- **9. Model evaluation**: Continuous monitoring and evaluation of the model's performance in production to ensure it remains effective and accurate over time.
- **10. Deployment**: The model is deployed in a production environment, ensuring it is accessible for real-time predictions.
- **11. Explainability**: Use SHAP or LIME to interpret model decisions.

## Usage

1. 
2.
3. 

