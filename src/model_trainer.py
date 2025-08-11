# src/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import save_model, save_preprocessor, load_preprocessor
from config import (
    TARGET_COLUMN, FEATURES,
    SMOTE_RANDOM_STATE, SMOTE_K_NEIGHBORS, TRAIN_TEST_SPLIT_RATIO,
    MODEL_RANDOM_STATE, RF_N_ESTIMATORS,LABEL_ENCODER_OCCUPATION_PATH, LABEL_ENCODER_TYPE_OF_LOAN_PATH,
    CREDIT_SCORE_CATEGORIES, PAYMENT_BEHAVIOUR_CATEGORIES,PAYMENT_BEHAVIOUR_COL,LABEL_ENCODER_PAYMENT_MIN_AMOUNT_PATH,
    OCCUPATION_COL, TYPE_OF_LOAN_COL, PAYMENT_OF_MIN_AMOUNT_COL, NUM_OF_LOAN_COL, LABEL_ENCODER_NUM_OF_LOAN_PATH,
    RF_MODEL_PATH, ORDINAL_ENCODER_PATH, LABEL_ENCODER_PATH, PAYMENT_BEHAVIOUR_ENCODER_PATH
)

def split_and_prepare_data(df: pd.DataFrame):
    """
    Splits the combined DataFrame into train and test sets based on Credit_Score nulls.
    Applies encoding to categorical columns and handles potential NaNs.
    """
    print("\n--- Splitting & Preparing Data ---")

    train_df = df[df[TARGET_COLUMN].notnull()].copy()
    test_df = df[df[TARGET_COLUMN].isnull()].copy()

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    credit_score_encoder = OrdinalEncoder(categories=[CREDIT_SCORE_CATEGORIES])
    train_df[TARGET_COLUMN] = credit_score_encoder.fit_transform(train_df[[TARGET_COLUMN]])
    save_preprocessor(credit_score_encoder, ORDINAL_ENCODER_PATH)
    print(f"Encoded '{TARGET_COLUMN}' and saved encoder to {ORDINAL_ENCODER_PATH}.")

    # --- UPDATED: NUM_OF_LOAN_COL removed from categorical_cols_to_encode ---
    categorical_cols_to_encode = [OCCUPATION_COL, TYPE_OF_LOAN_COL, PAYMENT_OF_MIN_AMOUNT_COL]
    
    label_encoders = {}
    encoder_paths = {
        OCCUPATION_COL: LABEL_ENCODER_OCCUPATION_PATH,
        TYPE_OF_LOAN_COL: LABEL_ENCODER_TYPE_OF_LOAN_PATH,
        PAYMENT_OF_MIN_AMOUNT_COL: LABEL_ENCODER_PAYMENT_MIN_AMOUNT_PATH,
        # REMOVED: NUM_OF_LOAN_COL: LABEL_ENCODER_NUM_OF_LOAN_PATH,
    }

    for col in categorical_cols_to_encode:
        if col in train_df.columns:
            train_df.loc[:, col] = train_df[col].astype(str).fillna('Unknown')
            test_df.loc[:, col] = test_df[col].astype(str).fillna('Unknown')

            all_categories = pd.concat([train_df[col], test_df[col]]).unique()
            if 'Unknown' not in all_categories:
                all_categories = np.append(all_categories, 'Unknown')

            le = LabelEncoder()
            le.fit(all_categories)

            train_df.loc[:, col] = le.transform(train_df[col])
            test_df.loc[:, col] = le.transform(test_df[col])
            save_preprocessor(le, encoder_paths[col])
            label_encoders[col] = le
            print(f"Encoded '{col}' and saved its label encoder to {encoder_paths[col]}. Classes: {le.classes_}")

    payment_behaviour_encoder = OrdinalEncoder(categories=[PAYMENT_BEHAVIOUR_CATEGORIES], handle_unknown='use_encoded_value', unknown_value=-1)
    if PAYMENT_BEHAVIOUR_COL in train_df.columns:
        train_df.loc[:, PAYMENT_BEHAVIOUR_COL] = train_df[PAYMENT_BEHAVIOUR_COL].astype(str).fillna('Unknown')
        test_df.loc[:, PAYMENT_BEHAVIOUR_COL] = test_df[PAYMENT_BEHAVIOUR_COL].astype(str).fillna('Unknown')
        
        train_df.loc[:, PAYMENT_BEHAVIOUR_COL] = payment_behaviour_encoder.fit_transform(train_df[[PAYMENT_BEHAVIOUR_COL]])
        test_df.loc[:, PAYMENT_BEHAVIOUR_COL] = payment_behaviour_encoder.transform(test_df[[PAYMENT_BEHAVIOUR_COL]])
        save_preprocessor(payment_behaviour_encoder, PAYMENT_BEHAVIOUR_ENCODER_PATH)
        print(f"Encoded '{PAYMENT_BEHAVIOUR_COL}' and saved its encoder to {PAYMENT_BEHAVIOUR_ENCODER_PATH}.")

    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(int)
    print("Class distribution in training set before SMOTE:")
    print(train_df[TARGET_COLUMN].value_counts(normalize=True))

    current_features_train = [f for f in FEATURES if f in train_df.columns]
    current_features_test = [f for f in FEATURES if f in test_df.columns]
    common_features = list(set(current_features_train) & set(current_features_test))
    print(f"Using common features for training and testing: {common_features}")

    X_train = train_df[common_features].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    X_test = test_df[common_features].copy()

    for col in X_train.columns:
        numeric_series = pd.to_numeric(X_train[col], errors='coerce')
        median_val = numeric_series.median()
        if pd.isna(median_val):
            print(f"WARNING: Feature '{col}' in X_train is all NaN after conversion. Filling with 0.")
            median_val = 0
        X_train.loc[:, col] = numeric_series.fillna(median_val)

    for col in X_test.columns:
        numeric_series = pd.to_numeric(X_test[col], errors='coerce')
        median_val_for_test = pd.to_numeric(train_df[col], errors='coerce').median() if col in train_df.columns else 0
        if pd.isna(median_val_for_test):
            median_val_for_test = 0
        X_test.loc[:, col] = numeric_series.fillna(median_val_for_test)

    print("Filled any remaining NaNs in feature sets with median (or 0 if median was NaN).")

    smote = SMOTE(random_state=SMOTE_RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("Applied SMOTE for class balancing.")

    print("\nBalanced training set class distribution:")
    print(y_train_balanced.value_counts(normalize=True))

    print("\nShapes:")
    print(f"Original training data features: {X_train.shape}")
    print(f"Balanced training data features: {X_train_balanced.shape}")
    print(f"True test set features: {X_test.shape}")

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_balanced, y_train_balanced, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=MODEL_RANDOM_STATE
    )
    print(f"Validation split: Train {X_train_split.shape}, Val {X_val_split.shape}")

    return X_train_split, y_train_split, X_val_split, y_val_split, X_test, test_df[TARGET_COLUMN]


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains the Random Forest Classifier model.
    ... (rest of the function is the same) ...
    """
    print("\n--- Training Random Forest Classifier Model ---")
    rf_classifier = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=MODEL_RANDOM_STATE)
    rf_classifier.fit(X_train, y_train)
    save_model(rf_classifier, RF_MODEL_PATH)
    print(f"Random Forest Classifier trained and saved to {RF_MODEL_PATH}.")
    return rf_classifier

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series):
    """
    Evaluates the trained model on a validation set.
    ... (rest of the function is the same) ...
    """
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_val)

    y_pred_rounded = np.round(y_pred).astype(int)

    target_names = CREDIT_SCORE_CATEGORIES

    mse = mean_squared_error(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred_rounded)
    precision = precision_score(y_val, y_pred_rounded, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred_rounded, average='weighted', zero_division=0)

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_rounded, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_val, y_pred_rounded, labels=np.unique(y_val))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()

    print("\nConfusion Matrix Insights:")
    print(f"Total Correct Predictions (TPs across all classes): {np.diag(cm).sum()}")
    print(f"Total Incorrect Predictions: {cm.sum() - np.diag(cm).sum()}")