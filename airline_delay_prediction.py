"""
AIRLINE DELAY PREDICTION SYSTEM
-------------------------------
A production-ready Machine Learning system to predict flight delays using 
Logistic Regression with L2 (Ridge) and L1 (Lasso) regularization.

Based on Lecture Material: "lect_02_03_linear_models.pdf"
- Logistic Regression (Slides 53-77)
- L2 Regularization (Slides 35-39)
- L1 Regularization (Slide 40)
- Feature Normalization (Slides 79-82)

Author: Antigravity
Date: 2025-12-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Configuration
DATA_FILE = 'filtered_flight_data.csv'
L2_MODEL_FILE = 'logistic_l2_model.pkl'
L1_MODEL_FILE = 'logistic_l1_model.pkl'
SCALER_FILE = 'feature_scaler.pkl'
CONFIG_FILE = 'model_config.json'
RANDOM_STATE = 42

def load_data(filepath):
    """
    Load flight data from CSV.
    """
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset {filepath} not found!")
    
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis.
    """
    print("\n--- EXPLORATORY DATA ANALYSIS ---")
    
    # Create target variable if not exists (Delay > 15 mins)
    # Using 'arr_delay' as the ground truth
    if 'arr_delay' not in df.columns:
        raise ValueError("Target column 'arr_delay' missing!")
        
    df['is_delayed'] = (df['arr_delay'] > 15).astype(int)
    
    # Delay Percentage
    delay_counts = df['is_delayed'].value_counts()
    delay_pct = delay_counts[1] / len(df) * 100
    print(f"\nTotal Flights: {len(df)}")
    print(f"Delayed Flights: {delay_counts[1]} ({delay_pct:.2f}%)")
    print(f"On-time Flights: {delay_counts[0]} ({100-delay_pct:.2f}%)")
    
    # Visualization: Delay Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_delayed', data=df)
    plt.title('Class Distribution: On-time (0) vs Delayed (1)')
    plt.xlabel('Delay Status')
    plt.ylabel('Count')
    plt.savefig('eda_delay_distribution.png')
    print("Saved 'eda_delay_distribution.png'")
    
    # Correlation Heatmap (Numerical features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Filter only relevant numeric columns for heatmap to avoid clutter
    eda_cols = ['dep_delay', 'arr_delay', 'distance', 'air_time', 'is_delayed']
    eda_cols = [c for c in eda_cols if c in df.columns]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[eda_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png')
    print("Saved 'eda_correlation_heatmap.png'")
    
    return df

def preprocess_data(df):
    """
    Preprocess data: handle missing values, encode features, normalize.
    """
    print("\n--- DATA PREPROCESSING ---")
    
    # 1. Feature Selection
    # Drop leakage features (available only after flight)
    leakage_cols = ['arr_delay', 'dep_delay', 'actual_elapsed_time', 'air_time', 
                    'wheels_on', 'wheels_off', 'taxi_in', 'taxi_out', 
                    'cancellation_code', 'carrier_delay', 'weather_delay', 
                    'nas_delay', 'security_delay', 'late_aircraft_delay', 
                    'dep_time', 'arr_time'] # dep_time/arr_time are actual times
    
    # Keep scheduled times and static info
    # Features to use:
    # - Month, Day of Month, Day of Week (Temporal)
    # - CRS_DEP_TIME (Scheduled Departure Time)
    # - OP_UNIQUE_CARRIER (Airline)
    # - ORIGIN, DEST (Airports)
    # - DISTANCE
    
    features = ['month', 'day_of_month', 'day_of_week', 'crs_dep_time', 
                'op_unique_carrier', 'origin', 'dest', 'distance']
    
    # Ensure columns exist
    features = [c for c in features if c in df.columns]
    
    print(f"Selected Features: {features}")
    
    X = df[features]
    y = df['is_delayed']
    
    # 2. Train-Test Split (80/20)
    # Stratify to maintain class balance
    print("Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 3. Feature Engineering Pipeline
    # Numeric features: Impute median -> Standard Scaler (Lecture slides 79-82)
    numeric_features = ['month', 'day_of_month', 'day_of_week', 'crs_dep_time', 'distance']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # Fit on TRAIN, transform TEST
    ])
    
    # Categorical features: Impute constant -> OneHotEncoder
    categorical_features = ['op_unique_carrier', 'origin', 'dest']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Handle new categories in test
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train, penalty, preprocessor):
    """
    Train Logistic Regression model with specified regularization.
    """
    reg_name = "L2 (Ridge)" if penalty == 'l2' else "L1 (Lasso)"
    print(f"\nTraining Logistic Regression with {reg_name} Regularization...")
    
    # Lecture: Sigmoid function, Cross-Entropy Loss
    # Solver 'saga' supports both L1 and L2 and is efficient for large datasets
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            penalty=penalty,
            C=1.0, # Inverse regularization strength
            solver='saga',
            max_iter=1000,
            class_weight='balanced', # Handle class imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1 # Parallelize
        ))
    ])
    
    model.fit(X_train, y_train)
    print(f"{reg_name} Model Trained.")
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance and return metrics.
    """
    print(f"\nEvaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred, y_prob

def get_feature_importance(model, feature_names):
    """
    Extract and rank feature importance from coefficients.
    """
    # Access the classifier step
    classifier = model.named_steps['classifier']
    coefs = classifier.coef_[0]
    
    # Create dataframe
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    })
    
    importance = importance.sort_values(by='Abs_Coefficient', ascending=False)
    return importance

def save_artifacts(l2_model, l1_model, preprocessor):
    """
    Save trained models and scaler.
    """
    print("\nSaving artifacts...")
    joblib.dump(l2_model, L2_MODEL_FILE)
    joblib.dump(l1_model, L1_MODEL_FILE)
    
    # Save scaler (it's part of the pipeline, but saving separately as requested)
    # Note: The pipeline object already contains the fitted scaler.
    # We will extract it for the specific file requirement.
    scaler = l2_model.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
    joblib.dump(scaler, SCALER_FILE)
    
    # Save config
    config = {
        'features': ['month', 'day_of_month', 'day_of_week', 'crs_dep_time', 
                     'op_unique_carrier', 'origin', 'dest', 'distance'],
        'target': 'is_delayed (>15 mins)',
        'l2_model_params': l2_model.named_steps['classifier'].get_params(),
        'l1_model_params': l1_model.named_steps['classifier'].get_params()
    }
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
        
    print("All artifacts saved successfully.")

def predict_flight_delay(flight_data, model):
    """
    Make a prediction for a single flight.
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([flight_data])
    
    # Predict
    prob = model.predict_proba(df)[0, 1]
    prediction = "DELAYED" if prob > 0.5 else "ON-TIME"
    
    return prediction, prob

def main():
    try:
        # 1. Load Data
        df = load_data(DATA_FILE)
        
        # 2. EDA
        df = perform_eda(df)
        
        # 3. Preprocessing & Split
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
        
        # 4. Train L2 Model (Ridge)
        l2_model = train_model(X_train, y_train, 'l2', preprocessor)
        
        # 5. Train L1 Model (Lasso)
        l1_model = train_model(X_train, y_train, 'l1', preprocessor)
        
        # 6. Evaluate
        l2_metrics, l2_pred, l2_prob = evaluate_model(l2_model, X_test, y_test, "L2 Model")
        l1_metrics, l1_pred, l1_prob = evaluate_model(l1_model, X_test, y_test, "L1 Model")
        
        # 7. Feature Importance
        # Get feature names after one-hot encoding
        feature_names = (l2_model.named_steps['preprocessor']
                         .transformers_[1][1] # categorical transformer
                         .named_steps['onehot']
                         .get_feature_names_out(['op_unique_carrier', 'origin', 'dest']))
        
        numeric_features = ['month', 'day_of_month', 'day_of_week', 'crs_dep_time', 'distance']
        all_features = np.concatenate([numeric_features, feature_names])
        
        l2_importance = get_feature_importance(l2_model, all_features)
        l1_importance = get_feature_importance(l1_model, all_features)
        
        l1_zeroed = (l1_importance['Coefficient'] == 0).sum()
        
        # 8. Comparison Report
        print("\n" + "="*44)
        print("AIRLINE DELAY PREDICTION - MODEL COMPARISON")
        print("DATASET SUMMARY")
        print(f"Total flights: {len(df)}")
        print(f"Delayed: {df['is_delayed'].sum()} ({df['is_delayed'].mean()*100:.2f}%)")
        print(f"On-time: {len(df)-df['is_delayed'].sum()} ({(1-df['is_delayed'].mean())*100:.2f}%)")
        print(f"Features used: {len(all_features)}")
        
        print("\nMODEL PERFORMANCE")
        print(f"{'Metric':<15} {'L2 (Ridge)':<15} {'L1 (Lasso)':<15}")
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            print(f"{metric:<15} {l2_metrics[metric]*100:.2f}%          {l1_metrics[metric]*100:.2f}%")
            
        best_model = "L2 (Ridge)" if l2_metrics['Accuracy'] > l1_metrics['Accuracy'] else "L1 (Lasso)"
        print(f"\nBEST MODEL: {best_model}")
        print("Reason: Higher Accuracy and balanced performance.")
        
        print("\nTOP 5 FEATURES (L2 Model)")
        print(l2_importance.head(5)[['Feature', 'Coefficient']].to_string(index=False))
        
        print("\nTOP 5 FEATURES (L1 Model)")
        print(l1_importance.head(5)[['Feature', 'Coefficient']].to_string(index=False))
        print(f"Features zeroed out by L1: {l1_zeroed}")
        
        # 9. Visualization
        # ROC Curve
        fpr_l2, tpr_l2, _ = roc_curve(y_test, l2_prob)
        fpr_l1, tpr_l1, _ = roc_curve(y_test, l1_prob)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_l2, tpr_l2, label=f'L2 (AUC = {l2_metrics["ROC-AUC"]:.2f})')
        plt.plot(fpr_l1, tpr_l1, label=f'L1 (AUC = {l1_metrics["ROC-AUC"]:.2f})', linestyle='--')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.savefig('model_roc_comparison.png')
        print("\nSaved 'model_roc_comparison.png'")
        
        # 10. Sample Prediction
        sample_flight = X_test.iloc[0].to_dict()
        print("\nSAMPLE PREDICTION")
        print(f"Input: {sample_flight}")
        
        pred_l2, prob_l2 = predict_flight_delay(sample_flight, l2_model)
        pred_l1, prob_l1 = predict_flight_delay(sample_flight, l1_model)
        
        print(f"L2 Model: {pred_l2} ({prob_l2*100:.1f}% delay risk)")
        print(f"L1 Model: {pred_l1} ({prob_l1*100:.1f}% delay risk)")
        
        # 11. Save
        save_artifacts(l2_model, l1_model, preprocessor)
        
        print("\n" + "="*44)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
