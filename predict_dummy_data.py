import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# Configuration
L1_MODEL_FILE = 'logistic_l1_model.pkl'
CONFIG_FILE = 'model_config.json'

def load_artifacts():
    print("Loading model and config...")
    model = joblib.load(L1_MODEL_FILE)
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    return model, config

def create_dummy_data():
    """
    Create dummy flight data representing different scenarios.
    """
    # Scenarios:
    # 1. High Risk: Winter, ORD (Chicago), Long distance, Holiday season
    # 2. Low Risk: Spring, PHX (Phoenix), Short distance, Mid-month
    # 3. Medium Risk: Summer, ATL (Atlanta), Afternoon
    
    data = [
        {
            'scenario': 'High Risk (Winter/ORD)',
            'month': 12, 'day_of_month': 24, 'day_of_week': 7, 
            'crs_dep_time': 1800, 'op_unique_carrier': 'UA', 
            'origin': 'ORD', 'dest': 'JFK', 'distance': 740.0
        },
        {
            'scenario': 'Low Risk (Spring/PHX)',
            'month': 4, 'day_of_month': 15, 'day_of_week': 3, 
            'crs_dep_time': 900, 'op_unique_carrier': 'WN', 
            'origin': 'PHX', 'dest': 'LAS', 'distance': 255.0
        },
        {
            'scenario': 'Medium Risk (Summer/ATL)',
            'month': 7, 'day_of_month': 10, 'day_of_week': 5, 
            'crs_dep_time': 1400, 'op_unique_carrier': 'DL', 
            'origin': 'ATL', 'dest': 'MCO', 'distance': 404.0
        },
        {
            'scenario': 'Long Haul (LAX -> JFK)',
            'month': 6, 'day_of_month': 1, 'day_of_week': 6, 
            'crs_dep_time': 800, 'op_unique_carrier': 'AA', 
            'origin': 'LAX', 'dest': 'JFK', 'distance': 2475.0
        },
        {
            'scenario': 'Short Haul (BOS -> LGA)',
            'month': 11, 'day_of_month': 27, 'day_of_week': 4, 
            'crs_dep_time': 1700, 'op_unique_carrier': 'B6', 
            'origin': 'BOS', 'dest': 'LGA', 'distance': 184.0
        }
    ]
    return pd.DataFrame(data)

def explain_prediction(model, row, feature_names):
    """
    Explain prediction by showing top contributing features.
    """
    # 1. Transform input using the pipeline's preprocessor
    # We need to access the preprocessor from the pipeline
    preprocessor = model.named_steps['preprocessor']
    
    # Transform the single row (must be DataFrame)
    row_df = pd.DataFrame([row])
    row_transformed = preprocessor.transform(row_df)
    
    # 2. Get coefficients
    classifier = model.named_steps['classifier']
    coefs = classifier.coef_[0]
    
    # 3. Calculate contribution: feature_value * coefficient
    # Note: row_transformed is a sparse matrix, convert to dense
    if hasattr(row_transformed, "toarray"):
        row_values = row_transformed.toarray()[0]
    else:
        row_values = row_transformed[0]
        
    contributions = row_values * coefs
    
    # 4. Map to feature names
    # Get feature names from preprocessor
    cat_features = (preprocessor.named_transformers_['cat']
                   .named_steps['onehot']
                   .get_feature_names_out(['op_unique_carrier', 'origin', 'dest']))
    num_features = ['month', 'day_of_month', 'day_of_week', 'crs_dep_time', 'distance']
    all_features = np.concatenate([num_features, cat_features])
    
    # Create DataFrame
    explanation = pd.DataFrame({
        'Feature': all_features,
        'Value': row_values,
        'Coefficient': coefs,
        'Contribution': contributions
    })
    
    # Sort by absolute contribution
    explanation['Abs_Contribution'] = explanation['Contribution'].abs()
    explanation = explanation.sort_values(by='Abs_Contribution', ascending=False)
    
    return explanation.head(5)

def main():
    try:
        model, config = load_artifacts()
        df = create_dummy_data()
        
        print("\n" + "="*60)
        print("FLIGHT DELAY PREDICTION ON DUMMY DATA")
        print("="*60)
        
        for index, row in df.iterrows():
            scenario = row['scenario']
            # Drop scenario for prediction
            flight_data = row.drop('scenario')
            
            # Predict
            prob = model.predict_proba(pd.DataFrame([flight_data]))[0, 1]
            prediction = "DELAYED" if prob > 0.5 else "ON-TIME"
            
            print(f"\nScenario: {scenario}")
            print(f"Flight: {flight_data['op_unique_carrier']} {flight_data['origin']} -> {flight_data['dest']}")
            print(f"Date: Month {flight_data['month']}, Day {flight_data['day_of_month']}")
            print("-" * 30)
            print(f"Prediction: {prediction}")
            print(f"Probability of Delay: {prob*100:.2f}%")
            
            # Explain
            print("\nTop Contributing Factors (Reasoning):")
            explanation = explain_prediction(model, flight_data, config['features'])
            
            for i, r in explanation.iterrows():
                effect = "Increases Risk" if r['Contribution'] > 0 else "Decreases Risk"
                print(f"  - {r['Feature']:<20}: {effect} (Contrib: {r['Contribution']:.4f})")
                
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
