import pandas as pd
import numpy as np
import joblib
import json
import os

# Configuration Paths - Assuming this file is in the same directory as scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L1_MODEL_FILE = os.path.join(BASE_DIR, 'logistic_l1_model.pkl')
CONFIG_FILE = os.path.join(BASE_DIR, 'model_config.json')

def load_artifacts():
    """
    Load the trained model and configuration.
    """
    if not os.path.exists(L1_MODEL_FILE) or not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError("Model files not found! Please run airline_delay_prediction.py first.")
        
    model = joblib.load(L1_MODEL_FILE)
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    return model, config

def get_readable_feature_name(feature_name, value=None):
    """
    Convert technical feature names to readable text.
    """
    if feature_name == "MONTH": 
        month_map = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 
                     7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        if value is not None:
            return f"Month: {month_map.get(value, 'Unknown')}"
        return "Month of Year"
        
    if feature_name == "DAY": return "Day of Month"
    
    if feature_name == "DAY_OF_WEEK": 
        day_map = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
        if value is not None:
            return f"Day: {day_map.get(value, 'Unknown')}"
        return "Day of Week"
        
    if feature_name == "SCHEDULED_DEPARTURE": return "Scheduled Departure Time"
    if feature_name == "DISTANCE": return "Flight Distance"
    
    if feature_name.startswith("AIRLINE_"):
        return f"Airline: {feature_name.split('_')[-1]}"
    if feature_name.startswith("ORIGIN_AIRPORT_"):
        return f"Origin Airport: {feature_name.split('_')[-1]}"
    if feature_name.startswith("DESTINATION_AIRPORT_"):
        return f"Destination Airport: {feature_name.split('_')[-1]}"
        
    return feature_name

def calculate_probability_contribution(model, data, weather_impact=0.0, weather_desc="Clear"):
    """
    Explain prediction by calculating the Marginal Probability Contribution of factors.
    Returns: DataFrame with ['Feature', 'Contribution', 'Description', 'Effect', 'Tag']
    """
    # 1. Transform input
    preprocessor = model.named_steps['preprocessor']
    row_df = pd.DataFrame([data])
    row_transformed = preprocessor.transform(row_df)
    
    # 2. Get coefficients & Base Probability
    classifier = model.named_steps['classifier']
    coefs = classifier.coef_[0]
    
    if hasattr(row_transformed, "toarray"):
        row_values = row_transformed.toarray()[0]
    else:
        row_values = row_transformed[0]
        
    # 3. Calculate Log-Odds Contribution (Beta * x)
    logit_contributions = row_values * coefs
    
    # 4. Calculate Base Probability
    # Sum of contributions + Intercept
    logit_sum = logit_contributions.sum() + classifier.intercept_[0]
    base_prob = 1 / (1 + np.exp(-logit_sum))
    
    # 5. Calculate Marginal Probability Contribution
    # Approximation: dP/dx = p * (1-p) * Beta
    prob_slope = base_prob * (1 - base_prob)
    prob_contributions = logit_contributions * prob_slope
    
    # 6. Map to Features
    cat_features = (preprocessor.named_transformers_['cat']
                   .named_steps['onehot']
                   .get_feature_names_out(['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']))
    num_features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'DISTANCE']
    all_features = np.concatenate([num_features, cat_features])
    
    explanation = pd.DataFrame({
        'FeatureRaw': all_features,
        'Contribution': prob_contributions,
        'Value': row_values # Scaled/Encoded values
    })
    
    explanation['Abs_Contribution'] = explanation['Contribution'].abs()
    # Filter noise
    explanation = explanation[explanation['Abs_Contribution'] > 0.0001]
    
    # 7. Add Weather (Hardware Logic)
    # Weather impact is ALREADY a probability delta, so we use it directly
    if weather_impact > 0:
        new_row = pd.DataFrame({
            'FeatureRaw': [f"Weather_Condition"],
            'Contribution': [weather_impact],
            'Abs_Contribution': [weather_impact],
            'Value': [1] 
        })
        explanation = pd.concat([new_row, explanation], ignore_index=True)
        
    explanation = explanation.sort_values(by='Abs_Contribution', ascending=False)
    
    # 8. Generate Friendly Descriptions
    results = []
    
    for _, row in explanation.iterrows():
        feature_raw = row['FeatureRaw']
        contr = row['Contribution']
        
        readable_name = ""
        description = ""
        
        # Weather
        if feature_raw == "Weather_Condition":
            readable_name = f"Weather: {weather_desc}"
            description = "Adverse weather conditions directly reduce capacity."
        else:
            # Need original value for mapping logic (e.g. which day of week)
            # Use data dict lookup for original raw values
            # Need to match feature_raw to data keys
            
            orig_value = None
            if feature_raw in data:
                orig_value = data[feature_raw]
            elif "DAY_OF_WEEK" in feature_raw: # One hot won't match, but numeric will
                 orig_value = data.get('DAY_OF_WEEK')
            elif "MONTH" in feature_raw:
                 orig_value = data.get('MONTH')

            readable_name = get_readable_feature_name(feature_raw, orig_value)
            
            # Simple static descriptions based on feature type
            if "Day" in readable_name: 
                if contr > 0: description = "High traffic period increases risk."
                else: description = "Low traffic period reduces risk."
            elif "Month" in readable_name: description = "Seasonal traffic pattern."
            elif "Time" in readable_name: description = "Time of day traffic congestion."
            elif "Airline" in readable_name: description = "Carrier historical performance."
            elif "Origin" in readable_name: description = "Airport infrastructure/congestion."
            elif "Destination" in readable_name: description = "Arrival airport capacity."
            elif "Distance" in readable_name: description = "Route length factor."
            else: description = "Historical trend factor."

        effect = "Increases Risk [!]" if contr > 0 else "Decreases Risk [OK]"
        tag = "increase_risk" if contr > 0 else "decrease_risk"
        
        results.append({
            "feature": readable_name,
            "effect": effect,
            "impact": f"{contr:.2%}", # Format as percentage
            "desc": description,
            "tag": tag
        })
        
    return results, base_prob
