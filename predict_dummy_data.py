import pandas as pd
import prediction_utils # Import shared utility
import traceback

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
            'MONTH': 12, 'DAY': 24, 'DAY_OF_WEEK': 7, 
            'SCHEDULED_DEPARTURE': 1800, 'AIRLINE': 'UA', 
            'ORIGIN_AIRPORT': 'ORD', 'DESTINATION_AIRPORT': 'JFK', 'DISTANCE': 740.0
        },
        {
            'scenario': 'Low Risk (Spring/PHX)',
            'MONTH': 4, 'DAY': 15, 'DAY_OF_WEEK': 3, 
            'SCHEDULED_DEPARTURE': 900, 'AIRLINE': 'WN', 
            'ORIGIN_AIRPORT': 'PHX', 'DESTINATION_AIRPORT': 'LAS', 'DISTANCE': 255.0
        },
        {
            'scenario': 'Medium Risk (Summer/ATL)',
            'MONTH': 7, 'DAY': 10, 'DAY_OF_WEEK': 5, 
            'SCHEDULED_DEPARTURE': 1400, 'AIRLINE': 'DL', 
            'ORIGIN_AIRPORT': 'ATL', 'DESTINATION_AIRPORT': 'MCO', 'DISTANCE': 404.0
        },
        {
            'scenario': 'Long Haul (LAX -> JFK)',
            'MONTH': 6, 'DAY': 1, 'DAY_OF_WEEK': 6, 
            'SCHEDULED_DEPARTURE': 800, 'AIRLINE': 'AA', 
            'ORIGIN_AIRPORT': 'LAX', 'DESTINATION_AIRPORT': 'JFK', 'DISTANCE': 2475.0
        },
        {
            'scenario': 'Short Haul (BOS -> LGA)',
            'MONTH': 11, 'DAY': 27, 'DAY_OF_WEEK': 4, 
            'SCHEDULED_DEPARTURE': 1700, 'AIRLINE': 'B6', 
            'ORIGIN_AIRPORT': 'BOS', 'DESTINATION_AIRPORT': 'LGA', 'DISTANCE': 184.0
        }
    ]
    return pd.DataFrame(data)

def main():
    try:
        model, config = prediction_utils.load_artifacts()
        df = create_dummy_data()
        
        print("\n" + "="*60)
        print("FLIGHT DELAY PREDICTION ON DUMMY DATA")
        print("="*60)
        
        for index, row in df.iterrows():
            scenario = row['scenario']
            # Drop scenario for prediction
            flight_data = row.drop('scenario')
            # Convert to dict for utility
            flight_dict = flight_data.to_dict()
            
            # Use utility for explanation and probability
            # No weather impact in this script, so 0
            explanation, prob = prediction_utils.calculate_probability_contribution(model, flight_dict)
            
            prediction = "DELAYED" if prob > 0.5 else "ON-TIME"
            
            print(f"\nScenario: {scenario}")
            print(f"Flight: {flight_dict['AIRLINE']} {flight_dict['ORIGIN_AIRPORT']} -> {flight_dict['DESTINATION_AIRPORT']}")
            print(f"Date: Month {flight_dict['MONTH']}, Day {flight_dict['DAY']}")
            print("-" * 30)
            print(f"Prediction: {prediction}")
            print(f"Probability of Delay: {prob*100:.2f}%")
            
            # Explain
            print("\nTop Contributing Factors (Probability Impact):")
            
            for item in explanation[:5]: # Show top 5
                print(f"  - {item['feature']:<30}: {item['effect']} (Impact: {item['impact']})")
                
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
