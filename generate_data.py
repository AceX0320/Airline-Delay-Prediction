import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_rows=10000):
    print(f"Generating {num_rows} rows of synthetic flight data...")
    
    # 1. Setup Lists for Categorical Data
    airlines = ['UA', 'AA', 'US', 'F9', 'B6', 'OO', 'AS', 'NK', 'WN', 'DL', 'EV', 'HA', 'MQ', 'VX']
    airports = ['ORD', 'SFO', 'JFK', 'LAX', 'MIA', 'ATL', 'DEN', 'DFW', 'SEA', 'LAS', 'MCO', 'PHX', 'IAH', 'CLT']
    cancellation_reasons = ['A', 'B', 'C', 'D'] # A: Carrier, B: Weather, C: NAS, D: Security

    data = []
    
    start_date = datetime(2015, 1, 1)
    
    for _ in range(num_rows):
        # Date Info
        flight_date = start_date + timedelta(days=random.randint(0, 364))
        year = flight_date.year
        month = flight_date.month
        day = flight_date.day
        day_of_week = flight_date.isoweekday()
        
        # Flight Info
        airline = random.choice(airlines)
        flight_number = random.randint(100, 9999)
        tail_number = f"N{random.randint(100, 999)}XX"
        
        origin = random.choice(airports)
        dest = random.choice([a for a in airports if a != origin])
        
        # Distance (Approximate based on route logic would be better, but random for now)
        distance = random.randint(200, 3000)
        
        # Times
        # Scheduled Departure (HHMM)
        sched_dep_hour = random.randint(5, 23)
        sched_dep_min = random.randint(0, 59)
        scheduled_departure = sched_dep_hour * 100 + sched_dep_min
        
        # Scheduled Time (Duration in mins) - roughly distance / 8 + 30
        scheduled_time = int(distance / 8 + random.randint(20, 40))
        
        # Delays & Status
        is_cancelled = 1 if random.random() < 0.02 else 0 # 2% cancellation
        is_diverted = 1 if random.random() < 0.005 and not is_cancelled else 0 # 0.5% diverted
        
        departure_delay = 0
        arrival_delay = 0
        taxi_out = random.randint(10, 40)
        taxi_in = random.randint(5, 20)
        wheels_off = 0
        wheels_on = 0
        departure_time = 0
        arrival_time = 0
        elapsed_time = 0
        air_time = 0
        
        cancellation_reason = np.nan
        air_system_delay = np.nan
        security_delay = np.nan
        airline_delay = np.nan
        late_aircraft_delay = np.nan
        weather_delay = np.nan
        
        if is_cancelled:
            cancellation_reason = random.choice(cancellation_reasons)
            # Times remain NaN or 0 as per schema usually
        else:
            # Generate delays
            # 80% on time, 20% delayed
            if random.random() < 0.2:
                departure_delay = int(np.random.exponential(scale=20)) + 5 # Skewed delay
            else:
                departure_delay = random.randint(-10, 5) # Early or on time
                
            # Arrival delay is correlated with dep delay
            arrival_delay = departure_delay + random.randint(-10, 10)
            
            # Calculate actual times
            sched_dep_dt = flight_date.replace(hour=sched_dep_hour, minute=sched_dep_min)
            actual_dep_dt = sched_dep_dt + timedelta(minutes=departure_delay)
            departure_time = actual_dep_dt.hour * 100 + actual_dep_dt.minute
            
            wheels_off_dt = actual_dep_dt + timedelta(minutes=taxi_out)
            wheels_off = wheels_off_dt.hour * 100 + wheels_off_dt.minute
            
            # Air time
            air_time = scheduled_time - 20 + random.randint(-10, 10) # Roughly scheduled - taxi
            if air_time < 0: air_time = 20
            
            wheels_on_dt = wheels_off_dt + timedelta(minutes=air_time)
            wheels_on = wheels_on_dt.hour * 100 + wheels_on_dt.minute
            
            actual_arr_dt = wheels_on_dt + timedelta(minutes=taxi_in)
            arrival_time = actual_arr_dt.hour * 100 + actual_arr_dt.minute
            
            elapsed_time = air_time + taxi_in + taxi_out
            
            # Delay attribution (if delayed > 15 mins)
            if arrival_delay > 15:
                # Distribute delay randomly among causes
                remaining_delay = arrival_delay
                
                if random.random() < 0.3:
                    carrier_delay = random.randint(0, remaining_delay)
                    airline_delay = carrier_delay
                    remaining_delay -= carrier_delay
                else:
                    airline_delay = 0
                    
                if random.random() < 0.3 and remaining_delay > 0:
                    wx_delay = random.randint(0, remaining_delay)
                    weather_delay = wx_delay
                    remaining_delay -= wx_delay
                else:
                    weather_delay = 0
                    
                # Assign rest to others randomly or leave 0
                late_aircraft_delay = remaining_delay # Simplify
                air_system_delay = 0
                security_delay = 0
            else:
                # If not delayed, these are usually NaN or 0 in dataset
                pass

        # Scheduled Arrival
        sched_arr_dt = sched_dep_dt + timedelta(minutes=scheduled_time)
        scheduled_arrival = sched_arr_dt.hour * 100 + sched_arr_dt.minute

        row = [
            year, month, day, day_of_week, airline, flight_number, tail_number,
            origin, dest, scheduled_departure, departure_time, departure_delay,
            taxi_out, wheels_off, scheduled_time, elapsed_time, air_time,
            distance, wheels_on, taxi_in, scheduled_arrival, arrival_time,
            arrival_delay, is_diverted, is_cancelled, cancellation_reason,
            air_system_delay, security_delay, airline_delay, late_aircraft_delay,
            weather_delay
        ]
        data.append(row)
        
    columns = [
        'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 
        'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 
        'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 
        'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 
        'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 
        'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 
        'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Introduce Nulls to match schema % roughly (optional, but good for realism)
    # The user provided null stats, so our code should handle nulls.
    # The generation logic above naturally leaves some NaNs for delay causes.
    
    output_file = 'filtered_flight_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {output_file} with shape {df.shape}")

if __name__ == "__main__":
    generate_synthetic_data()
