import pandas as pd
import numpy as np
from datetime import datetime

# Parameters
num_cows = 100
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 12, 31)
date_range = pd.date_range(start_date, end_date)

# Generate cow IDs and assign breeds
cow_ids = [f"Cow_{i+1}" for i in range(num_cows)]
breeds = ['Holstein', 'Jersey', 'Guernsey', 'Brown Swiss']
cow_breed_map = {cow: np.random.choice(breeds) for cow in cow_ids}

# Seasonal adjustment function
def seasonal_factor(month):
    if month in [3, 4, 5]:
        return 1.10
    elif month in [6, 7, 8]:
        return 0.90
    else:
        return 1.00

# Weather impact function
def weather_impact(temp, humidity):
    temp_effect = 1 - abs(temp - 15) * 0.01
    humidity_effect = 1 - abs(humidity - 60) * 0.005
    return max(0.8, temp_effect * humidity_effect)

# Economic parameters
milk_price_per_kg = 0.45  # USD per kg
feed_cost_per_kg = 0.25   # USD per kg

# Create dataset
data = []
for date in date_range:
    temperature = round(np.random.normal(15, 10), 1)
    humidity = round(np.random.uniform(40, 90), 1)
    
    for cow in cow_ids:
        base_yield = np.random.normal(25, 5)
        feed_intake = round(np.random.normal(20, 3), 2)
        lactation_stage = np.random.choice(['Early', 'Mid', 'Late'])
        health_score = np.random.randint(70, 100)
        breed = cow_breed_map[cow]
        
        # Adjust milk yield
        adjusted_yield = base_yield * seasonal_factor(date.month) * weather_impact(temperature, humidity)
        milk_yield = round(adjusted_yield, 2)
        
        # Calculate revenue, feed cost, and profit
        revenue = round(milk_yield * milk_price_per_kg, 2)
        feed_cost = round(feed_intake * feed_cost_per_kg, 2)
        profit = round(revenue - feed_cost, 2)
        
        data.append([
            date, cow, breed, milk_yield, feed_intake, lactation_stage,
            health_score, temperature, humidity, milk_price_per_kg, revenue, feed_cost, profit
        ])

# Convert to DataFrame
df = pd.DataFrame(data, columns=[
    'Date', 'Cow_ID', 'Breed', 'Milk_Yield_kg', 'Feed_Intake_kg',
    'Lactation_Stage', 'Health_Score', 'Temperature_C', 'Humidity_%',
    'Milk_Price_USD_per_kg', 'Revenue_USD', 'Feed_Cost_USD', 'Profit_USD'
])

# Save detailed dataset
df.to_csv('cedar_creek_dairy_dataset_profit.csv', index=False)

# Monthly summary for revenue and profit
df['Month'] = df['Date'].dt.to_period('M')
monthly_summary = df.groupby('Month')[['Revenue_USD', 'Profit_USD']].sum().reset_index()
monthly_summary.to_csv('monthly_summary.csv', index=False)

print("Dataset with Profit and monthly summary saved as cedar_creek_dairy_dataset_profit.csv and monthly_summary.csv")