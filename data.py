import pandas as pd
import numpy as np

# Values based on monthly kWh from 
    # https://cornhusker-power.com/rebates/household-appliances/
    # https://www.energysage.com/electricity/house-watts/how-many-watts-does-an-electric-furnace-use/
# Estimation of daily kWh used (min, max)
appliances = {
    'Washer': (0, 0.1),            # Adjusted for hourly usage
    'Dryer': (0, 0.3),             # Adjusted for hourly usage
    'Lighting': (0, 0.05),         # Adjusted for hourly usage
    'Heating': (0, 3)              # Adjusted for hourly usage
}

n_days = 5

data = []
for day in range(n_days):
    day_data = []
    for hour in range(24):
        entry = {'Day': day+1, 'Hour': hour}
        for appliance, (low, high) in appliances.items():
            usage = np.random.uniform(low, high)
            entry[appliance] = usage
        day_data.append(entry)
    day_df = pd.DataFrame(day_data)
    
    total_day_usage = day_df.drop(columns=['Day', 'Hour']).sum().to_dict()
    total_day_usage['Day'] = day + 1
    total_day_usage['Hour'] = 'Total usage (kWh) for day',[day + 1]
    data.extend(day_data)
    data.append(total_day_usage)

df = pd.DataFrame(data)


print(df.head(50))

