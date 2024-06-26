import random

# Generate random outside temperature values
def generate_outside_temperatures(temp_range=(70, 100), seed=1):
    if seed is not None:
        random.seed(seed)
    
    # Day (hour 0) starts at a temperature between temp_range[0] and temp_range[0] + 10
    temp_start = random.randint(temp_range[0], temp_range[0] + 10)
    hourly_temperatures = []
    temp = temp_start

    for hour in range(24):
        # Morning hours, slow increase
        if 0 <= hour <= 10:
            temp += random.randint(1, 2)
        # Mid-day hours, steady increase
        elif 11 <= hour <= 17:
            temp += random.randint(1, 3)
        # Night hours, steady decrease
        else:
            temp -= random.randint(1, 3)
        
        hourly_temperatures.append(temp)
    
    return hourly_temperatures


time_based_temperatures = generate_outside_temperatures()


