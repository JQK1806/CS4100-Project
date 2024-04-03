 day_data = []
    for hour in range(24):
        entry = {'Day': day+1, 'Hour': hour}
        for appliance, (low, high) in appliances.items():
            usage = np.random.uniform(low, high)
            entry[appliance] = usage
        day_data.append(entry)
    day_df = pd.DataFrame(day_data)
    