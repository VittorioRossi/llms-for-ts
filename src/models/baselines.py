import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def naive_forecast(data, target_size=1, seasonality=0):
    if not isinstance(seasonality, list):
        seasonality = [seasonality]
    
    preds = []
    for i in range(target_size):
        seasonal_values = [data[-s+i] for s in seasonality]
        preds.append(np.mean(seasonal_values))
    
    return preds

def mean_forecast(data, target_size=1, seasonality=0):
    avg = np.mean(data)
    return [avg for _ in range(target_size)] 

def ses_forecast(data, target_size=1, seasonality=0):
    if seasonality > 0:
        # Initialize the Exponential Smoothing model with seasonality
        model = ExponentialSmoothing(
            data,
            seasonal='add',  # or 'mul' for multiplicative seasonality
            seasonal_periods=seasonality,
            initialization_method='estimated'
        ).fit()

    # Generate the forecast for the given target size
    forecast = model.forecast(target_size)
    return forecast.tolist()
