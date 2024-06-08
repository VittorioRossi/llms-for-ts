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
    model = ExponentialSmoothing(data, 
                                 seasonal_periods=seasonality).fit()
    
    return model.forecast(target_size)[0]