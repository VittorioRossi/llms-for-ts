import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def naive_forecast(data, target_size=1, seasonality=0):
    if isinstance(seasonality, list):
        ses = []
        for s in seasonality:
            ses.append(data[-s])
        return np.mean(ses)
    
    if seasonality == 0:
        return data[-target_size:]
    
    return data[-seasonality:-seasonality+target_size]

def mean_forecast(data, target_size=1, seasonality=0):
    return [np.mean(data) for _ in range(target_size)] 

def ses_forecast(data, target_size=1, seasonality=0):
    model = ExponentialSmoothing(data, 
                                 seasonal_periods=seasonality).fit()
    
    return model.forecast(target_size).fittedvalues