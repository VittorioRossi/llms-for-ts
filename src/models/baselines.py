import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def naive_forecast(data, target_size=1, seasonality=0):
    if seasonality == 0:
        return data[-target_size:]
    else:
        return data[-seasonality:-seasonality+target_size]

def mean_forecast(data, target_size=1, seasonality=0):
    return [np.mean(data)]

def ses_forecast(data, target_size=1, seasonality=0):
    model = SimpleExpSmoothing(data, seasonal_periods=seasonality).fit()
    return model.forecast(target_size)[0]