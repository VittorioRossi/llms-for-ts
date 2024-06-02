import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def naive_forecast(data, target_size=1):
    return data[-target_size:]

def mean_forecast(data, target_size=1):
    return [np.mean(data)]

def ses_forecast(data, target_size=1):
    model = SimpleExpSmoothing(data).fit()
    return model.forecast(target_size)[0]