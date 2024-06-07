import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Function to plot the time series
def plot_time_series(data, title='Time Series Plot'):
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Function to generate ACF plot
def plot_acf_for_time_series(data, title='ACF Plot'):
    plt.figure(figsize=(10, 6))
    plot_acf(data, lags=50, alpha=0.05)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()