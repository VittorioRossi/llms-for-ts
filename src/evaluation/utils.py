import numpy as np

def mean_absolute_error(y_true:float, y_pred:float) -> float:
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true:float, y_pred:float) -> float:
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true:float, y_pred:float) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true:float, y_pred:float) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y_true:np.array, y_pred:np.array) -> dict:
    # check that both have shape (n)
    if y_true.shape != y_pred.shape:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

    assert y_true.shape == y_pred.shape, f'Shapes do not match: {y_true.shape} != {y_pred.shape}'

    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }