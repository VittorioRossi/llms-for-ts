import numpy as np

def mean_absolute_error(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y_true:np.array, y_pred:np.array) -> dict:

    assert y_true.shape == y_pred.shape, f'Shapes do not match: {y_true.shape} != {y_pred.shape}'

    nan_mask = np.isnan(y_true) | np.isnan(y_pred)
    total_elements = y_true.size
    nan_elements = np.sum(nan_mask)
    nan_fraction = nan_elements / total_elements


    mask = ~nan_mask
    mask_zeros = (y_true != 0)
    zero_fraction = (1-np.sum(mask_zeros)) / len(y_true)
    return {
        'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
        'mse': mean_squared_error(y_true[mask], y_pred[mask]),
        'rmse': root_mean_squared_error(y_true[mask], y_pred[mask]),
        'mape': mean_absolute_percentage_error(y_true[mask_zeros & mask], y_pred[mask_zeros & mask]),
        'nan_fraction': nan_fraction,
        'zero_fraction': zero_fraction
    }