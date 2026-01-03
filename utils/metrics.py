import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def to_2d(arr):
    arr = np.asarray(arr)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def eval_metrics_1d(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }
