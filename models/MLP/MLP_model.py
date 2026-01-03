from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def build(random_state: int, params: dict):
    reg = MLPRegressor(random_state=random_state, **params)
    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])