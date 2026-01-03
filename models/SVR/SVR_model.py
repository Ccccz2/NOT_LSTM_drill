from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def build(random_state: int, params: dict):
    # SVR 没有 random_state，这里仅为保留接口一致性
    reg = SVR(**params)
    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])
