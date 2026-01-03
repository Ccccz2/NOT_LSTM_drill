from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


def build(random_state: int, params: dict):
    reg = XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        **params
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", reg),
    ])
