from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


def build(random_state: int, params: dict):
    reg = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        **params
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", reg),
    ])
