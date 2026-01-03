from xgboost import XGBRegressor

def build(random_state: int, params: dict):
    return XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        **params
    )
