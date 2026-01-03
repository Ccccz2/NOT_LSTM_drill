from sklearn.ensemble import RandomForestRegressor

def build(random_state: int, params: dict):
    return RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        **params
    )
