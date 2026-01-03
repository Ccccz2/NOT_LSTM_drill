# models/__init__.py
from .MLP.MLP_model import build as build_mlp
from .SVR.SVR_model import build as build_svr
from .Random_Forest.Random_Forest_model import build as build_rf

def build_models(random_state: int, params: dict):
    models = {
        "MLP": build_mlp(random_state, params.get("MLP", {})),
        "SVR": build_svr(random_state, params.get("SVR", {})),
        "Random_Forest": build_rf(random_state, params.get("Random_Forest", {})),
    }
    try:
        from .XGBoost.XGBoost_model import build as build_xgb
        models["XGBoost"] = build_xgb(random_state, params.get("XGBoost", {}))
    except Exception:
        print("[WARN] xgboost 未安装或不可用：将跳过 XGBoost。")
    return models
