from datetime import datetime
from pathlib import Path

# 列名
COL_V   = "钻速(mm/s)"
COL_W   = "角速度(1/s)"
COL_F   = "推力(kN)"
COL_TQ  = "扭矩(N·m)"
COL_ID  = "岩石可钻性指数"
COL_UCS = "单轴抗压强度(MPa)"

TASKS = {
    "Task_A: 5->UCS": {
        "X": [COL_V, COL_W, COL_F, COL_TQ, COL_ID],
        "Y": [COL_UCS],
    },
    "Task_B: 4->[Id,UCS]": {
        "X": [COL_V, COL_W, COL_F, COL_TQ],
        "Y": [COL_ID, COL_UCS],
    },
}

# 路径
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "cz_drill.xlsx"

RUN_TAG = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
OUT_DIR = ROOT / "results" / RUN_TAG

# CV
RANDOM_STATE = 42
KFOLD_SPLITS = 5
GROUPK_SPLITS_MAX = 4

# 模型超参
MODEL_PARAMS = {

    "MLP": dict(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=5e-4,
        max_iter=12000,
        early_stopping=True,
        n_iter_no_change=20,
        tol=1e-4,
    ),

    "SVR": dict(C=50, epsilon=0.05, kernel="rbf"),

    "Random_Forest": dict(n_estimators=800),

    "XGBoost": dict(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
    ),
}


# =========================
# 自动调参(GridSearchCV)
# =========================
ENABLE_TUNING = True      # 先保持 False,run_all 接好后再改为True!!!!!!!!!!
TUNING_SCORING = "r2"
TUNING_N_JOBS = -1         # -1 用全部CPU

# 参数网格：用于 GridSearchCV
PARAM_GRIDS = {
    "SVR": {
        "model__C": [1, 10, 100, 300],
        "model__epsilon": [0.01, 0.05, 0.1, 0.2],
        "model__gamma": ["scale", "auto"],
    },
    "Random_Forest": {
        "model__n_estimators": [300, 800, 1200],
        "model__max_depth": [None, 3, 5, 8],
        "model__min_samples_split": [2, 4, 8],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.8, 1.0],
    },
    "MLP": {
        "model__hidden_layer_sizes": [(16,), (32,), (64, 32), (64, 32, 16)],
        "model__alpha": [1e-5, 1e-4, 1e-3],
        "model__learning_rate_init": [ 5e-4, 1e-4],
        "model__activation": ["relu", "tanh"],
    },
    "XGBoost": {
        "model__n_estimators": [300, 800, 1200],
        "model__max_depth": [2, 3, 4, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    },
}