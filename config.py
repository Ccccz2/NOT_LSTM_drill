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
        learning_rate_init=1e-3,
        max_iter=3000,
        early_stopping=True,
        n_iter_no_change=40,
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