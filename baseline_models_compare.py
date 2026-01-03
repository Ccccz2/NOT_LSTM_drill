import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

'''
现在的部分已经由run_all文件实现，本文件可以不使用，目前已经将task A对齐，可以实现单目标对照，task B的多输出还没有调整好
'''


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False


# =========================
# 0. 配置区（分类数据X与Y）
# =========================
DATA_PATH = r"cz_drill.xlsx"  # 运行脚本时放同目录；或写完整路径
COL_V   = "钻速(mm/s)"
COL_W   = "角速度(1/s)"
COL_F   = "推力(kN)"
COL_TQ  = "扭矩(N·m)"
COL_ID  = "岩石可钻性指数"
COL_UCS = "单轴抗压强度(MPa)"

TASKS = {
    "Task_A_5toUCS": {
        "X": [COL_V, COL_W, COL_F, COL_TQ, COL_ID],
        "Y": [COL_UCS],
    },
    "Task_B_4toIdUCS": {
        "X": [COL_V, COL_W, COL_F, COL_TQ],
        "Y": [COL_ID, COL_UCS],
    },
}
GROUP_COL_CANDIDATES = ["group", "Group", "GROUP", "set", "Set", "SET"]  # 若存在分组列则启用GroupKFold
SEED = 42

# 交叉验证设置：小样本建议
N_SPLITS = 5


# =========================
# 1. 工具函数
# =========================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_cv(best_estimator, X, y, cv):
    """用给定cv做外层评估：返回每折指标 + 均值"""
    rows = []
    fold = 0
    for train_idx, test_idx in cv.split(X, y):
        fold += 1
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = best_estimator
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        rows.append({
            "fold": fold,
            "R2": r2_score(y_te, pred),
            "RMSE": rmse(y_te, pred),
            "MAE": mean_absolute_error(y_te, pred),
        })
    df = pd.DataFrame(rows)
    mean_row = df[["R2", "RMSE", "MAE"]].mean().to_dict()
    mean_row["fold"] = "mean"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df

def pick_group_column(df):
    """如果数据里有分组列（比如4组数据），优先启用GroupKFold，避免信息泄漏"""
    for c in GROUP_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


# =========================
# 2. 读取数据
# =========================
df = pd.read_excel(DATA_PATH)

# 处理 TARGET_COL（支持索引或列名）
df = pd.read_excel(DATA_PATH)

# =========================
# 3. 统一预处理（缺失值 + 标准化）
#   - SVR/MLP需要标准化
#   - RF/XGB不强制，但为了统一脚手架，只在各自pipeline里决定是否加Scaler
# =========================
numeric_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_preprocess_no_scale = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

# ColumnTransformer：只对数值列做处理（当前X已是数值列）
preprocess_scale = ColumnTransformer(
    transformers=[("num", numeric_preprocess, X.columns)],
    remainder="drop"
)
preprocess_noscale = ColumnTransformer(
    transformers=[("num", numeric_preprocess_no_scale, X.columns)],
    remainder="drop"
)

# =========================
# 4. CV策略：优先GroupKFold，否则KFold
# =========================
if groups is not None and groups.nunique() >= 2:
    cv_outer = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
    cv_for_search = cv_outer  # 为了小样本稳定，这里搜索也用同一cv
    split_iter = cv_outer.split(X, y, groups=groups)
else:
    cv_outer = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_for_search = cv_outer
    split_iter = cv_outer.split(X, y)

# =========================
# 5. 定义模型 + 参数网格
# =========================
models = {}

# 5.1 SVR（RBF核）
models["SVR"] = {
    "pipeline": Pipeline(steps=[
        ("prep", preprocess_scale),
        ("model", SVR(kernel="rbf"))
    ]),
    "param_grid": {
        "model__C": [1, 10, 100, 300],
        "model__epsilon": [0.01, 0.1, 0.2],
        "model__gamma": ["scale", "auto"],
    }
}

# 5.2 Random Forest
models["RandomForest"] = {
    "pipeline": Pipeline(steps=[
        ("prep", preprocess_noscale),
        ("model", RandomForestRegressor(
            random_state=SEED,
            n_estimators=500
        ))
    ]),
    "param_grid": {
        "model__max_depth": [None, 3, 5, 8],
        "model__min_samples_split": [2, 4, 8],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.8, 1.0],
    }
}

# 5.3 XGBoost（回归）
if XGBOOST_OK:
    models["XGBoost"] = {
        "pipeline": Pipeline(steps=[
            ("prep", preprocess_noscale),
            ("model", XGBRegressor(
                random_state=SEED,
                n_estimators=800,
                objective="reg:squarederror",
                tree_method="hist"
            ))
        ]),
        "param_grid": {
            "model__max_depth": [2, 3, 4, 6],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__reg_lambda": [1.0, 5.0, 10.0],
        }
    }

# 5.4 MLP（浅层神经网络对照）
models["MLP"] = {
    "pipeline": Pipeline(steps=[
        ("prep", preprocess_scale),
        ("model", MLPRegressor(
            random_state=SEED,
            max_iter=3000,
            early_stopping=True,
            n_iter_no_change=30
        ))
    ]),
    "param_grid": {
        "model__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
        "model__alpha": [1e-5, 1e-4, 1e-3],
        "model__learning_rate_init": [1e-3, 5e-4, 1e-4],
        "model__activation": ["relu", "tanh"],
    }
}


# =========================
# 6. 统一训练 + 网格搜索 + 外层评估
# =========================
all_summary = []
all_folds = []


for task_name, spec in TASKS.items():
    X = df[spec["X"]].copy()
    Y = df[spec["Y"]].copy()

    X = X.select_dtypes(include=[np.number])
    Y = Y.select_dtypes(include=[np.number])

    for name, cfg in models.items():
        print(f"\n=== {name} ===")

        pipe = cfg["pipeline"]
        grid = cfg["param_grid"]

        # 用R2作为搜索目标（你也可以改成neg_root_mean_squared_error）
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="r2",
            cv=cv_for_search,
            n_jobs=-1
        )

        if groups is not None and groups.nunique() >= 2:
            search.fit(X, y, model__sample_weight=None, **{"groups": groups})
        else:
            search.fit(X, y)

        print("Best params:", search.best_params_)
        print("Best CV R2:", search.best_score_)

        # 外层折评估（为了报告稳定）
        if groups is not None and groups.nunique() >= 2:
            folds_df = evaluate_cv(search.best_estimator_, X, y, cv_outer if hasattr(cv_outer, "split") else cv_outer)
            # GroupKFold需要groups，这里重写一下split
            rows = []
            fold = 0
            for train_idx, test_idx in cv_outer.split(X, y, groups=groups):
                fold += 1
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                m = search.best_estimator_
                m.fit(X_tr, y_tr)
                pred = m.predict(X_te)
                rows.append({"fold": fold, "R2": r2_score(y_te, pred), "RMSE": rmse(y_te, pred), "MAE": mean_absolute_error(y_te, pred)})
            folds_df = pd.DataFrame(rows)
            mean_row = folds_df[["R2", "RMSE", "MAE"]].mean().to_dict()
            mean_row["fold"] = "mean"
            folds_df = pd.concat([folds_df, pd.DataFrame([mean_row])], ignore_index=True)
        else:
            folds_df = evaluate_cv(search.best_estimator_, X, y, cv_outer)

        folds_df.insert(0, "model", name)
        all_folds.append(folds_df)

        mean_metrics = folds_df[folds_df["fold"] == "mean"].iloc[0].to_dict()
        all_summary.append({
            "model": name,
            "R2_mean": mean_metrics["R2"],
            "RMSE_mean": mean_metrics["RMSE"],
            "MAE_mean": mean_metrics["MAE"],
            "best_params": str(search.best_params_)
        })

# 汇总
summary_df = pd.DataFrame(all_summary).sort_values(by="RMSE_mean", ascending=True)
folds_df = pd.concat(all_folds, ignore_index=True)

print("\n\n===== Summary (sorted by RMSE_mean) =====")
print(summary_df)

# 保存结果
summary_df.to_csv("baseline_models_summary.csv", index=False, encoding="utf-8-sig")
folds_df.to_csv("baseline_models_folds.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter("baseline_models_results.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    folds_df.to_excel(writer, sheet_name="folds", index=False)

print("\nSaved: baseline_models_summary.csv / baseline_models_folds.csv / baseline_models_results.xlsx")
