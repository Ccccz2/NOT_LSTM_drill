import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.base import clone
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor


from .metrics import to_2d, eval_metrics_1d

@dataclass
class AggRow:
    eval_scheme: str
    task: str
    model: str
    target: str
    metric: str
    mean: float
    std: float

def wrap_multioutput(model, n_targets: int):
    if n_targets <= 1:
        return model
    if isinstance(model, MultiOutputRegressor):
        return model
    if isinstance(model, RandomForestRegressor):
        return model
    return MultiOutputRegressor(model)

def run_cv(X, Y, models, task_name, eval_scheme, splitter, groups: Optional[np.ndarray] = None):
    Xv = X.values
    Yv = to_2d(Y.values)
    target_names = list(Y.columns)
    n_targets = Yv.shape[1]

    oof_rows = []
    out_rows = []

    # groups 离散化（更稳）
    if groups is not None:
        groups = np.asarray(groups)
        groups = np.round(groups.astype(float), 3)

    for model_name, base_model in models.items():
        fold_id = 0
        fold_store = {t: {"R2": [], "RMSE": [], "MAE": []} for t in target_names}

        split_iter = splitter.split(Xv, Yv) if groups is None else splitter.split(Xv, Yv, groups)

        for tr_idx, te_idx in split_iter:
            Xtr, Xte = Xv[tr_idx], Xv[te_idx]
            Ytr, Yte = Yv[tr_idx], Yv[te_idx]

            model = clone(wrap_multioutput(base_model, n_targets))
            if n_targets == 1:
                ytr_fit = Ytr.ravel()  # (n,)
                yte_eval = Yte.ravel()  # (n,)

                model.fit(Xtr, ytr_fit)
                ypred = model.predict(Xte)
                ypred_eval = np.asarray(ypred).ravel()
                for idx, yt, yp in zip(te_idx, yte_eval, ypred_eval):
                    oof_rows.append({
                        "eval_scheme": eval_scheme,
                        "task": task_name,
                        "model": model_name,
                        "fold": int(fold_id),
                        "row_index": int(idx),
                        f"y_true_{target_names[0]}": float(yt),
                        f"y_pred_{target_names[0]}": float(yp),
                    })
                tname = target_names[0]
                m = eval_metrics_1d(yte_eval, ypred_eval)
                for k, v in m.items():
                    fold_store[tname][k].append(v)


            else:
                model.fit(Xtr, Ytr)
                Ypred = to_2d(model.predict(Xte))
                for k, idx in enumerate(te_idx):
                    row = {
                        "eval_scheme": eval_scheme,
                        "task": task_name,
                        "model": model_name,
                        "fold": int(fold_id),
                        "row_index": int(idx),
                    }
                    for j, tname in enumerate(target_names):
                        row[f"y_true_{tname}"] = float(Yte[k, j])
                        row[f"y_pred_{tname}"] = float(Ypred[k, j])
                    oof_rows.append(row)

                for j, tname in enumerate(target_names):
                    m = eval_metrics_1d(Yte[:, j], Ypred[:, j])
                    for k, v in m.items():
                        fold_store[tname][k].append(v)
            fold_id += 1

        for tname in target_names:
            for metric in ["R2", "RMSE", "MAE"]:
                vals = np.array(fold_store[tname][metric], dtype=float)
                out_rows.append(AggRow(
                    eval_scheme=eval_scheme,
                    task=task_name,
                    model=model_name,
                    target=tname,
                    metric=metric,
                    mean=float(vals.mean()),
                    std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                ))

        if n_targets > 1:
            for metric in ["R2", "RMSE", "MAE"]:
                per_target_means = np.array([np.mean(fold_store[t][metric]) for t in target_names], dtype=float)
                out_rows.append(AggRow(
                    eval_scheme=eval_scheme,
                    task=task_name,
                    model=model_name,
                    target="MACRO_AVG",
                    metric=metric,
                    mean=float(per_target_means.mean()),
                    std=float(per_target_means.std(ddof=1)) if len(per_target_means) > 1 else 0.0
                ))

    long_df = pd.DataFrame([r.__dict__ for r in out_rows])
    oof_df = pd.DataFrame(oof_rows)
    return long_df, oof_df
