# utils/report.py
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor


def _safe_tag(s: str) -> str:
    """Windows 文件名安全化：去掉 : * ? 等字符"""
    return re.sub(r'[\\/:*?"<>|]+', "_", s)


def _unwrap_estimator(est):
    """拆掉 MultiOutputRegressor 外壳，拿到内部 estimator（Pipeline/模型）"""
    if isinstance(est, MultiOutputRegressor):
        return est.estimator
    return est


def _extract_pipeline_info(pipe: Pipeline) -> Dict[str, Any]:
    info = {"wrapper": "Pipeline", "steps": []}
    for step_name, step in pipe.steps:
        info["steps"].append({
            "step": step_name,
            "class": step.__class__.__name__,
            "params": step.get_params(deep=False) if hasattr(step, "get_params") else {}
        })
    return info


def _extract_final_params(models_for_run: Dict[str, Any],
                          task: str,
                          eval_scheme: str,
                          tag: str) -> pd.DataFrame:
    rows = []
    for model_name, est in models_for_run.items():
        is_multi = isinstance(est, MultiOutputRegressor)
        base = _unwrap_estimator(est)

        if isinstance(base, Pipeline):
            info = _extract_pipeline_info(base)
            final_step = None
            for s in info["steps"]:
                if s["step"] == "model":
                    final_step = s
                    break
            rows.append({
                "task": task,
                "eval_scheme": eval_scheme,
                "tag": tag,
                "model": model_name,
                "is_multioutput_wrapped": is_multi,
                "final_step_class": final_step["class"] if final_step else None,
                "final_step_params_json": json.dumps(final_step["params"], ensure_ascii=False) if final_step else None,
                "pipeline_steps_json": json.dumps(info["steps"], ensure_ascii=False),
            })
        else:
            rows.append({
                "task": task,
                "eval_scheme": eval_scheme,
                "tag": tag,
                "model": model_name,
                "is_multioutput_wrapped": is_multi,
                "final_step_class": base.__class__.__name__,
                "final_step_params_json": json.dumps(base.get_params(deep=False), ensure_ascii=False)
                if hasattr(base, "get_params") else "{}",
                "pipeline_steps_json": None,
            })
    return pd.DataFrame(rows)


def _metrics_to_mean_std_table(cv_long: pd.DataFrame, nd: int = 3) -> pd.DataFrame:
    """
    输入：cv_long = 保存的长表（eval_scheme/task/model/target/metric/mean/std）
    输出：每个 task×eval×target×model 一行，R2/RMSE/MAE 变成 mean±std
    """
    df = cv_long.copy()
    if df.empty:
        return df

    def fmt(m, s):
        if pd.isna(m):
            return ""
        if pd.isna(s):
            s = 0.0
        return f"{m:.{nd}f}±{s:.{nd}f}"

    df["mean_std"] = df.apply(lambda r: fmt(r["mean"], r["std"]), axis=1)

    pivot = df.pivot_table(
        index=["task", "eval_scheme", "target", "model"],
        columns="metric",
        values="mean_std",
        aggfunc="first"
    ).reset_index()

    # 固定列顺序（如果缺某列也不会报错）
    cols = ["task", "eval_scheme", "target", "model"]
    for c in ["R2", "RMSE", "MAE"]:
        if c in pivot.columns:
            cols.append(c)
    return pivot[cols]


def export_run_report(out_dir: Path,
                      cv_long: pd.DataFrame,
                      tuning_dfs: List[pd.DataFrame],
                      final_model_snapshots: List[Dict[str, Any]],
                      filename: str = "hyperparams_all.xlsx") -> Path:
    """
    out_dir/artifacts 下输出一个总表 Excel：
    - final_params
    - tuning_best
    - metrics_summary
    - merged_report
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) final_params（由 snapshots 生成）
    final_params_list = []
    for snap in final_model_snapshots:
        final_params_list.append(_extract_final_params(
            models_for_run=snap["models_for_run"],
            task=snap["task"],
            eval_scheme=snap["eval_scheme"],
            tag=snap["tag"]
        ))
    final_params_df = pd.concat(final_params_list, ignore_index=True) if final_params_list else pd.DataFrame()

    # 2) tuning_best（汇总 tune_models 的输出）
    tuning_best_df = pd.concat(tuning_dfs, ignore_index=True) if tuning_dfs else pd.DataFrame()
    # 保证有列
    for c in ["task", "eval_scheme", "tag", "model", "tuned", "best_score", "best_params"]:
        if c not in tuning_best_df.columns:
            tuning_best_df[c] = None

    # 3) metrics_summary（cv_long -> mean±std 透视表）
    metrics_summary_df = _metrics_to_mean_std_table(cv_long)

    # 4) merged_report（四表合并：final_params + tuning_best + metrics_summary）
    # metrics_summary 中 key：task/eval_scheme/model/target
    # tuning_best 中 key：task/eval_scheme/model（tag 一样更准确）
    # final_params 中 key：task/eval_scheme/tag/model
    merged = None

    # 先把 tuning_best 和 final_params 通过 task/eval/tag/model 合并（优先 tag）
    if not final_params_df.empty:
        tb = tuning_best_df.copy()
        # 有些情况下 tuning_best 可能没 tag/task/eval（需要在 run_all 里补上），可能要修改
        keys_fp = ["task", "eval_scheme", "tag", "model"]
        keys_tb = ["task", "eval_scheme", "tag", "model"] if all(k in tb.columns for k in keys_fp) else ["model"]

        merged1 = final_params_df.merge(
            tb,
            how="left",
            on=keys_tb,
            suffixes=("", "_tuning")
        )
    else:
        merged1 = tuning_best_df.copy()

    # 再把 metrics_summary 加进去：按 task/eval/model/target 合并（target 不在 final_params 里，会扩增行）
    if not metrics_summary_df.empty and not merged1.empty:
        merged = metrics_summary_df.merge(
            merged1,
            how="left",
            on=["task", "eval_scheme", "model"],
            suffixes=("", "_params")
        )
    elif not metrics_summary_df.empty:
        merged = metrics_summary_df.copy()
    else:
        merged = merged1

    # 输出 Excel
    out_path = out_dir / filename
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        final_params_df.to_excel(writer, sheet_name="final_params", index=False)
        tuning_best_df.to_excel(writer, sheet_name="tuning_best", index=False)
        metrics_summary_df.to_excel(writer, sheet_name="metrics_summary", index=False)
        merged.to_excel(writer, sheet_name="merged_report", index=False)

    return out_path
