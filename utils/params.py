# utils/params.py
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor

def _unwrap_estimator(est):
    """拆掉 MultiOutputRegressor，拿到内部 estimator（通常是 Pipeline 或模型）"""
    if isinstance(est, MultiOutputRegressor):
        return est.estimator
    return est

def _extract_pipeline_info(pipe: Pipeline) -> Dict[str, Any]:
    info = {"type": "Pipeline", "steps": []}
    for name, step in pipe.steps:
        step_info = {
            "step": name,
            "class": step.__class__.__name__,
            # deep=False 只拿“本层”超参，避免拿到所有嵌套对象
            "params": step.get_params(deep=False) if hasattr(step, "get_params") else {}
        }
        info["steps"].append(step_info)
    return info

def extract_hyperparams(models_for_run: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for model_name, est in models_for_run.items():
        base = _unwrap_estimator(est)

        if isinstance(base, Pipeline):
            out[model_name] = _extract_pipeline_info(base)
        else:
            out[model_name] = {
                "type": base.__class__.__name__,
                "params": base.get_params(deep=False) if hasattr(base, "get_params") else {}
            }
    return out

def save_hyperparams(models_for_run: Dict[str, Any], out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    hp = extract_hyperparams(models_for_run)

    # 1) JSON（完整）
    json_path = out_dir / f"hyperparams_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hp, f, ensure_ascii=False, indent=2)

    # 2) CSV（扁平化，方便看）
    rows = []
    for model_name, info in hp.items():
        if info.get("type") == "Pipeline":
            # 找到 pipeline 里最后一步名为 model 的那一层
            model_step = None
            for s in info["steps"]:
                if s["step"] == "model":
                    model_step = s
                    break
            rows.append({
                "model": model_name,
                "wrapper": "Pipeline",
                "final_step_class": model_step["class"] if model_step else None,
                "final_step_params": json.dumps(model_step["params"], ensure_ascii=False) if model_step else None
            })
        else:
            rows.append({
                "model": model_name,
                "wrapper": info.get("type"),
                "final_step_class": info.get("type"),
                "final_step_params": json.dumps(info.get("params", {}), ensure_ascii=False)
            })

    csv_path = out_dir / f"hyperparams_{tag}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    return json_path, csv_path
