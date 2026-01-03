# run_all.py
# -*- coding: utf-8 -*-
"""
Single entry point:
- 加载配置
- 加载数据
- 读取models中具体模型
- 运行A + B两个任务  +  两个评估方案 (KFold + GroupKFold)
    任务A：X=[钻速(mm/s), 角速度(1/s), 推力(kN), 扭矩(N·m), 岩石可钻性指数] -> Y=[单轴抗压强度(MPa)]
    任务B：X=[钻速(mm/s), 角速度(1/s), 推力(kN), 扭矩(N·m)] -> Y=[岩石可钻性指数, 单轴抗压强度(MPa)]
    评估：
    - KFold(5) : 常规5折
    - GroupKFold : 按 UCS 分组
- 保存 cv_results_long.csv + summary/latex/word tables
- 保存 config_snapshot.json + run.log
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

import config as CFG
from models import build_models

from utils.data import load_data, assert_columns, clean_xy
from utils.cv import run_cv
from utils.tables import build_summary_table, save_latex_table, save_word_tables
from utils.io import ensure_dir, setup_logger, save_config_snapshot


def _get_groups_for_ucs(Y_df: pd.DataFrame) -> np.ndarray:
    # GroupKFold groups: UCS 近似区间避免浮点数浮动
    if CFG.COL_UCS not in Y_df.columns:
        raise KeyError(f"GroupKFold requires '{CFG.COL_UCS}' in Y columns, but it's not found.")
    g = pd.to_numeric(Y_df[CFG.COL_UCS], errors="coerce").to_numpy()
    g = np.round(g.astype(float), 3)
    return g


def _make_out_dirs(out_dir: Path) -> dict[str, Path]:
    # 创建输出目录并返回路径
    paths = {
        "root": out_dir,
        "artifacts": out_dir / "artifacts",
        "figures": out_dir / "figures",
        "summary_tables": out_dir / "summary_tables",
        "latex_tables": out_dir / "latex_tables",
        "word_tables": out_dir / "word_tables",
    }
    for p in paths.values():
        ensure_dir(p)
    return paths


def _snapshot_config() -> dict:
    # 创建配置快照
    snap = {
        "RUN_TAG": getattr(CFG, "RUN_TAG", None),
        "DATA_PATH": str(getattr(CFG, "DATA_PATH", "")),
        "OUT_DIR": str(getattr(CFG, "OUT_DIR", "")),
        "RANDOM_STATE": getattr(CFG, "RANDOM_STATE", None),
        "KFOLD_SPLITS": getattr(CFG, "KFOLD_SPLITS", None),
        "GROUPK_SPLITS_MAX": getattr(CFG, "GROUPK_SPLITS_MAX", None),
        "TASKS": getattr(CFG, "TASKS", None),
        "MODEL_PARAMS": getattr(CFG, "MODEL_PARAMS", None),
        "python_datetime": datetime.now().isoformat(timespec="seconds"),
    }
    return snap


def main():
    # -----------------------
    # 0) 输出 dirs & logger
    # -----------------------
    out_dir: Path = CFG.OUT_DIR
    paths = _make_out_dirs(out_dir)

    logger = setup_logger(paths["root"] / "run.log")
    logger.info("=== RUN START ===")
    logger.info(f"DATA_PATH: {CFG.DATA_PATH}")
    logger.info(f"OUT_DIR  : {CFG.OUT_DIR}")

    # 保存config快照
    cfg_snapshot = _snapshot_config()
    save_config_snapshot(paths["root"] / "config_snapshot.json", cfg_snapshot)

    # -----------------------
    # 1) 数据加载
    # -----------------------
    df = load_data(CFG.DATA_PATH)
    logger.info(f"Loaded data: rows={len(df)}, cols={len(df.columns)}")

    # -----------------------
    # 2) 模型构建
    # -----------------------
    models = build_models(CFG.RANDOM_STATE, CFG.MODEL_PARAMS)
    logger.info(f"Models: {list(models.keys())}")

    # -----------------------
    # 3) 运行A + B两个任务  +  两个评估方案
    # -----------------------
    all_long = []
    all_oof = []

    for task_name, spec in CFG.TASKS.items():
        x_cols = spec["X"]
        y_cols = spec["Y"]

        # 验证列
        assert_columns(df, x_cols, tag=f"{task_name} X")
        assert_columns(df, y_cols, tag=f"{task_name} Y")

        # 清理列序并删除缺失值
        X_df, Y_df = clean_xy(df, x_cols, y_cols)
        logger.info(
            f"[{task_name}] X={x_cols}, Y={y_cols} | after clean: n={len(X_df)}"
        )

        # 3.1 KFold
        kf = KFold(n_splits=CFG.KFOLD_SPLITS, shuffle=True, random_state=CFG.RANDOM_STATE)
        logger.info(f"[{task_name}] Running KFold (n_splits={CFG.KFOLD_SPLITS}) ...")
        long_k, oof_k = run_cv(
            X=X_df,
            Y=Y_df,
            models=models,
            task_name=task_name,
            eval_scheme="KFold",
            splitter=kf,
            groups=None,
        )
        all_long.append(long_k)
        all_oof.append(oof_k)

        # 3.2 GroupKFold (by UCS groups)
        try:
            groups = _get_groups_for_ucs(Y_df)
            n_groups = int(pd.Series(groups).nunique())
            n_splits = min(getattr(CFG, "GROUPK_SPLITS_MAX", 4), n_groups)
            if n_splits < 2:
                logger.warning(f"[{task_name}] GroupKFold skipped: n_groups={n_groups} < 2")
            else:
                gkf = GroupKFold(n_splits=n_splits)
                logger.info(f"[{task_name}] Running GroupKFold (n_splits={n_splits}, n_groups={n_groups}) ...")
                long_g, oof_g = run_cv(
                    X=X_df,
                    Y=Y_df,
                    models=models,
                    task_name=task_name,
                    eval_scheme="GroupKFold(UCS)",
                    splitter=gkf,
                    groups=groups,
                )
                all_long.append(long_g)
                all_oof.append(oof_g)
        except KeyError as e:
            logger.warning(f"[{task_name}] GroupKFold skipped: {e}")

    cv_long = pd.concat(all_long, ignore_index=True)
    oof_df = pd.concat(all_oof, ignore_index=True)

    # 保存到 artifacts
    oof_path = paths["artifacts"] / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved OOF predictions: {oof_path}")

    # -----------------------
    # 6) Figures (OOF scatter + R2 bar)
    # -----------------------
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]  # 依次尝试
        mpl.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块

        # 识别所有 target（y_true_ 开头）
        true_cols = [c for c in oof_df.columns if c.startswith("y_true_")]

        # 6.1 OOF scatter: y_true vs y_pred
        for (task, scheme, model), sub in oof_df.groupby(["task", "eval_scheme", "model"]):
            for tc in true_cols:
                tname = tc.replace("y_true_", "")
                pc = f"y_pred_{tname}"
                if pc not in sub.columns:
                    continue

                yy = sub[tc].astype(float).to_numpy()
                yp = sub[pc].astype(float).to_numpy()

                plt.figure()
                plt.scatter(yy, yp)
                mn = float(min(yy.min(), yp.min()))
                mx = float(max(yy.max(), yp.max()))
                plt.plot([mn, mx], [mn, mx])  # y=x 参考线
                plt.xlabel("y_true")
                plt.ylabel("y_pred")
                plt.title(f"{task} | {scheme} | {model} | {tname}")

                fname = f"OOF_{task}_{scheme}_{model}_{tname}.png"
                fname = (
                    fname.replace(":", "_")
                    .replace("->", "to")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("|", "_")
                    .replace(" ", "")
                    .replace("/", "_")
                )
                plt.tight_layout()
                plt.savefig(paths["figures"] / fname, dpi=220)
                plt.close()

        # 6.2 R2 bar plot: from cv_long
        for (task, scheme, target), sub in cv_long.groupby(["task", "eval_scheme", "target"]):
            ss = sub[sub["metric"] == "R2"].copy()
            if ss.empty:
                continue
            ss = ss.sort_values("mean", ascending=False)

            plt.figure()
            plt.bar(ss["model"], ss["mean"])
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("R2 (mean)")
            plt.title(f"{task} | {scheme} | {target}")

            fname = f"R2_{task}_{scheme}_{target}.png"
            fname = (
                fname.replace(":", "_")
                .replace("->", "to")
                .replace("[", "")
                .replace("]", "")
                .replace("|", "_")
                .replace(" ", "")
                .replace("/", "_")
            )
            plt.tight_layout()
            plt.savefig(paths["figures"] / fname, dpi=220)
            plt.close()

        logger.info(f"Saved figures to: {paths['figures']}")

    except Exception as e:
        logger.warning(f"Figure generation skipped: {e}")

    # -----------------------
    # 4) 保存 long results
    # -----------------------
    long_csv_path = paths["root"] / "cv_results_long.csv"
    cv_long.to_csv(long_csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved long CV results: {long_csv_path}")

    # -----------------------
    # 5)构建summary table + export latex/word
    # -----------------------
    all_tables = []

    # 遍历：task × eval_scheme × target
    tasks = cv_long["task"].dropna().unique().tolist()
    schemes = cv_long["eval_scheme"].dropna().unique().tolist()

    for task in tasks:
        for scheme in schemes:
            targets = (
                cv_long[(cv_long["task"] == task) & (cv_long["eval_scheme"] == scheme)]["target"]
                .dropna().unique().tolist()
            )
            for target in targets:
                df_tbl = build_summary_table(cv_long, task=task, eval_scheme=scheme, target=target)
                title = f"{task} | {scheme} | target={target}"
                all_tables.append((title, df_tbl))

                # 每张表存一份 summary csv（可选）
                safe_name = (
                    title.replace(":", "_")
                    .replace("->", "to")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("|", "_")
                    .replace(" ", "")
                    .replace("/", "_")
                )
                out_csv = paths["summary_tables"] / f"{safe_name}.csv"
                if not df_tbl.empty:
                    df_tbl.to_csv(out_csv, index=False, encoding="utf-8-sig")

                # 每张表存一份 latex（可选）
                out_tex = paths["latex_tables"] / f"{safe_name}.tex"
                save_latex_table(
                    df_tbl,
                    out_path=str(out_tex),
                    caption=title,
                    label=f"tab:{safe_name.lower()[:40]}"
                )

    # Word：把所有表写进同一个 docx
    out_docx = paths["word_tables"] / "all_tables.docx"
    save_word_tables(all_tables, out_docx=str(out_docx))


if __name__ == "__main__":
    main()
