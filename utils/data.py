import pandas as pd

def assert_columns(df: pd.DataFrame, cols: list[str], tag: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{tag}] 缺少列：{missing}\n"
            f"当前表头：{list(df.columns)}\n"
            f"确保Excel表头与脚本列名完全一致（含大小写、单位、·符号）。"
        )

def load_data(path):
    df = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    return df.dropna(how="all").copy()

def clean_xy(df: pd.DataFrame, x_cols: list[str], y_cols: list[str]):
    sub = df[x_cols + y_cols].copy()
    for c in x_cols + y_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(axis=0, how="any").reset_index(drop=True)
    return sub[x_cols], sub[y_cols]
