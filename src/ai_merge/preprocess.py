import re
import unicodedata
import pandas as pd

def normalize_series(ser: pd.Series) -> pd.Series:
    ser = ser.astype("string")
    ser = ser.str.normalize("NFKC").str.strip().str.lower()
    ser = ser.str.replace(r"\s+", " ", regex=True)
    return ser

def add_normalized_columns(
    df: pd.DataFrame,
    cols: list[str],
    corrections: dict[str, dict] | None = None,
    suffix: str = "__norm",
) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        norm_col = f"{c}{suffix}"
        df[norm_col] = normalize_series(df[c])
        if corrections and (c in corrections):
            mapping = corrections[c]
            df[norm_col] = df[norm_col].map(lambda x: mapping.get(x, x))
    return df

def build_key_series(df: pd.DataFrame, cols: list[str], sep: str = "\x1f") -> pd.Series:
    if not cols:
        return pd.Series([""], index=df.index, dtype="string")
    return df[cols].astype("string").fillna("").agg(sep.join, axis=1)