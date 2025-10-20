import re
import unicodedata
import pandas as pd

def normalize_text(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_series(ser: pd.Series) -> pd.Series:
    ser = ser.astype("string")
    ser = ser.str.normalize("NFKC").str.strip().str.lower()
    ser = ser.str.replace(r"\s+", " ", regex=True)
    return ser

def apply_mapping_series(ser: pd.Series, mapping: dict | None) -> pd.Series:
    if not mapping:
        return ser
    # Assume mapping keys and values are already normalized
    return ser.map(lambda x: mapping.get(x, x))

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
            df[norm_col] = apply_mapping_series(df[norm_col], corrections[c])
    return df

def build_key_series(df: pd.DataFrame, cols: list[str], sep: str = "\x1f") -> pd.Series:
    # Creates a single string key by joining normalized columns
    if not cols:
        return pd.Series([""], index=df.index, dtype="string")
    key = df[cols].astype("string").fillna("").agg(sep.join, axis=1)
    return key