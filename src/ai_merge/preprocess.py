import re
import unicodedata
import pandas as pd
from typing import Optional

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

def check_id_uniqueness(df: pd.DataFrame, id_var: str) -> dict:
    """
    Check if ID variable exists and is unique.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    id_var : str
        Name of the ID variable to check
    
    Returns
    -------
    dict
        Dictionary with validation statistics:
        - exists: bool
        - is_unique: bool
        - n_total: int
        - n_unique: int
        - n_missing: int
        - n_duplicates: int
        - duplicate_rate: float
        - duplicate_examples: dict (if duplicates exist)
    """
    if id_var not in df.columns:
        return {
            "exists": False,
            "is_unique": False,
            "error": f"Variable '{id_var}' not found in dataframe"
        }
    
    n_total = len(df)
    n_unique = df[id_var].nunique()
    n_missing = df[id_var].isna().sum()
    n_duplicates = n_total - n_unique
    
    is_unique = (n_duplicates == 0 and n_missing == 0)
    
    stats = {
        "exists": True,
        "is_unique": is_unique,
        "n_total": n_total,
        "n_unique": n_unique,
        "n_missing": n_missing,
        "n_duplicates": n_duplicates,
        "duplicate_rate": n_duplicates / n_total if n_total > 0 else 0,
    }
    
    # Get examples of duplicate IDs if they exist
    if n_duplicates > 0:
        duplicate_ids = df[id_var][df[id_var].duplicated(keep=False)].value_counts().head(5)
        stats["duplicate_examples"] = duplicate_ids.to_dict()
    
    return stats

def create_sequential_id(
    df: pd.DataFrame,
    new_id_name: str = "_merge_id",
    start: int = 1
) -> pd.DataFrame:
    """
    Create a sequential ID variable from start to n.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    new_id_name : str, default "_merge_id"
        Name for the new ID variable
    start : int, default 1
        Starting value for the ID sequence
    
    Returns
    -------
    pd.DataFrame
        Dataframe with new ID column added
    """
    df = df.copy()
    df[new_id_name] = range(start, start + len(df))
    return df

def ensure_unique_id(
    df: pd.DataFrame,
    id_var: Optional[str] = None,
    new_id_name: str = "_merge_id",
    force_create: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str, dict]:
    """
    Ensure dataframe has a unique ID variable. Create one if needed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    id_var : str, optional
        Name of the ID variable to check. If None, will create new ID.
    new_id_name : str, default "_merge_id"
        Name for the newly created ID variable
    force_create : bool, default False
        Always create new ID even if existing ID is unique
    verbose : bool, default True
        Print information about ID checking and creation
    
    Returns
    -------
    tuple[pd.DataFrame, str, dict]
        - Modified dataframe
        - Name of the ID variable to use
        - Dictionary with statistics about the ID check
    """
    df = df.copy()
    stats = {}
    
    # Case 1: Force create new ID
    if force_create:
        if verbose:
            print(f"Force creating new ID variable '{new_id_name}'.")
        df = create_sequential_id(df, new_id_name=new_id_name)
        stats = {
            "action": "created",
            "reason": "force_create=True",
            "id_used": new_id_name
        }
        return df, new_id_name, stats
    
    # Case 2: No ID variable specified
    if id_var is None:
        if verbose:
            print(f"No ID variable specified. Creating new ID variable '{new_id_name}'.")
        df = create_sequential_id(df, new_id_name=new_id_name)
        stats = {
            "action": "created",
            "reason": "no_id_specified",
            "id_used": new_id_name
        }
        return df, new_id_name, stats
    
    # Case 3: Check specified ID variable
    id_stats = check_id_uniqueness(df, id_var)
    
    # ID doesn't exist
    if not id_stats["exists"]:
        if verbose:
            print(f"ID variable '{id_var}' not found. Creating new ID variable '{new_id_name}'.")
        df = create_sequential_id(df, new_id_name=new_id_name)
        stats = {
            "action": "created",
            "reason": "id_not_found",
            "id_used": new_id_name,
            "original_id": id_var
        }
        return df, new_id_name, stats
    
    # ID exists and is unique
    if id_stats["is_unique"]:
        if verbose:
            print(f"✓ ID variable '{id_var}' is unique ({id_stats['n_unique']} observations).")
        stats = {
            "action": "used_existing",
            "id_used": id_var,
            **id_stats
        }
        return df, id_var, stats
    
    # ID exists but has issues
    if verbose:
        print(f"✗ ID variable '{id_var}' has issues:")
        if id_stats["n_duplicates"] > 0:
            print(f"  - {id_stats['n_duplicates']} duplicate observations ({id_stats['duplicate_rate']:.1%})")
        if id_stats["n_missing"] > 0:
            print(f"  - {id_stats['n_missing']} missing values")
        print(f"Creating new unique ID variable '{new_id_name}'.")
    
    df = create_sequential_id(df, new_id_name=new_id_name)
    stats = {
        "action": "created",
        "reason": "id_not_unique",
        "id_used": new_id_name,
        "original_id": id_var,
        **id_stats
    }
    return df, new_id_name, stats