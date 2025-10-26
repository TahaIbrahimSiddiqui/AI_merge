import re
import unicodedata
import pandas as pd


def normalize_text(s: str | None) -> str | None:
    """
    Normalize a single string for fuzzy matching.
    
    This function standardizes text by:
    - Converting Unicode characters to a standard form (NFKC)
    - Converting to lowercase
    - Removing leading/trailing whitespace
    - Collapsing multiple spaces into single spaces
    
    Example:
        >>> normalize_text("  Apple   Inc.™  ")
        "apple inc.™"
    """
    if s is None:
        return None
    s = str(s)
    # NFKC normalization: handles unicode variants (e.g., ™ → TM, ﬁ → fi)
    s = unicodedata.normalize("NFKC", s)
    # Lowercase and strip whitespace
    s = s.strip().lower()
    # Collapse multiple whitespace into single space
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_series(ser: pd.Series) -> pd.Series:
    """
    Normalize an entire pandas Series of strings for fuzzy matching.
    
    
    Performs the same operations as normalize_text():
    - Unicode normalization (NFKC)
    - Lowercase conversion
    - Whitespace stripping and collapsing
    
        
    Example:
        >>> df['company_name_normalized'] = normalize_series(df['company_name'])
    """
    ser = ser.astype("string")
    # Vectorized Unicode normalization
    ser = ser.str.normalize("NFKC")
    # Vectorized strip and lowercase
    ser = ser.str.strip().str.lower()
    # Vectorized whitespace collapsing
    ser = ser.str.replace(r"\s+", " ", regex=True)
    return ser


def build_key_series(df: pd.DataFrame, cols: list[str], sep: str = "\x1f") -> pd.Series:
    """
    Create composite keys by joining multiple columns together.
    
    This is useful for matching on multiple fields at once. For example,
    matching on (company_name + city + state) gives you more confidence
    than just company_name alone.
    
    Uses a rare Unicode separator (\x1f - "Unit Separator") to avoid
    conflicts with actual data that might contain common separators like
    "-", "_", or "|".
    
    Args:
        df: DataFrame containing the columns to join
        cols: List of column names to combine into a key
        sep: Separator character (default: \x1f, a control character)
        
    """
    if not cols:
        return pd.Series([""], index=df.index, dtype="string")
    
    # Convert all columns to string, fill NaN with empty string, then join
    key = df[cols].astype("string").fillna("").agg(sep.join, axis=1)
    return key

