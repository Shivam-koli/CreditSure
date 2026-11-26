"""Format parsers shared by both cleaning pipelines."""
import numpy as np
import pandas as pd

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%d.%m.%Y")
_MONEY_RE = r"(?i)^\s*(?:rs\.?|inr)?\s*(-?[\d,]*\.?\d+)\s*(lakh|lac|l|k)?\s*$"


def blank_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace in every text column and turn empty strings into NaN."""
    out = df.copy()
    for c in out.columns:
        s = out[c].astype(object)
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        out[c] = s.replace("", np.nan)
    return out


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse the four date layouts seen in the landing files (never ambiguous: separators differ)."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    todo = s.notna()
    for fmt in DATE_FORMATS:
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        ok = parsed.notna()
        out.loc[parsed.index[ok]] = parsed[ok].astype("datetime64[ns]")
        todo.loc[parsed.index[ok]] = False
    return out


def parse_money(s: pd.Series) -> tuple[pd.Series, int]:
    """'Rs. 12,50,000' / 'INR 45,000' / '12.5 L' / '4 Lakh' / '45000' -> float. Returns (values, n_strings_fixed)."""
    s = s.astype(object)
    txt = s.map(lambda v: v if isinstance(v, str) else (str(v) if pd.notna(v) else None))
    parts = txt.str.extract(_MONEY_RE)
    num = pd.to_numeric(parts[0].str.replace(",", "", regex=False), errors="coerce")
    mult = parts[1].str.lower().map({"l": 1e5, "lakh": 1e5, "lac": 1e5, "k": 1e3}).fillna(1.0)
    fixed = int((txt.notna() & txt.str.contains(r"(?i)[a-z]|,", regex=True, na=False)).sum())
    return num * mult, fixed
