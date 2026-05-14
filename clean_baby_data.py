"""
clean_baby_data.py
==================

Cleaning + standardization pipeline for the multi-sheet `baby.xlsx` dataset
(infant feeding study: behavioral binaries + valence/arousal time series).

Each function is independent and testable. The end-to-end pipeline is wired
together in `run_pipeline()` at the bottom. The accompanying notebook
`explore_baby_data.ipynb` walks through the same steps interactively.

Design decisions agreed with the analyst:
  - Unit of observation: one row per (patient, day)
  - Missingness: convert FIT_FAILED / NA / S/I / '-' / None to NaN, AND
    keep parallel boolean indicator columns (`*_fit_failed`, `*_was_missing`)
  - `Sí, poco`: ordinal 0 / 0.5 / 1
  - `Distribución videos` sheet: used ONLY for QA cross-check, not a feature
  - Column-drop threshold by missingness: configurable, default 0.60
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. SCHEMA                                                                    #
# --------------------------------------------------------------------------- #

# Position-based column map. The patient sheets have a two-row header where
# row 0 is a group label (Valencia 1, Arousal 2, ...) and row 1 is a
# timepoint sub-label (0,5 antes de prueba 1, Durante, ...). Rather than
# fight with the merged-cell structure, we read with header=None and assign
# clean names by position.
STANDARD_COLUMNS: dict[int, str] = {
    0:  "day",
    1:  "time_total",
    2:  "time_test1",
    3:  "time_test2",
    4:  "rejects",
    5:  "cries",
    6:  "touches",
    7:  "tries",
    8:  "consumes",
    9:  "valence_start",
    10: "arousal_start",
    11: "valence_test1_before",
    12: "valence_test1_during",
    13: "valence_test1_after",
    14: "valence_test2_before",
    15: "valence_test2_during",
    16: "valence_test2_after",
    17: "arousal_test1_before",
    18: "arousal_test1_during",
    19: "arousal_test1_after",
    20: "arousal_test2_before",
    21: "arousal_test2_during",
    22: "arousal_test2_after",
    23: "valence_end",
    24: "arousal_end",
    25: "comments",
}

# ID3 has 8 additional columns (cols 26-33) representing a 3rd test event.
# Cols 26 & 27 are blanks in the source.
EXTRA_COLUMNS_ID3: dict[int, str] = {
    28: "valence_test3_before",
    29: "valence_test3_during",
    30: "valence_test3_after",
    31: "arousal_test3_before",
    32: "arousal_test3_during",
    33: "arousal_test3_after",
}

# All numeric "time" columns (seconds, possibly stored as MM:SS.mmm strings)
TIME_COLS: tuple[str, ...] = ("time_total", "time_test1", "time_test2")

# All categorical behavioral columns (yes/no/yes-a-little)
BEHAVIOR_COLS: tuple[str, ...] = ("rejects", "cries", "touches", "tries", "consumes")

# All continuous valence/arousal columns
VA_COLS_BASE: tuple[str, ...] = (
    "valence_start", "arousal_start",
    "valence_test1_before", "valence_test1_during", "valence_test1_after",
    "valence_test2_before", "valence_test2_during", "valence_test2_after",
    "arousal_test1_before", "arousal_test1_during", "arousal_test1_after",
    "arousal_test2_before", "arousal_test2_during", "arousal_test2_after",
    "valence_end", "arousal_end",
)
VA_COLS_EXTRA: tuple[str, ...] = tuple(EXTRA_COLUMNS_ID3.values())
VA_COLS: tuple[str, ...] = VA_COLS_BASE + VA_COLS_EXTRA

# Tokens that should be interpreted as "this value is missing / failed".
MISSING_TOKENS: set[str] = {"", "-", "na", "n/a", "s/i", "nan", "none"}
FIT_FAILED_TOKEN: str = "fit_failed"


# --------------------------------------------------------------------------- #
# 2. LOADING + CONCATENATION                                                   #
# --------------------------------------------------------------------------- #

def _is_patient_sheet(name: str) -> bool:
    return bool(re.fullmatch(r"ID\d+", name))


def load_sheet(xlsx_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load a single patient sheet, normalize headers by position."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # Drop the two header rows (group label + timepoint sub-label)
    data = raw.iloc[2:].reset_index(drop=True)

    # Build a column-name list using the standard map; fall back to "extra_N"
    n_cols = data.shape[1]
    names: list[str] = []
    for i in range(n_cols):
        if i in STANDARD_COLUMNS:
            names.append(STANDARD_COLUMNS[i])
        elif i in EXTRA_COLUMNS_ID3:
            names.append(EXTRA_COLUMNS_ID3[i])
        else:
            names.append(f"extra_{i}")
    data.columns = names

    # Drop the blank padding cols (col 26, 27 in ID3 -> 'extra_26', 'extra_27')
    data = data.loc[:, [c for c in data.columns if not c.startswith("extra_")]]

    # Drop rows where the day cell is empty (trailing blanks in the sheet)
    data = data[data["day"].notna()].copy()

    # Tag with patient_id
    data.insert(0, "patient_id", sheet_name)

    return data


def load_all_patients(xlsx_path: str | Path) -> pd.DataFrame:
    """Load all ID* sheets and concatenate into a long-format DataFrame."""
    xl = pd.ExcelFile(xlsx_path)
    patient_sheets = [s for s in xl.sheet_names if _is_patient_sheet(s)]
    frames = [load_sheet(xlsx_path, s) for s in patient_sheets]
    return pd.concat(frames, ignore_index=True, sort=False)


def load_distribucion_videos(xlsx_path: str | Path) -> pd.DataFrame:
    """Load the `Distribución videos` metadata sheet for QA cross-check."""
    raw = pd.read_excel(xlsx_path, sheet_name="Distribución videos", header=1)
    raw.columns = ["patient_label", "days_n", "videos_n", "coder"]
    raw = raw.dropna(subset=["patient_label"]).copy()
    # The labels in this sheet are "D1", "D2", ... -> map to "ID1", "ID2", ...
    raw["patient_id"] = raw["patient_label"].str.replace("D", "ID", regex=False)
    return raw[["patient_id", "days_n", "videos_n", "coder"]]


# --------------------------------------------------------------------------- #
# 3. STANDARDIZATION HELPERS                                                   #
# --------------------------------------------------------------------------- #

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _normalize_token(x) -> str:
    """Lowercase, strip whitespace, strip accents. Returns '' for None/NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return _strip_accents(str(x).strip().lower())


# ---- 3a. Time parsing ---------------------------------------------------- #

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")

def parse_time_to_seconds(value) -> float:
    """
    Convert a time entry to seconds (float).
    Accepts:
      - plain numbers (already seconds)               -> float(value)
      - HH:MM:SS.mmm strings ("00:00:14.333")         -> 14.333
      - missing tokens (None, '-', 'NA', 'S/I', etc.) -> NaN
      - malformed strings ("1,34,033", "1.00.60")     -> NaN (logged)
    """
    # Numeric pass-through
    if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
        return float(value)

    tok = _normalize_token(value)
    if tok in MISSING_TOKENS or tok == FIT_FAILED_TOKEN:
        return np.nan

    # HH:MM:SS.mmm
    m = _TIME_RE.match(tok)
    if m:
        h, mnt, sec = m.groups()
        return int(h) * 3600 + int(mnt) * 60 + float(sec)

    # Fallback: try plain float
    try:
        return float(tok.replace(",", "."))
    except ValueError:
        return np.nan


# ---- 3b. Day parsing ----------------------------------------------------- #

def parse_day(value) -> tuple[float, str | None]:
    """
    Split a day cell into (numeric_day, flag).
    Examples:
      1     -> (1.0, None)
      '1E'  -> (1.0, 'E')
      '12 ' -> (12.0, None)
      None  -> (NaN, None)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return (np.nan, None)
    s = str(value).strip()
    m = re.match(r"^(\d+)([A-Za-z]*)$", s)
    if not m:
        return (np.nan, s if s else None)
    num, suffix = m.groups()
    return (float(num), suffix if suffix else None)


# ---- 3c. Behavioral binarization (ordinal 0/0.5/1) ----------------------- #

def parse_yesno(value) -> float:
    """
    Map a yes/no/'yes a little' cell to ordinal {0, 0.5, 1}.
    Missing tokens -> NaN.
    """
    tok = _normalize_token(value)
    if tok in MISSING_TOKENS:
        return np.nan
    if tok == "no":
        return 0.0
    if "poco" in tok:        # 'si, poco' / 'sí, poco' / 'si poco'
        return 0.5
    if tok in ("si", "sí"):  # accent already stripped, kept for safety
        return 1.0
    return np.nan


# ---- 3d. Valence / Arousal --------------------------------------------- #

def parse_va(value) -> tuple[float, bool, bool]:
    """
    Parse a valence/arousal cell.
    Returns (numeric_value, fit_failed_flag, other_missing_flag).
    """
    if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
        return (float(value), False, False)

    tok = _normalize_token(value)
    if tok == FIT_FAILED_TOKEN:
        return (np.nan, True, False)
    if tok in MISSING_TOKENS:
        return (np.nan, False, True)

    try:
        return (float(tok.replace(",", ".")), False, False)
    except ValueError:
        return (np.nan, False, True)


# --------------------------------------------------------------------------- #
# 4. STANDARDIZE DATAFRAME                                                     #
# --------------------------------------------------------------------------- #

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all per-column cleaning. Returns a new DataFrame."""
    out = df.copy()

    # --- day ---
    parsed = out["day"].map(parse_day)
    out["day"] = parsed.map(lambda t: t[0])
    out["day_flag"] = parsed.map(lambda t: t[1])

    # --- time columns ---
    for col in TIME_COLS:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].map(
                lambda v: _normalize_token(v) in MISSING_TOKENS
            )
            out[col] = out[col].map(parse_time_to_seconds)

    # --- behavioral binaries ---
    for col in BEHAVIOR_COLS:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].map(
                lambda v: _normalize_token(v) in MISSING_TOKENS
            )
            out[col] = out[col].map(parse_yesno)

    # --- valence / arousal ---
    for col in VA_COLS:
        if col not in out.columns:
            continue
        parsed_va = out[col].map(parse_va)
        out[col] = parsed_va.map(lambda t: t[0])
        out[f"{col}_fit_failed"] = parsed_va.map(lambda t: t[1])
        out[f"{col}_was_missing"] = parsed_va.map(lambda t: t[2])

    # --- comments stay as-is (string or NaN) ---
    if "comments" in out.columns:
        out["comments"] = out["comments"].where(out["comments"].notna(), None)

    return out


# --------------------------------------------------------------------------- #
# 5. AUDITS                                                                    #
# --------------------------------------------------------------------------- #

def missingness_audit(df: pd.DataFrame, cols: Iterable[str] | None = None) -> pd.DataFrame:
    """% missing per column (only feature columns by default)."""
    if cols is None:
        cols = [c for c in df.columns
                if c not in ("patient_id", "day_flag", "comments")
                and not c.endswith("_was_missing")
                and not c.endswith("_fit_failed")]
    miss = df[list(cols)].isna().mean().sort_values(ascending=False)
    return miss.to_frame("pct_missing")


def per_patient_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Days per patient + completeness summary."""
    feature_cols = [c for c in df.columns
                    if c not in ("patient_id", "day_flag", "comments")
                    and not c.endswith("_was_missing")
                    and not c.endswith("_fit_failed")]
    g = df.groupby("patient_id", sort=False)
    summary = pd.DataFrame({
        "n_days": g.size(),
        "pct_missing_overall": g[feature_cols].apply(
            lambda d: d.isna().mean().mean()
        ),
    })
    return summary


def qa_against_distribucion(
    cleaned: pd.DataFrame, dist: pd.DataFrame
) -> pd.DataFrame:
    """Compare row counts per patient against the Distribución videos sheet."""
    counts = cleaned.groupby("patient_id").size().rename("rows_in_data")
    merged = dist.merge(counts, on="patient_id", how="outer")
    merged["row_vs_days_diff"] = merged["rows_in_data"] - merged["days_n"]
    return merged


# --------------------------------------------------------------------------- #
# 6. SEGREGATE                                                                 #
# --------------------------------------------------------------------------- #

def segregate(
    df: pd.DataFrame, drop_threshold: float = 0.60
) -> dict[str, pd.DataFrame]:
    """
    Split the cleaned frame into:
      - X_features:   numeric matrix ready for TDA
      - X_meta:       patient_id, day, day_flag, comments
      - X_indicators: *_fit_failed and *_was_missing booleans

    Columns above `drop_threshold` missingness are removed from X_features
    (kept in X_indicators if relevant).
    """
    meta_cols = ["patient_id", "day", "day_flag", "comments"]
    indicator_cols = [c for c in df.columns
                      if c.endswith("_fit_failed") or c.endswith("_was_missing")]
    feature_cols = [c for c in df.columns
                    if c not in meta_cols and c not in indicator_cols
                    and c != "day"]

    miss = df[feature_cols].isna().mean()
    keep = miss[miss <= drop_threshold].index.tolist()
    dropped = miss[miss > drop_threshold].index.tolist()

    X_meta = df[[c for c in meta_cols if c in df.columns]].copy()
    X_features = df[keep].copy()
    X_indicators = df[indicator_cols].copy()

    return {
        "X_meta": X_meta,
        "X_features": X_features,
        "X_indicators": X_indicators,
        "dropped_columns": dropped,
    }


# --------------------------------------------------------------------------- #
# 7. NORMALIZE                                                                 #
# --------------------------------------------------------------------------- #

def normalize(X: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    """
    Per-column normalization.
      - 'zscore'  : (x - mean) / std        (NaN-safe; std=0 -> kept as 0)
      - 'minmax'  : (x - min) / (max - min) (NaN-safe; constant col -> 0)
    """
    out = X.copy().astype(float)
    if method == "zscore":
        mu = out.mean(skipna=True)
        sigma = out.std(skipna=True).replace(0, np.nan)
        out = (out - mu) / sigma
        out = out.fillna(0) if False else out  # keep NaN; user decides imputation
    elif method == "minmax":
        mn = out.min(skipna=True)
        mx = out.max(skipna=True)
        rng = (mx - mn).replace(0, np.nan)
        out = (out - mn) / rng
    else:
        raise ValueError(f"Unknown method: {method}")
    return out


# --------------------------------------------------------------------------- #
# 8. END-TO-END PIPELINE                                                       #
# --------------------------------------------------------------------------- #

def run_pipeline(
    xlsx_path: str | Path,
    output_dir: str | Path,
    drop_threshold: float = 0.60,
) -> dict:
    """Run the full pipeline end-to-end and save outputs to `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    raw = load_all_patients(xlsx_path)
    dist = load_distribucion_videos(xlsx_path)

    # Clean
    cleaned = standardize(raw)

    # Audit
    audit_col = missingness_audit(cleaned)
    audit_pat = per_patient_audit(cleaned)
    qa = qa_against_distribucion(cleaned, dist)

    # Segregate
    splits = segregate(cleaned, drop_threshold=drop_threshold)

    # Normalize features (both flavors saved side-by-side)
    X_z = normalize(splits["X_features"], "zscore")
    X_mm = normalize(splits["X_features"], "minmax")

    # Save
    cleaned.to_csv(output_dir / "cleaned_long.csv", index=False)
    splits["X_features"].to_csv(output_dir / "X_features.csv", index=False)
    splits["X_indicators"].to_csv(output_dir / "X_indicators.csv", index=False)
    splits["X_meta"].to_csv(output_dir / "X_meta.csv", index=False)
    X_z.to_csv(output_dir / "X_features_zscore.csv", index=False)
    X_mm.to_csv(output_dir / "X_features_minmax.csv", index=False)
    audit_col.to_csv(output_dir / "audit_column_missingness.csv")
    audit_pat.to_csv(output_dir / "audit_per_patient.csv")
    qa.to_csv(output_dir / "qa_vs_distribucion.csv", index=False)

    try:
        cleaned.to_parquet(output_dir / "cleaned_long.parquet", index=False)
    except Exception:
        pass  # parquet engine optional

    return {
        "raw_shape": raw.shape,
        "cleaned_shape": cleaned.shape,
        "n_features_kept": splits["X_features"].shape[1],
        "n_features_dropped": len(splits["dropped_columns"]),
        "dropped_columns": splits["dropped_columns"],
        "audit_col": audit_col,
        "audit_pat": audit_pat,
        "qa": qa,
        "splits": splits,
        "X_zscore": X_z,
        "X_minmax": X_mm,
    }


if __name__ == "__main__":
    import sys
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "baby.xlsx"
    out = sys.argv[2] if len(sys.argv) > 2 else "./out"
    result = run_pipeline(xlsx, out)
    print(f"Raw shape:      {result['raw_shape']}")
    print(f"Cleaned shape:  {result['cleaned_shape']}")
    print(f"Features kept:  {result['n_features_kept']}")
    print(f"Features dropped (> threshold): {result['dropped_columns']}")
