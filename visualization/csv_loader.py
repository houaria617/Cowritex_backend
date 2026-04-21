"""
csv_loader.py
=============
CSV ingestion for the CoWriteX Visualization Module.

Reads a CSV file from disk, auto-detects its layout (wide / matrix / pair),
and converts it into the data dict expected by generate_chart() / generate_table().

Layout rules
------------
    wide   : multiple columns — first is a string label column, rest are numeric
             → bar, line, scatter, boxplot, violin
    matrix : square-ish grid of numbers (e.g. correlation / confusion matrix)
             → heatmap
    pair   : exactly one label column + one numeric column
             → pie, or any other single-series chart

Usage
-----
    from visualization.csv_loader import load_csv

    data, hints = load_csv("visualization/sample_data.csv", chart_type="bar")
    # hints = {"x_label": "Month", "y_label": "Papers"}
    # data  = {"x": [...], "y": [...]}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Format Detection
# ---------------------------------------------------------------------------

def _classify(df: pd.DataFrame) -> str:
    """
    Classify a DataFrame as 'wide', 'matrix', or 'pair'.

    Rules
    -----
    pair   : exactly 1 non-numeric column + exactly 1 numeric column
    matrix : ≤1 non-numeric column, ≥2 numeric columns, and rows ≈ number of
             numeric columns (square-ish)
    wide   : everything else (multiple mixed columns)
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    string_cols  = df.select_dtypes(exclude="number").columns.tolist()

    n_num = len(numeric_cols)
    n_str = len(string_cols)

    # pair: one label + one value
    if n_str == 1 and n_num == 1:
        return "pair"

    # matrix: all (or all-but-one) columns are numeric, and shape is square-ish
    if n_str <= 1 and n_num >= 2:
        n_rows = len(df)
        if abs(n_rows - n_num) <= 3 and n_rows >= 2:
            return "matrix"

    return "wide"


# ---------------------------------------------------------------------------
# Per-Layout Data Builders
# ---------------------------------------------------------------------------

def _build_wide(df: pd.DataFrame, chart_type: str) -> tuple[dict, dict]:
    """Build a data dict from a wide-format DataFrame."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    string_cols  = df.select_dtypes(exclude="number").columns.tolist()

    if not numeric_cols:
        raise ValueError(
            "CSV has no numeric columns — cannot generate a chart from this file."
        )

    # boxplot / violin: multiple numeric columns → one group per column
    if chart_type in ("boxplot", "violin"):
        if len(numeric_cols) >= 2:
            data  = {
                "y":      [df[c].dropna().tolist() for c in numeric_cols],
                "labels": numeric_cols,
            }
            hints = {"x_label": "Group", "y_label": numeric_cols[0]}
        else:
            data  = {"y": df[numeric_cols[0]].dropna().tolist()}
            hints = {"x_label": "", "y_label": numeric_cols[0]}
        return data, hints

    # bar / line / scatter: x = first string col (or row index), y = first numeric col
    if string_cols:
        x_col = string_cols[0]
        y_col = numeric_cols[0]
        data  = {"x": df[x_col].tolist(), "y": df[y_col].tolist()}
        hints = {"x_label": x_col, "y_label": y_col}
    elif len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        data  = {"x": df[x_col].tolist(), "y": df[y_col].tolist()}
        hints = {"x_label": x_col, "y_label": y_col}
    else:
        y_col = numeric_cols[0]
        data  = {"x": df.index.tolist(), "y": df[y_col].tolist()}
        hints = {"x_label": "Index", "y_label": y_col}

    return data, hints


def _build_matrix(df: pd.DataFrame) -> tuple[dict, dict]:
    """Build a data dict from a matrix-format DataFrame (for heatmaps)."""
    string_cols  = df.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    row_labels = (
        df[string_cols[0]].astype(str).tolist()
        if string_cols
        else [str(i) for i in range(len(df))]
    )

    matrix = df[numeric_cols].values.tolist()
    data   = {
        "matrix":   matrix,
        "x_labels": numeric_cols,
        "y_labels": row_labels,
    }
    hints = {"x_label": "", "y_label": ""}
    return data, hints


def _build_pair(df: pd.DataFrame, chart_type: str) -> tuple[dict, dict]:
    """Build a data dict from a two-column (label + value) DataFrame."""
    string_cols  = df.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    label_col = string_cols[0]
    value_col = numeric_cols[0]

    if chart_type == "pie":
        data = {
            "labels": df[label_col].tolist(),
            "values": df[value_col].tolist(),
        }
    else:
        data = {
            "x": df[label_col].tolist(),
            "y": df[value_col].tolist(),
        }

    hints = {"x_label": label_col, "y_label": value_col}
    return data, hints


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------

def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to coerce string columns that look like numbers into actual numerics.

    Handles common real-world formats:
        "$1,699"  →  1699.0
        "1,200"   →  1200.0
        "N/A"     →  NaN
        " 42 "    →  42.0

    Only object (string) columns are touched; already-numeric columns are left alone.
    If a column can't be coerced at all, it stays as a string.
    """
    for col in df.select_dtypes(include="object").columns:
        cleaned = (
            df[col]
            .str.strip()
            .str.replace(r"[\$,]", "", regex=True)   # strip $ and thousands commas
            .str.replace("N/A", "", regex=False)       # treat N/A as missing
            .str.replace("n/a", "", regex=False)
            .str.replace("NA",  "", regex=False)
        )
        converted = pd.to_numeric(cleaned, errors="coerce")
        # Only replace the column if at least half the non-null values parsed
        if converted.notna().sum() >= len(df) * 0.4:
            df[col] = converted
    return df



def load_csv(filepath: str | Path, chart_type: str) -> tuple[dict, dict]:
    """
    Load a CSV file and return (data_dict, label_hints) ready for generate_chart().

    Parameters
    ----------
    filepath   : Path to the CSV file (relative or absolute).
    chart_type : One of bar | line | scatter | pie | boxplot | violin | heatmap.
                 Drives how the columns are mapped to x / y / labels / matrix.

    Returns
    -------
    data   : dict — pass directly to generate_chart() or generate_table().
    hints  : dict — suggested {"x_label": ..., "y_label": ...} from column names.

    Raises
    ------
    FileNotFoundError : CSV path does not exist.
    ValueError        : CSV is empty or has no usable numeric data.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path.resolve()}\n"
            "Place your CSV inside the visualization/ directory and update CSV_FILENAME."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop spurious "Unnamed: N" columns — these are leftover row-number indexes
    # from CSVs saved with df.to_csv() without index=False
    unnamed_cols = [c for c in df.columns if str(c).strip().startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Clean "dirty" numeric strings: strip $, commas, N/A → real floats
    df = _clean_numeric(df)

    # heatmap always uses matrix layout
    if chart_type == "heatmap":
        return _build_matrix(df)

    layout = _classify(df)

    # A matrix-shaped CSV requested for a non-heatmap chart → treat as wide
    if layout == "matrix" and chart_type != "heatmap":
        layout = "pair" if chart_type == "pie" else "wide"

    if layout == "pair":
        return _build_pair(df, chart_type)

    return _build_wide(df, chart_type)
