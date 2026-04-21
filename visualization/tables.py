"""
tables.py
=========
Table generation logic for the CoWriteX Visualization Module.

Converts structured data dicts → pandas DataFrames, then exports as:
    - LaTeX (.tex)  via export.export_latex_table()
    - PNG / PDF     via export.export_matplotlib_table()
"""

from __future__ import annotations

import pandas as pd

from visualization.export import export_latex_table, export_matplotlib_table, SUPPORTED_FORMATS
from visualization.charts import resolve_config


# ---------------------------------------------------------------------------
# Public Function
# ---------------------------------------------------------------------------

def generate_table(data: dict, config: dict) -> str:
    """
    Generate and save a table from structured data.

    Parameters
    ----------
    data : dict
        Column-oriented dict compatible with pd.DataFrame(data).
        Example:
            {
                "Method":    ["Baseline", "Proposed"],
                "Precision": [0.81, 0.93],
                "Recall":    [0.77, 0.92],
            }

    config : dict
        UI-driven configuration (all fields optional — defaults applied).
        Key field:
            "format": "latex" → saves a .tex file
            "format": "png" / "pdf" → saves a rendered table image

    Returns
    -------
    str : Absolute path to the saved file (inside visualization/outputs/).

    Raises
    ------
    ValueError : Empty data or invalid config.
    """
    resolved = resolve_config(config)

    if not data:
        raise ValueError("'data' is empty — cannot generate a table.")

    try:
        df = pd.DataFrame(data)
    except Exception as exc:
        raise ValueError(f"Could not build DataFrame from data: {exc}") from exc

    if df.empty:
        raise ValueError("DataFrame is empty after construction — check your data.")

    fmt = resolved["format"]

    if fmt == "latex":
        return export_latex_table(df, resolved)
    else:
        return export_matplotlib_table(df, resolved)
