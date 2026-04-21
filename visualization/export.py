"""
export.py
=========
Handles all file saving and export for the CoWriteX Visualization Module.

Supports:
    - PNG / PDF  → via matplotlib savefig
    - LaTeX      → native pgfplots code generated from raw data (zero external files)

Outputs are saved to:  visualization/outputs/
"""

from __future__ import annotations

import re as _re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# outputs/ lives inside the visualization package folder
OUTPUTS_DIR = Path(__file__).parent / "outputs"

SUPPORTED_FORMATS = {"png", "pdf", "latex"}

# Formats that matplotlib savefig actually understands
_RASTER_FORMATS = {"png", "pdf"}


# ---------------------------------------------------------------------------
# Public Export Functions
# ---------------------------------------------------------------------------

def export_figure(fig: plt.Figure, config: dict) -> str:
    """
    Save a matplotlib Figure to visualization/outputs/.

    Parameters
    ----------
    fig    : matplotlib Figure object
    config : dict — expects keys: "filename", "format" (png | pdf)

    Returns
    -------
    str : Absolute path to the saved file.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fmt = config.get("format", "png").lower()
    stem = Path(config.get("filename", "chart")).stem
    out_path = OUTPUTS_DIR / f"{stem}.{fmt}"

    fig.savefig(out_path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_path.resolve())


def export_latex_table(df: pd.DataFrame, config: dict) -> str:
    """
    Export a DataFrame to a LaTeX .tex file.

    Parameters
    ----------
    df     : pandas DataFrame
    config : dict — expects keys: "filename", "title"

    Returns
    -------
    str : Absolute path to the saved .tex file.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    stem = Path(config.get("filename", "table")).stem
    out_path = OUTPUTS_DIR / f"{stem}.tex"

    latex_str = df.to_latex(
        index=False,
        caption=config.get("title", "Table"),
        label=f"tab:{stem}",
        float_format="%.4f",
    )

    out_path.write_text(latex_str, encoding="utf-8")
    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# LaTeX chart export — fully self-contained pgfplots
# ---------------------------------------------------------------------------

def _escape_tex(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    for ch, repl in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        s = s.replace(ch, repl)
    return s


def _hex_to_pgf_color(color: str, name: str = "chartcolor") -> tuple[str, str]:
    """
    Convert a hex color string to a pgfplots \\definecolor declaration.

    Returns (definition_line, color_name).
    If the color is not hex, returns ("", color) so pgfplots uses it as-is.
    """
    m = _re.match(r"#([0-9a-fA-F]{6})", str(color))
    if m:
        r, g, b = (int(m.group(1)[i:i+2], 16) / 255 for i in (0, 2, 4))
        defn = f"\\definecolor{{{name}}}{{rgb}}{{{r:.3f},{g:.3f},{b:.3f}}}"
        return defn, name
    return "", color or "blue!60"


def _pgfplots_bar(data: dict, config: dict, color: str) -> str:
    import math
    xs = [str(v) for v in data["x"]]
    ys = data["y"]
    # filter out NaN / None pairs
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None and not (isinstance(y, float) and math.isnan(y))]
    valid_xs = [p[0] for p in pairs]
    xlabel = _escape_tex(config.get("x_label", ""))
    ylabel = _escape_tex(config.get("y_label", ""))
    coords = "\n".join(f"    ({x},{y})" for x, y in pairs)
    # scale width dynamically — give LOTS of space per bar (1.2cm each) to prevent label overlap
    n_bars = len(valid_xs)
    calc_width = max(15.0, n_bars * 1.2)

    return (
        "\\begin{axis}[\n"
        "    ybar,\n"
        f"    xlabel={{{xlabel}}},\n"
        f"    ylabel={{{ylabel}}},\n"
        "    symbolic x coords={" + ",".join(valid_xs) + "},\n"
        "    xtick=data,\n"
        "    x tick label style={rotate=90, anchor=east, font=\\tiny},\n"
        "    enlarge x limits=0.03,\n"
        "    bar width=0.5cm,\n"
        "    nodes near coords,\n"
        "    nodes near coords style={font=\\tiny, rotate=90, anchor=west},\n"
        f"    width={calc_width}cm, height=10cm,\n"
        "]\n"
        f"\\addplot[fill={color}] coordinates {{\n"
        + coords + "\n};\n"
        "\\end{axis}"
    )


def _pgfplots_line(data: dict, config: dict, color: str) -> str:
    import math
    pairs = [(x, y) for x, y in zip(data["x"], data["y"])
             if y is not None and not (isinstance(y, float) and math.isnan(y))]
    coords = "\n".join(f"    ({x},{y})" for x, y in pairs)
    xlabel = _escape_tex(config.get("x_label", ""))
    ylabel = _escape_tex(config.get("y_label", ""))
    return (
        "\\begin{axis}[\n"
        f"    xlabel={{{xlabel}}},\n"
        f"    ylabel={{{ylabel}}},\n"
        "    width=\\textwidth, height=8cm,\n"
        "    grid=major,\n"
        "]\n"
        f"\\addplot[color={color}, mark=*, thick] coordinates {{\n"
        + coords + "\n};\n"
        "\\end{axis}"
    )


def _pgfplots_scatter(data: dict, config: dict, color: str) -> str:
    import math
    pairs = [(x, y) for x, y in zip(data["x"], data["y"])
             if y is not None and not (isinstance(y, float) and math.isnan(y))]
    coords = "\n".join(f"    ({x},{y})" for x, y in pairs)
    xlabel = _escape_tex(config.get("x_label", ""))
    ylabel = _escape_tex(config.get("y_label", ""))
    return (
        "\\begin{axis}[\n"
        f"    xlabel={{{xlabel}}},\n"
        f"    ylabel={{{ylabel}}},\n"
        "    width=\\textwidth, height=8cm,\n"
        "    grid=major,\n"
        "]\n"
        f"\\addplot[only marks, color={color}, mark=*] coordinates {{\n"
        + coords + "\n};\n"
        "\\end{axis}"
    )


def _pgfplots_pie(data: dict) -> str:
    labels = data["labels"]
    values = data["values"]
    slices = ", ".join(
        f"{v}/{_escape_tex(str(l))}" for v, l in zip(values, labels)
    )
    return f"\\pie[text=legend, radius=3]{{{slices}}}"


def _pgfplots_heatmap(data: dict, config: dict) -> str:
    import numpy as _np
    matrix   = _np.array(data["matrix"])
    x_labels = data.get("x_labels", [str(i) for i in range(matrix.shape[1])])
    y_labels = data.get("y_labels", [str(i) for i in range(matrix.shape[0])])
    rows, cols = matrix.shape
    coords = "\n".join(
        f"    ({j},{i}) [{matrix[i,j]:.4f}]"
        for i in range(rows) for j in range(cols)
    )
    xlabel = _escape_tex(config.get("x_label", ""))
    ylabel = _escape_tex(config.get("y_label", ""))
    return (
        "\\begin{axis}[\n"
        f"    xlabel={{{xlabel}}},\n"
        f"    ylabel={{{ylabel}}},\n"
        "    width=\\textwidth, height=8cm,\n"
        f"    xtick={{0,...,{cols-1}}},\n"
        f"    ytick={{0,...,{rows-1}}},\n"
        "    xticklabels={" + ",".join(_escape_tex(str(l)) for l in x_labels) + "},\n"
        "    yticklabels={" + ",".join(_escape_tex(str(l)) for l in y_labels) + "},\n"
        "    colormap/hot,\n"
        "    colorbar,\n"
        "]\n"
        "\\addplot[matrix plot*, mesh/cols=" + str(cols) + "] coordinates {\n"
        + coords + "\n};\n"
        "\\end{axis}"
    )


def _pgfplots_boxplot(data: dict, config: dict, color: str) -> str:
    import numpy as _np
    raw    = data["y"]
    groups = raw if isinstance(raw[0], (list, _np.ndarray)) else [raw]
    labels = data.get("labels", [f"Group {i+1}" for i in range(len(groups))])
    entries = []
    for lbl, grp in zip(labels, groups):
        arr   = _np.array(grp, dtype=float)
        q1    = float(_np.percentile(arr, 25))
        med   = float(_np.median(arr))
        q3    = float(_np.percentile(arr, 75))
        iqr   = q3 - q1
        wlo   = float(arr[arr >= q1 - 1.5 * iqr].min())
        whi   = float(arr[arr <= q3 + 1.5 * iqr].max())
        entries.append(
            f"\\addplot+[boxplot prepared={{lower whisker={wlo:.4f},"
            f"lower quartile={q1:.4f},median={med:.4f},"
            f"upper quartile={q3:.4f},upper whisker={whi:.4f}}},"
            f"fill={color},fill opacity=0.6] coordinates {{}};\n"
            f"\\addlegendentry{{{_escape_tex(str(lbl))}}}"
        )
    xlabel = _escape_tex(config.get("x_label", ""))
    ylabel = _escape_tex(config.get("y_label", ""))
    return (
        "\\begin{axis}[\n"
        "    boxplot/draw direction=y,\n"
        f"    xlabel={{{xlabel}}},\n"
        f"    ylabel={{{ylabel}}},\n"
        "    width=\\textwidth, height=8cm,\n"
        "    xtick={" + ",".join(str(i+1) for i in range(len(groups))) + "},\n"
        "    xticklabels={" + ",".join(_escape_tex(str(l)) for l in labels) + "},\n"
        "]\n"
        + "\n".join(entries) + "\n"
        "\\end{axis}"
    )


_PGFPLOT_BUILDERS = {
    "bar":     _pgfplots_bar,
    "line":    _pgfplots_line,
    "scatter": _pgfplots_scatter,
    "boxplot": _pgfplots_boxplot,
    "violin":  _pgfplots_boxplot,   # same stats representation
}


def export_latex_chart(fig: plt.Figure, config: dict, data: dict | None = None) -> str:
    """
    Export a chart as a **fully self-contained** LaTeX .tex file.

    Uses native pgfplots commands built directly from the raw *data* dict.
    No PNG, no SVG, no external files — compiles standalone with::

        pdflatex <filename>.tex

    Supported: bar, line, scatter, pie, heatmap, boxplot, violin.

    Parameters
    ----------
    fig    : matplotlib Figure (closed and discarded — not used for output)
    config : dict — same schema as generate_chart() config
    data   : dict — the same raw data dict passed to generate_chart()
    """
    plt.close(fig)  # discard — we use raw data, not the rendered figure

    if data is None:
        raise ValueError(
            "export_latex_chart requires the raw 'data' dict to generate "
            "self-contained pgfplots code. Pass it from generate_chart()."
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    stem       = Path(config.get("filename", "chart")).stem
    title      = config.get("title", stem.replace("_", " ").title())
    chart_type = config.get("type", "bar")

    color_def, pgf_color = _hex_to_pgf_color(config.get("color", ""), "chartcolor")
    safe_title = _escape_tex(title)

    # ── Build the pgfplots body ───────────────────────────────────────────────
    if chart_type == "pie":
        body    = _pgfplots_pie(data)
        pie_pkg = "\\usepackage{pgf-pie}\n"
    elif chart_type == "heatmap":
        body    = _pgfplots_heatmap(data, config)
        pie_pkg = ""
    elif chart_type in _PGFPLOT_BUILDERS:
        body    = _PGFPLOT_BUILDERS[chart_type](data, config, pgf_color)
        pie_pkg = ""
    else:
        raise ValueError(f"LaTeX export not supported for chart type: '{chart_type}'")

    # ── Assemble full standalone .tex document ────────────────────────────────
    latex_doc = (
        "\\documentclass[tikz,margin=2mm]{standalone}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{pgfplots}\n"
        + pie_pkg
        + "\\usepackage{pgfplotstable}\n"
        "\\pgfplotsset{compat=1.18}\n"
        + (color_def + "\n" if color_def else "")
        + "\n"
        "\\begin{document}\n"
        "\n"
        "\\begin{tikzpicture}\n"
        "    " + body.replace("\n", "\n    ") + "\n"
        "\\end{tikzpicture}\n"
        "\n"
        "\\end{document}\n"
    )

    tex_path = OUTPUTS_DIR / f"{stem}.tex"
    tex_path.write_text(latex_doc, encoding="utf-8")
    return str(tex_path.resolve())


# ---------------------------------------------------------------------------
# Table export helpers
# ---------------------------------------------------------------------------

def export_matplotlib_table(df: pd.DataFrame, config: dict) -> str:
    """
    Render a DataFrame as a styled matplotlib table and save as image/pdf.

    Parameters
    ----------
    df     : pandas DataFrame
    config : dict — same schema as generate_chart config

    Returns
    -------
    str : Absolute path to the saved file.
    """
    n_rows, n_cols = df.shape
    fig_height = max(2.0, 0.5 * (n_rows + 2))
    fig_width = max(6.0, 1.6 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor("#F2F4F6" if i % 2 == 0 else "white")

    ax.set_title(config.get("title", ""), fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()

    return export_figure(fig, config)
