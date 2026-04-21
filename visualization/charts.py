"""
charts.py
=========
Chart generation logic for the CoWriteX Visualization Module.

Handles all matplotlib rendering for:
    bar | line | scatter | pie | boxplot | violin | heatmap

All rendering uses matplotlib only — no seaborn.
"""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from visualization.export import export_figure, export_latex_chart, SUPPORTED_FORMATS

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

SUPPORTED_CHART_TYPES = {"bar", "line", "scatter", "pie", "boxplot", "violin", "heatmap"}

DEFAULTS: dict[str, Any] = {
    "type": "bar",
    "title": "Untitled Chart",
    "x_label": "",
    "y_label": "",
    "color": "steelblue",
    "size": (10, 6),
    "format": "png",
    "filename": "chart",
}


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------

def resolve_config(config: dict) -> dict:
    """
    Merge user-supplied config with defaults and validate all fields.

    Raises
    ------
    ValueError : Unsupported chart type, format, or invalid color/size.
    """
    resolved = {**DEFAULTS, **config}

    chart_type = resolved["type"].lower()
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ValueError(
            f"Unsupported chart type: '{chart_type}'. "
            f"Supported: {sorted(SUPPORTED_CHART_TYPES)}"
        )

    fmt = resolved["format"].lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported export format: '{fmt}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    if not mcolors.is_color_like(resolved["color"]):
        raise ValueError(
            f"Invalid color: '{resolved['color']}'. "
            "Use a named color (e.g. 'steelblue') or hex (e.g. '#3A7BD5')."
        )

    size = resolved["size"]
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        raise ValueError("'size' must be a (width, height) tuple in inches.")

    resolved["type"] = chart_type
    resolved["format"] = fmt
    return resolved


# ---------------------------------------------------------------------------
# Shared Style Helper
# ---------------------------------------------------------------------------

def _apply_style(ax: plt.Axes, config: dict) -> None:
    """Apply title, axis labels, grid, and clean spines to an Axes."""
    ax.set_title(config["title"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(config["x_label"], fontsize=11)
    ax.set_ylabel(config["y_label"], fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Data Key Validation
# ---------------------------------------------------------------------------

def _ensure_keys(data: dict, required: list[str], chart_type: str) -> None:
    """Raise a clear KeyError if required data keys are missing."""
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Missing required key(s) for '{chart_type}': {missing}. "
            f"Got: {list(data.keys())}"
        )


# ---------------------------------------------------------------------------
# Individual Chart Renderers
# ---------------------------------------------------------------------------

def _render_bar(ax: plt.Axes, data: dict, config: dict) -> None:
    """Render a bar chart with value labels on each bar."""
    _ensure_keys(data, ["x", "y"], "bar")
    x, y = data["x"], data["y"]
    positions = np.arange(len(x))

    bars = ax.bar(
        positions, y,
        color=config["color"],
        width=0.6,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(positions)

    # Smart rotation: vertical when many labels to avoid overlap
    n = len(x)
    if n > 12:
        rotation, ha = 90, "right"
    elif n > 6:
        rotation, ha = 45, "right"
    else:
        rotation, ha = 0, "center"
    ax.set_xticklabels(x, rotation=rotation, ha=ha, fontsize=8)

    # Only annotate bars when there are few enough to read clearly
    if n <= 20:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )


def _render_line(ax: plt.Axes, data: dict, config: dict) -> None:
    """Render a line chart with markers and a soft fill below the curve."""
    _ensure_keys(data, ["x", "y"], "line")

    n_pts = len(data["y"])

    # For large datasets: drop markers and use a thinner line so the chart stays readable
    if n_pts > 50:
        marker, markersize, linewidth = None, 0, 1.5
    else:
        marker, markersize, linewidth = "o", 5, 2

    ax.plot(
        data["x"], data["y"],
        color=config["color"],
        marker=marker,
        markersize=markersize,
        linewidth=linewidth,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    # Use numeric indices for fill so it aligns correctly whether x is
    # strings (auto-mapped to 0,1,2,…) or actual numbers.
    try:
        x_fill = [float(v) for v in data["x"]]
    except (TypeError, ValueError):
        x_fill = list(range(len(data["y"])))
    ax.fill_between(x_fill, data["y"], alpha=0.08, color=config["color"])


def _render_scatter(ax: plt.Axes, data: dict, config: dict) -> None:
    """Render a scatter plot."""
    _ensure_keys(data, ["x", "y"], "scatter")
    ax.scatter(
        data["x"], data["y"],
        color=config["color"],
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
        s=60,
    )


def _render_pie(ax: plt.Axes, data: dict, config: dict) -> None:
    """Render a pie chart using tab10 colormap."""
    _ensure_keys(data, ["labels", "values"], "pie")
    labels, values = data["labels"], data["values"]

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(labels))]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.set_title(config["title"], fontsize=14, fontweight="bold", pad=12)
    ax.axis("equal")


def _render_boxplot(ax: plt.Axes, data: dict, config: dict) -> None:
    """
    Render a box plot. Supports single or multi-group data.

    data["y"] can be:
        - A flat list → single box
        - A list of lists → one box per group (use data["labels"] for group names)
    """
    _ensure_keys(data, ["y"], "boxplot")
    raw = data["y"]

    if isinstance(raw[0], (list, np.ndarray)):
        plot_data = raw
        labels = data.get("labels", [f"Group {i+1}" for i in range(len(raw))])
    else:
        plot_data = [raw]
        labels = data.get("labels", [""])

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        labels=labels,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"markerfacecolor": config["color"], "markersize": 4},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(config["color"])
        patch.set_alpha(0.75)


def _render_violin(ax: plt.Axes, data: dict, config: dict) -> None:
    """
    Render a violin plot. Supports single or multi-group data.

    data["y"] follows the same format as boxplot.
    """
    _ensure_keys(data, ["y"], "violin")
    raw = data["y"]

    if isinstance(raw[0], (list, np.ndarray)):
        plot_data = raw
        labels = data.get("labels", [f"Group {i+1}" for i in range(len(raw))])
    else:
        plot_data = [raw]
        labels = data.get("labels", [""])

    parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)

    for pc in parts["bodies"]:
        pc.set_facecolor(config["color"])
        pc.set_alpha(0.6)
        pc.set_edgecolor("white")

    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if partname in parts:
            parts[partname].set_edgecolor(config["color"])
            parts[partname].set_linewidth(1.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)


def _render_heatmap(ax: plt.Axes, data: dict, config: dict) -> None:
    """
    Render a heatmap using pure matplotlib (no seaborn).

    Expected data keys:
        matrix   : 2D list/array of numeric values
        x_labels : optional column labels
        y_labels : optional row labels
    """
    _ensure_keys(data, ["matrix"], "heatmap")
    matrix = np.array(data["matrix"], dtype=float)

    # Respect the colormap derived from the user's chosen color (via config_presets).
    cmap = config.get("colormap", "YlOrRd")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    x_labels = data.get("x_labels", [str(i) for i in range(matrix.shape[1])])
    y_labels = data.get("y_labels", [str(i) for i in range(matrix.shape[0])])

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)

    thresh = (matrix.max() + matrix.min()) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val < thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color)

    ax.spines[:].set_visible(False)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Renderer Dispatcher
# ---------------------------------------------------------------------------

_RENDERERS = {
    "bar":      _render_bar,
    "line":     _render_line,
    "scatter":  _render_scatter,
    "pie":      _render_pie,
    "boxplot":  _render_boxplot,
    "violin":   _render_violin,
    "heatmap":  _render_heatmap,
}


# ---------------------------------------------------------------------------
# Public Function
# ---------------------------------------------------------------------------

def generate_chart(data: dict, config: dict) -> str:
    """
    Generate and save a chart from structured data.

    Parameters
    ----------
    data : dict
        Raw input data. Required keys vary by chart type:
        - bar / line / scatter : {"x": [...], "y": [...]}
        - pie                  : {"labels": [...], "values": [...]}
        - boxplot / violin     : {"y": [...]} or {"y": [[...], ...], "labels": [...]}
        - heatmap              : {"matrix": [[...], ...], "x_labels": [...], "y_labels": [...]}

    config : dict
        UI-driven configuration (all fields optional — defaults applied):
        {
            "type":     "bar" | "line" | "scatter" | "pie" | "boxplot" | "violin" | "heatmap",
            "title":    str,
            "x_label":  str,
            "y_label":  str,
            "color":    str,              # Named color or hex
            "size":     (width, height),  # In inches
            "format":   "png" | "pdf",
            "filename": str
        }

    Returns
    -------
    str : Absolute path to the saved file (inside visualization/outputs/).

    Raises
    ------
    ValueError : Unsupported type, format, or invalid color/size.
    KeyError   : Missing required data keys for the selected chart type.
    """
    resolved = resolve_config(config)
    chart_type = resolved["type"]

    fig, ax = plt.subplots(figsize=resolved["size"])

    _RENDERERS[chart_type](ax, data, resolved)

    if chart_type != "pie":
        _apply_style(ax, resolved)

    # For bar charts with many labels (rotated 90°), pad the bottom so labels don't clip
    if chart_type == "bar":
        n_items = len(data.get("x", []))
        if n_items > 12:
            fig.subplots_adjust(bottom=0.30)
        elif n_items > 6:
            fig.subplots_adjust(bottom=0.20)

    fig.tight_layout()

    # Route to the correct exporter
    if resolved["format"] == "latex":
        return export_latex_chart(fig, resolved, data)
    return export_figure(fig, resolved)
