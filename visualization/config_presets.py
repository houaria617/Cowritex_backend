"""
config_presets.py
=================
UI-friendly named presets for color and figure size.

Maps human-readable frontend values directly to matplotlib-compatible values.
This is the single translation layer between the UI and the charting engine.

Usage
-----
    from visualization.config_presets import resolve_ui_config

    # Config coming from the frontend / be hardcoded in test_csv.py:
    ui_config = {
        "type":     "bar",
        "color":    "Blue",       # ← named preset
        "size":     "medium",     # ← named preset
        "format":   "png",
        "filename": "my_chart",
    }

    config = resolve_ui_config(ui_config)
    # config["color"]    → "#3A7BD5"
    # config["size"]     → (9.0, 6.0)
    # config["colormap"] → "Blues"  (used by heatmap renderer)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color Presets  (named label → matplotlib hex)
# ---------------------------------------------------------------------------

COLOR_PRESETS: dict[str, str] = {
    "blue":     "#3A7BD5",
    "red":      "#E74C3C",
    "green":    "#27AE60",
    "purple":   "#8E44AD",
    "orange":   "#E67E22",
    "teal":     "#1ABC9C",
    "navy":     "#2C3E50",
    "pink":     "#E91E8C",
    "yellow":   "#F1C40F",
    "grey":     "#95A5A6",
    "gray":     "#95A5A6",
    "brown":    "#A04000",
    "coral":    "#E8735A",
}

# ---------------------------------------------------------------------------
# Size Presets  (named label → (width_inches, height_inches))
# ---------------------------------------------------------------------------

SIZE_PRESETS: dict[str, tuple[float, float]] = {
    # Generic
    "small":            (6.0,  4.0),
    "medium":           (9.0,  6.0),
    "large":            (12.0, 8.0),
    "square":           (7.0,  7.0),
    # UI size labels (map "X cm" to a sensible inch equivalent)
    "8 cm":             (6.0,  4.0),
    "10 cm":            (8.0,  5.5),
    "12 cm":            (10.0, 6.5),
    # Academic/publication presets
    "journal column":   (3.5,  2.5),
    "journal full":     (7.2,  4.5),
    "presentation":     (13.3, 7.5),
}

# ---------------------------------------------------------------------------
# Colormap Presets  (used for heatmap only — derived from chosen color)
# ---------------------------------------------------------------------------

COLORMAP_PRESETS: dict[str, str] = {
    "blue":     "Blues",
    "red":      "Reds",
    "green":    "Greens",
    "purple":   "Purples",
    "orange":   "Oranges",
    "teal":     "GnBu",
    "navy":     "Blues",
    "pink":     "RdPu",
    "yellow":   "YlOrRd",
    "grey":     "Greys",
    "gray":     "Greys",
    "brown":    "copper",
    "coral":    "YlOrRd",
}

DEFAULT_COLORMAP = "YlOrRd"

# ---------------------------------------------------------------------------
# Public Resolver
# ---------------------------------------------------------------------------

def resolve_ui_config(ui_config: dict) -> dict:
    """
    Translate UI-friendly config values into matplotlib-ready ones.

    Transformations applied
    -----------------------
    color   : "Blue"     → "#3A7BD5"       (hex — falls through unchanged if
                                             already a valid color string/hex)
    size    : "medium"   → (9.0, 6.0)      (inches — falls through unchanged
                                             if already a tuple or list)
    colormap: injected   → "Blues"         (derived from color; used by heatmap)

    Parameters
    ----------
    ui_config : dict with any subset of chart config keys.

    Returns
    -------
    dict : copy of ui_config with resolved values.
    """
    config = dict(ui_config)

    # ── Color ──────────────────────────────────────────────────────────────
    raw_color = config.get("color", "")
    if isinstance(raw_color, str):
        normalized = raw_color.lower().strip()
        if normalized in COLOR_PRESETS:
            config["color"]    = COLOR_PRESETS[normalized]
            config["colormap"] = COLORMAP_PRESETS.get(normalized, DEFAULT_COLORMAP)
        else:
            # Already a hex / named matplotlib color — keep as-is
            config.setdefault("colormap", DEFAULT_COLORMAP)
    else:
        config.setdefault("colormap", DEFAULT_COLORMAP)

    # ── Size ────────────────────────────────────────────────────────────────
    raw_size = config.get("size", "medium")
    if isinstance(raw_size, str):
        normalized = raw_size.lower().strip()
        config["size"] = SIZE_PRESETS.get(normalized, SIZE_PRESETS["medium"])
    # If already a tuple/list → leave unchanged

    return config
