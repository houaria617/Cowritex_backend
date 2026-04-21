#!/usr/bin/env python
"""
test_csv.py
===========
Drop a CSV into visualization/ and generate a chart — no CLI args needed.

HOW TO USE
----------
1. Copy your CSV into  visualization/
2. Edit the CONFIGURATION block below (filename, chart type, color …)
3. Run one of:
       python -m visualization.test_csv          (from project root)
       python test_csv.py                        (from inside visualization/)

Outputs land in:  visualization/outputs/
"""

import sys
from pathlib import Path

# Allow running as both  `python test_csv.py`  and  `python -m visualization.test_csv`
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
#  ✏️  EDIT THIS BLOCK — all settings are here, nothing else needs changing
# ─────────────────────────────────────────────────────────────────────────────

CSV_FILENAME  = "GPU_price.csv"   # CSV file located in visualization/
CHART_TYPE    = "bar"               # bar | line | scatter | pie | boxplot | violin | heatmap

# Color: use a named preset OR a hex string like "#3A7BD5"
# Named presets: blue | red | green | purple | orange | teal | navy | pink | yellow | grey | coral
COLOR         = "pink"

# Size: use a named preset OR a (width, height) tuple in inches
# Named presets: small | medium | large | square | 8 cm | 10 cm | journal column | presentation
SIZE          = "large"

OUTPUT_FORMAT = "latex"               # png | pdf | latex
OUTPUT_NAME   = "csv_chart"         # saved as  outputs/<OUTPUT_NAME>.<OUTPUT_FORMAT>
                                    # note: latex also saves <OUTPUT_NAME>.png alongside the .tex

# Optional: also export a table of the raw CSV data
EXPORT_TABLE  = False
TABLE_FORMAT  = "png"               # png | pdf | latex

# ─────────────────────────────────────────────────────────────────────────────


from visualization.csv_loader     import load_csv
from visualization.config_presets import resolve_ui_config
from visualization                import generate_chart, generate_table


def main() -> None:
    viz_dir  = Path(__file__).parent
    csv_path = viz_dir / CSV_FILENAME

    print("\n" + "=" * 60)
    print("  CoWriteX - CSV Visualization Test")
    print("=" * 60)

    # ── 1. Load CSV & auto-detect layout ─────────────────────────────────────
    print(f"\n[>>] Loading : {csv_path.resolve()}")
    data, hints = load_csv(csv_path, CHART_TYPE)
    print(f"    data keys  : {list(data.keys())}")
    print(f"    label hints: {hints}")

    # ── 2. Build config (UI values → internal matplotlib values) ─────────────
    stem = Path(CSV_FILENAME).stem.replace("_", " ").title()
    ui_config = {
        "type":     CHART_TYPE,
        "title":    f"{stem} — {CHART_TYPE.capitalize()} Chart",
        "x_label":  hints.get("x_label", ""),
        "y_label":  hints.get("y_label", ""),
        "color":    COLOR,
        "size":     SIZE,
        "format":   OUTPUT_FORMAT,
        "filename": OUTPUT_NAME,
    }

    config = resolve_ui_config(ui_config)

    print(f"\n[**] Config  :")
    print(f"    type     = {config['type']}")
    print(f"    color    = {config['color']}")
    print(f"    size     = {config['size']}  (inches)")
    print(f"    colormap = {config.get('colormap', 'n/a')}  (heatmap only)")
    print(f"    format   = {config['format']}")

    # ── 3. Generate chart ─────────────────────────────────────────────────────
    print(f"\n[>>] Generating {CHART_TYPE} chart...")
    chart_path = generate_chart(data=data, config=config)
    print(f"    [OK] Saved -> {chart_path}")

    # ── 4. (Optional) Export table ────────────────────────────────────────────
    if EXPORT_TABLE:
        import pandas as pd

        raw_df     = pd.read_csv(csv_path)
        table_data = raw_df.to_dict(orient="list")

        # Auto-scale table size based on data dimensions
        t_width  = max(8.0, 1.6 * len(raw_df.columns))
        t_height = max(3.0, 0.45 * len(raw_df))

        table_path = generate_table(
            data=table_data,
            config={
                "title":    f"{stem} — Data Table",
                "format":   TABLE_FORMAT,
                "filename": f"{OUTPUT_NAME}_table",
                "size":     (t_width, t_height),
            },
        )
        print(f"    [OK] Table  -> {table_path}")

    print("\n" + "=" * 60)
    print("  All outputs -> visualization/outputs/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
