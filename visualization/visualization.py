# -*- coding: utf-8 -*-
"""
visualization.py
================
Main entry point for the CoWriteX Visualization Module.

Sub-modules
-----------
    charts.py          -> generate_chart()      bar, line, scatter, pie, boxplot, violin, heatmap
    tables.py          -> generate_table()      png, pdf, latex
    export.py          -> export_figure()       saves matplotlib Figure -> outputs/
    csv_loader.py      -> load_csv()            auto-detects CSV layout, returns data dict
    config_presets.py  -> resolve_ui_config()   maps UI label strings to matplotlib values

All outputs are saved to:  visualization/outputs/

Usage (from other modules):
    from visualization import generate_chart, generate_table, export_figure
    from visualization import load_csv, resolve_ui_config

Usage (standalone suite):
    python -m visualization.visualization

CSV test (user-editable):
    python -m visualization.test_csv
"""

from visualization.charts          import generate_chart
from visualization.tables          import generate_table
from visualization.export          import export_figure
from visualization.csv_loader      import load_csv
from visualization.config_presets  import resolve_ui_config

__all__ = [
    "generate_chart",
    "generate_table",
    "export_figure",
    "load_csv",
    "resolve_ui_config",
]


# ---------------------------------------------------------------------------
# Test Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    from visualization.export import OUTPUTS_DIR

    print("=" * 60)
    print("  CoWriteX -- Visualization Module Test Suite")
    print("=" * 60)

    # -- 1. Bar Chart ---------------------------------------------------
    print("\n[1/8] Bar Chart...")
    path = generate_chart(
        data={
            "x": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "y": [42, 58, 37, 65, 80, 72],
        },
        config={
            "type": "bar",
            "title": "Monthly Research Output",
            "x_label": "Month",
            "y_label": "Papers Published",
            "color": "#3A7BD5",
            "size": (10, 6),
            "format": "png",
            "filename": "bar_chart",
        },
    )
    print(f"  [OK] Saved -> {path}")

    # -- 2. Line Chart --------------------------------------------------
    print("\n[2/8] Line Chart...")
    path = generate_chart(
        data={
            "x": list(range(1, 11)),
            "y": [3.1, 4.5, 3.8, 5.2, 6.7, 5.9, 7.1, 6.4, 8.0, 9.2],
        },
        config={
            "type": "line",
            "title": "Model Loss over Epochs",
            "x_label": "Epoch",
            "y_label": "Loss",
            "color": "#E67E22",
            "size": (10, 5),
            "format": "png",
            "filename": "line_chart",
        },
    )
    print(f"  [OK] Saved -> {path}")

    # -- 3. Scatter Plot ------------------------------------------------
    print("\n[3/8] Scatter Plot...")
    np.random.seed(42)
    path = generate_chart(
        data={
            "x": np.random.randn(60).tolist(),
            "y": np.random.randn(60).tolist(),
        },
        config={
            "type": "scatter",
            "title": "Citation Count vs. Impact Factor",
            "x_label": "Normalized Citations",
            "y_label": "Impact Factor",
            "color": "#E74C3C",
            "size": (8, 6),
            "format": "png",
            "filename": "scatter_plot",
        },
    )
    print(f"  [OK] Saved -> {path}")

    # -- 4. Heatmap -----------------------------------------------------
    print("\n[4/8] Heatmap (pure matplotlib, no seaborn)...")
    path = generate_chart(
        data={
            "matrix": [
                [0.95, 0.72, 0.31, 0.18],
                [0.55, 0.88, 0.64, 0.47],
                [0.22, 0.40, 0.77, 0.90],
                [0.10, 0.33, 0.58, 0.82],
            ],
            "x_labels": ["Metric A", "Metric B", "Metric C", "Metric D"],
            "y_labels": ["Model 1", "Model 2", "Model 3", "Model 4"],
        },
        config={
            "type": "heatmap",
            "title": "Model Performance Comparison",
            "x_label": "Evaluation Metric",
            "y_label": "Model",
            "color": "steelblue",
            "size": (9, 6),
            "format": "png",
            "filename": "heatmap",
        },
    )
    print(f"  [OK] Saved -> {path}")

    # -- 5. Boxplot -----------------------------------------------------
    print("\n[5/8] Boxplot (multi-group)...")
    path = generate_chart(
        data={
            "y": [
                np.random.normal(70, 10, 50).tolist(),
                np.random.normal(80, 8, 50).tolist(),
                np.random.normal(65, 15, 50).tolist(),
            ],
            "labels": ["Baseline", "Proposed", "Ablation"],
        },
        config={
            "type": "boxplot",
            "title": "Model Accuracy Distribution",
            "x_label": "Experiment",
            "y_label": "Accuracy (%)",
            "color": "#27AE60",
            "size": (9, 6),
            "format": "png",
            "filename": "boxplot",
        },
    )
    print(f"  [OK] Saved -> {path}")

    # -- 6a. Table -> LaTeX --------------------------------------------
    print("\n[6a/8] Table Export -> LaTeX (.tex)...")
    table_data = {
        "Method":    ["Baseline", "Proposed", "Ablation A", "Ablation B"],
        "Precision": [0.812, 0.934, 0.890, 0.876],
        "Recall":    [0.779, 0.921, 0.867, 0.853],
        "F1-Score":  [0.795, 0.927, 0.878, 0.864],
    }
    path = generate_table(
        data=table_data,
        config={
            "title": "Comparison of Methods",
            "format": "latex",
            "filename": "results_table",
        },
    )
    print(f"  [OK] Saved -> {path}")

    # -- 6b. Table -> PNG ----------------------------------------------
    print("\n[6b/8] Table Export -> PNG...")
    path = generate_table(
        data=table_data,
        config={
            "title": "Comparison of Methods",
            "format": "png",
            "filename": "results_table_img",
            "size": (10, 4),
        },
    )
    print(f"  [OK] Saved -> {path}")

    # -- 7. Violin Plot ------------------------------------------------
    print("\n[7/8] Violin Plot (multi-group, PDF export)...")
    path = generate_chart(
        data={
            "y": [
                np.random.normal(65, 12, 80).tolist(),
                np.random.normal(75, 9, 80).tolist(),
            ],
            "labels": ["GPT-3.5", "GPT-4"],
        },
        config={
            "type": "violin",
            "title": "LLM Score Distribution",
            "x_label": "Model",
            "y_label": "ROUGE Score",
            "color": "#8E44AD",
            "size": (8, 6),
            "format": "pdf",
            "filename": "violin_plot",
        },
    )
    print(f"  [OK] Saved -> {path}")

    # -- 8. CSV Demo ---------------------------------------------------
    print("\n[8/8] CSV Demo (sample_data.csv -> bar chart via resolve_ui_config)...")
    viz_dir  = Path(__file__).parent
    csv_path = viz_dir / "sample_data.csv"

    if csv_path.exists():
        csv_data, hints = load_csv(csv_path, "bar")
        config = resolve_ui_config({
            "type":     "bar",
            "title":    "Sample Data — Bar Chart (from CSV)",
            "x_label":  hints.get("x_label", ""),
            "y_label":  hints.get("y_label", ""),
            "color":    "teal",      # named preset -> #1ABC9C
            "size":     "medium",    # named preset -> (9, 6) inches
            "format":   "png",
            "filename": "csv_demo_bar",
        })
        path = generate_chart(data=csv_data, config=config)
        print(f"  [OK] Saved -> {path}")
    else:
        print("  [SKIP] sample_data.csv not found — run test_csv.py separately.")

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {OUTPUTS_DIR.resolve()}")
    print("=" * 60)
