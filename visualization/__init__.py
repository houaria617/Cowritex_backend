"""
visualization/__init__.py
=========================
Public API for the CoWriteX Visualization Module.

Any external module (orchestrator, API routes, tests) should import from here:

    from visualization import generate_chart, generate_table, export_figure

Module layout:
    charts.py        → generate_chart()  — all chart types (bar, line, scatter, pie, boxplot, violin, heatmap)
    tables.py        → generate_table()  — png, pdf, latex output
    export.py        → export_figure()   — saves matplotlib figures to visualization/outputs/
    visualization.py → orchestrator + __main__ test block
"""

from visualization.charts        import generate_chart
from visualization.tables        import generate_table
from visualization.export        import export_figure
from visualization.csv_loader    import load_csv
from visualization.config_presets import resolve_ui_config

__all__ = [
    "generate_chart",
    "generate_table",
    "export_figure",
    "load_csv",
    "resolve_ui_config",
]
