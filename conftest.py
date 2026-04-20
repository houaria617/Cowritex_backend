"""
conftest.py  —  REPO ROOT
─────────────────────────
Adds the repo root to sys.path so every top-level package
(orchestrator, database, writing_agent, literature_agent, visualization)
is importable from any test file.

Works with pytest 6, 7, and 8 — no pythonpath= ini option needed.
Must live at the repo root alongside pyproject.toml.
"""
import sys
import os

# Insert repo root at the front of sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
