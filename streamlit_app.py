"""Streamlit Cloud entrypoint for NATIP.

Run the dashboard script directly instead of importing it as a package module.
This avoids Streamlit Cloud watcher/import edge cases around ``app.dashboard``
while keeping the local project package layout unchanged.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

runpy.run_path(str(PROJECT_ROOT / "app" / "dashboard" / "streamlit_app.py"), run_name="__main__")
