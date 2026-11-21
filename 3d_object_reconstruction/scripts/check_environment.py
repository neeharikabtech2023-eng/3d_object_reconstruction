from __future__ import annotations

import importlib
import shutil
import sys

MODULES = ["cv2", "open3d", "streamlit", "numpy", "PIL", "plotly", "trimesh"]

print(f"Python: {sys.version}")
for module_name in MODULES:
    try:
        module = importlib.import_module(module_name)
        print(f"[OK] {module_name}: {getattr(module, '__version__', 'imported')}")
    except Exception as exc:
        print(f"[MISSING] {module_name}: {exc}")

colmap = shutil.which("colmap") or shutil.which("COLMAP")
print(f"COLMAP on PATH: {colmap or 'not found (set COLMAP_EXE or enter path in app sidebar)'}")
