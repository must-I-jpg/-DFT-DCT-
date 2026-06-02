#!/usr/bin/env python3
"""Build a double-clickable desktop package with PyInstaller."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED_MODULES = [
    "IPython",
    "PyQt5",
    "black",
    "bokeh",
    "cryptography",
    "dask",
    "distributed",
    "docutils",
    "h5py",
    "matplotlib",
    "numba",
    "pandas",
    "pyarrow",
    "pytest",
    "scipy",
    "skimage",
    "sphinx",
    "tables",
    "tkinter",
]


def main() -> None:
    system = platform.system()
    bundle_mode = "--onedir" if system == "Darwin" else "--onefile"
    separator = os.pathsep
    environment = os.environ.copy()
    environment["PYINSTALLER_CONFIG_DIR"] = str(ROOT / ".pyinstaller-cache")
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        bundle_mode,
        "--name",
        "图像频域水印",
        "--add-data",
        f"{ROOT / 'templates'}{separator}templates",
        "--add-data",
        f"{ROOT / 'static'}{separator}static",
    ]
    for module in EXCLUDED_MODULES:
        command.extend(["--exclude-module", module])
    command.append(str(ROOT / "app.py"))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)

    if system == "Darwin":
        result = ROOT / "dist" / "图像频域水印.app"
    elif system == "Windows":
        result = ROOT / "dist" / "图像频域水印.exe"
    else:
        result = ROOT / "dist" / "图像频域水印"
    print(f"Build complete: {result}")


if __name__ == "__main__":
    main()
