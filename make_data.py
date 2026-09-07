#!/usr/bin/env python3
"""Regenerate every bundled dataset. Equivalent to `python -m skmastery.datasets`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from skmastery.datasets import DATA_DIR, build_all

if __name__ == "__main__":
    print(f"writing to {DATA_DIR}")
    for name, shape in build_all(refresh=True).items():
        print(f"  {name:<20} {shape[0]:>7,} rows × {shape[1]:>2} cols")
