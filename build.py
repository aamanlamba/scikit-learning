#!/usr/bin/env python3
"""Convert the jupytext `.py` sources in build/ into notebooks, then execute them.

Usage
-----
    python build.py                 # convert + execute everything
    python build.py --no-exec       # convert only (fast)
    python build.py 03 07           # only the modules whose names start 03 / 07
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "build"
OUT_NB = ROOT / "notebooks"
OUT_SOL = ROOT / "solutions"
OUT_GUIDE = ROOT / "guided"


def targets(prefixes: list[str]) -> list[Path]:
    files = sorted(SRC.glob("*.py"))
    if prefixes:
        files = [f for f in files if any(f.name.startswith(p) for p in prefixes)]
    return files


def convert(path: Path) -> Path:
    nb = jupytext.read(path, fmt="py:percent")
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    if "_solutions" in path.stem:
        dest_dir = OUT_SOL
    elif "_guided" in path.stem:
        dest_dir = OUT_GUIDE
    else:
        dest_dir = OUT_NB
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (path.stem + ".ipynb")
    nbformat.write(nb, dest)
    return dest


def execute(path: Path, timeout: int = 1200) -> tuple[bool, str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        return False, str(exc)[-2500:]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:1400]
    nbformat.write(nb, path)
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*", default=[])
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]

    failures = []
    for src in targets(args.prefixes):
        dest = convert(src)
        line = f"{dest.relative_to(ROOT)!s:<52}"
        if args.no_exec:
            print(line + "converted")
            continue
        t0 = time.time()
        ok, err = execute(dest)
        dt = time.time() - t0
        print(line + (f"OK   {dt:6.1f}s" if ok else f"FAIL {dt:6.1f}s"))
        if not ok:
            failures.append((dest.name, err))

    for name, err in failures:
        print(f"\n===== {name} =====\n{err}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
