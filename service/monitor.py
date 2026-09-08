"""Reads the prediction log and emits the alerts the memo promised."""

import json
import sys
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parent / "artifact"


def load_log(path="predictions.log"):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


def psi(expected, actual, n_bins=10, eps=1e-6):
    """Population Stability Index between a reference and a current sample."""
    import numpy as np

    edges = pd.qcut(expected, n_bins, retbins=True, duplicates="drop")[1]
    edges[0], edges[-1] = -float("inf"), float("inf")
    e = pd.cut(expected, edges).value_counts(normalize=True).sort_index().to_numpy() + eps
    a = pd.cut(actual, edges).value_counts(normalize=True).sort_index().to_numpy() + eps
    return float(np.sum((a - e) * np.log(a / e)))


def run(log_path="predictions.log"):
    manifest = json.loads((ART / "manifest.json").read_text())
    df = load_log(log_path)
    if df.empty:
        print("no predictions logged yet")
        return 0
    feats = pd.DataFrame(list(df["features"]))
    ref = pd.read_json(ART / "reference_inputs.json", orient="table")
    alerts = []

    refer_rate = float((df["decision"] == "refer").mean())
    baseline = manifest["training"]["base_rate"]
    if abs(refer_rate - baseline) > 0.06:
        alerts.append(f"CRITICAL refer rate {refer_rate:.3f} vs baseline {baseline:.3f}")
    elif abs(refer_rate - baseline) > 0.03:
        alerts.append(f"WARN     refer rate {refer_rate:.3f} vs baseline {baseline:.3f}")

    for col, spec in manifest["schema"].items():
        if "min" not in spec or col not in feats:
            continue
        value = psi(ref[col].dropna(), pd.to_numeric(feats[col], errors="coerce").dropna())
        if value > 0.25:
            alerts.append(f"CRITICAL PSI {value:.3f} on {col}")
        elif value > 0.10:
            alerts.append(f"WARN     PSI {value:.3f} on {col}")

    p95 = float(df["latency_ms"].quantile(0.95))
    if p95 > 50:
        alerts.append(f"CRITICAL p95 latency {p95:.1f} ms exceeds the 50 ms budget")

    print(f"{len(df):,} predictions | refer rate {refer_rate:.3f} | p95 latency {p95:.2f} ms")
    for a in alerts:
        print(" ", a)
    if not alerts:
        print("  all monitors within limits")
    return 1 if any(a.startswith("CRITICAL") for a in alerts) else 0


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
