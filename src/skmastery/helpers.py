"""Small plotting / reporting helpers so the notebooks stay about scikit-learn.

Nothing here is clever. It exists so that a notebook cell reads
`plot_pr_roc(y, scores)` instead of fifteen lines of matplotlib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 20261007

# A small, colour-blind-safe categorical palette used across the curriculum.
PALETTE = [
    "#3d5a80",  # deep blue
    "#ee6c4d",  # coral
    "#98c1d9",  # light blue
    "#84a98c",  # sage
    "#c9ada7",  # dusty rose
    "#6d6875",  # slate
]


def set_plot_style() -> None:
    """Consistent, quiet matplotlib defaults."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.figsize": (7.5, 4.2),
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.9,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )
    plt.close("all")


def describe_frame(df: pd.DataFrame, max_unique: int = 8) -> pd.DataFrame:
    """A compact column profile: dtype, missingness, cardinality, sample values.

    The first thing to run on any new table, and far more useful than
    ``df.describe()`` when the frame is mixed-type.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        if nunique <= max_unique:
            sample = ", ".join(map(str, sorted(s.dropna().unique())[:max_unique]))
        else:
            sample = ", ".join(map(str, s.dropna().unique()[:3])) + ", …"
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_%": round(100 * s.isna().mean(), 2),
                "n_unique": nunique,
                "sample": sample[:60],
            }
        )
    return pd.DataFrame(rows).set_index("column")


def summarize_cv(cv_results: dict, sort_by: str | None = None, top: int = 10) -> pd.DataFrame:
    """Turn ``search.cv_results_`` into a readable leaderboard."""
    df = pd.DataFrame(cv_results)
    score_cols = [c for c in df.columns if c.startswith("mean_test_")]
    keep = ["params"] + score_cols + [c.replace("mean_", "std_") for c in score_cols]
    keep += [c for c in ["mean_fit_time", "rank_test_score"] if c in df.columns]
    out = df[[c for c in keep if c in df.columns]].copy()
    sort_by = sort_by or score_cols[0]
    return out.sort_values(sort_by, ascending=False).head(top).reset_index(drop=True)


def plot_confusion(y_true, y_pred, labels=None, ax=None, normalize=None, title="Confusion matrix"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, normalize=normalize, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    ax.grid(False)
    return ax


def plot_pr_roc(y_true, y_score, label="model", axes=None):
    """ROC and precision-recall side by side, with the no-skill baselines drawn.

    On imbalanced problems the PR curve is the honest one; ROC flatters.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        PrecisionRecallDisplay,
        RocCurveDisplay,
        average_precision_score,
        roc_auc_score,
    )

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    RocCurveDisplay.from_predictions(y_true, y_score, name=label, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "--", color="0.7", lw=1)
    axes[0].set_title(f"ROC — AUC {roc_auc_score(y_true, y_score):.3f}")

    PrecisionRecallDisplay.from_predictions(y_true, y_score, name=label, ax=axes[1])
    base = np.mean(y_true)
    axes[1].axhline(base, ls="--", color="0.7", lw=1)
    axes[1].set_title(f"PR — AP {average_precision_score(y_true, y_score):.3f} (base {base:.3f})")
    axes[1].set_ylim(0, 1.02)
    return axes


def plot_calibration(y_true, prob_dict: dict, n_bins: int = 10, ax=None, strategy="quantile"):
    """Reliability diagram for one or more models."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="0.7", lw=1, label="perfectly calibrated")
    for name, p in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy=strategy)
        brier = brier_score_loss(y_true, p)
        ax.plot(mean_pred, frac_pos, "o-", ms=4, label=f"{name} (Brier {brier:.4f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration")
    ax.legend(loc="upper left")
    return ax


def leakage_report(X: pd.DataFrame, y, top: int = 12) -> pd.DataFrame:
    """Rank features by univariate association with the target.

    A blunt instrument, and that is the point: any single feature that is
    almost as predictive as your whole model deserves a conversation with
    whoever owns the source table *before* you ship anything.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    binary = len(np.unique(y)) == 2
    rows = []
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s):
            if s.nunique() > 50:
                continue
            codes = s.astype("category").cat.codes
            # target-mean encode, then score
            enc = pd.Series(y).groupby(codes.to_numpy()).transform("mean") if binary else None
            score = roc_auc_score(y, enc) if binary and enc is not None else np.nan
            kind = "categorical"
        else:
            v = s.fillna(s.median())
            score = roc_auc_score(y, v) if binary else np.nan
            score = max(score, 1 - score) if score == score else score
            kind = "numeric"
        rows.append({"feature": col, "kind": kind, "univariate_auc": score})
    out = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
    return out.head(top).reset_index(drop=True)


def section(title: str, char: str = "─", width: int = 68) -> None:
    """Readable console separator for long notebook outputs."""
    print(f"\n{title}\n{char * min(width, max(len(title), 20))}")
