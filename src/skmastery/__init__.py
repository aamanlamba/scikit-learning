"""Shared helpers for the scikit-learn mastery curriculum."""

from .datasets import (
    DATA_DIR,
    load_card_fraud,
    load_credit_risk,
    load_insurance_claims,
    load_support_tickets,
    load_telco_churn,
)
from .helpers import (
    RANDOM_STATE,
    describe_frame,
    leakage_report,
    plot_calibration,
    plot_confusion,
    plot_pr_roc,
    set_plot_style,
    summarize_cv,
)

__all__ = [
    "DATA_DIR",
    "RANDOM_STATE",
    "load_credit_risk",
    "load_card_fraud",
    "load_insurance_claims",
    "load_telco_churn",
    "load_support_tickets",
    "set_plot_style",
    "describe_frame",
    "summarize_cv",
    "plot_confusion",
    "plot_pr_roc",
    "plot_calibration",
    "leakage_report",
]

__version__ = "1.0.0"
