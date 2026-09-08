import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.scoring import SchemaError, Scorer

ART = Path(__file__).resolve().parents[1] / "artifact"


@pytest.fixture(scope="module")
def scorer():
    return Scorer(ART)


@pytest.fixture(scope="module")
def rows(scorer):
    ref = pd.read_json(ART / "reference_inputs.json", orient="table")
    return ref[scorer.features].head(20).to_dict("records")


def test_reference_predictions_are_exact(scorer):
    """The control that catches 'loads but differs'. Module 13.1."""
    ref = pd.read_json(ART / "reference_inputs.json", orient="table")
    want = pd.read_csv(ART / "reference_predictions.csv")["expected_proba"].to_numpy()
    got = scorer.model.predict_proba(ref[scorer.features])[:, 1]
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_threshold_matches_its_documented_basis(scorer):
    """Module 13.5 found a manifest whose threshold contradicted its own basis."""
    d = scorer.manifest["decision"]
    assert abs(d["threshold"] - d["cost_fp"] / (d["cost_fp"] + d["cost_fn"])) < 1e-9


def test_missing_field_is_rejected(scorer, rows):
    bad = [{k: v for k, v in rows[0].items() if k != "credit_score"}]
    with pytest.raises(SchemaError, match="missing required fields"):
        scorer.score(bad)


def test_out_of_range_batch_is_rejected(scorer, rows):
    """The Module 13.5 incident, as a permanent regression test."""
    bad = [dict(r) for r in rows]
    for r in bad:
        r["credit_score"] = (r["credit_score"] - 300) / 550 * 100     # 0-100 scale
    with pytest.raises(SchemaError, match="credit_score"):
        scorer.score(bad)


def test_latency_budget(scorer, rows):
    from time import perf_counter
    scorer.score(rows[:1])
    t0 = perf_counter()
    for _ in range(20):
        scorer.score(rows[:1])
    per_call_ms = (perf_counter() - t0) / 20 * 1000
    assert per_call_ms < 50, f"single-row latency {per_call_ms:.1f} ms exceeds the 50 ms budget"


def test_decision_matches_threshold(scorer, rows):
    for rec in scorer.score(rows):
        expected = "refer" if rec["probability"] >= rec["threshold"] else "approve"
        assert rec["decision"] == expected


def test_every_prediction_is_logged(scorer, rows, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="scoring"):
        scorer.score(rows[:5])
    logged = [r for r in caplog.records if r.name == "scoring"]
    assert len(logged) >= 5
    assert "request_id" in json.loads(logged[-1].message)


def test_fairness_position_is_documented(scorer):
    """The model does NOT clear the four-fifths screen, and the manifest says so.

    A release must not be able to quietly forget that. Module 12 established the
    disparity; this test asserts it is still disclosed.
    """
    f = scorer.manifest["fairness"]
    assert "measured_at_release_holdout" in f and "clears_screen" in f
    assert f["clears_screen"] == (f["measured_at_release_holdout"] >= f["regulatory_screen"])
    if not f["clears_screen"]:
        assert f.get("note"), "a disparity below the screen requires written justification"


def test_fairness_has_not_regressed(scorer):
    """Guard the DOCUMENTED position, not an aspiration.

    Asserting the four-fifths rule here would ship a test that fails on day one,
    which trains everyone to skip it. The useful test is that the model has not
    got worse than the position that was reviewed and signed off.
    """
    df = pd.read_json(ART / "fairness_sample.json", orient="table")[scorer.features]
    p = scorer.model.predict_proba(df)[:, 1]
    band = pd.cut(df["age"], [18, 30, 45, 60, 100])
    sel = (pd.DataFrame({"g": band, "d": p >= scorer.threshold})
           .dropna().groupby("g", observed=True)["d"].mean())
    ratio = float(sel.min() / sel.max()) if sel.max() > 0 else 1.0
    f = scorer.manifest["fairness"]
    floor = f["fairness_sample_ratio"] - f["regression_tolerance"]
    assert ratio >= floor, (
        f"selection-rate ratio {ratio:.3f} on the fairness sample is below the "
        f"recorded release value {f['fairness_sample_ratio']:.3f} by more than "
        f"{f['regression_tolerance']}. This is a regression and needs review."
    )
