"""Scoring core. Deliberately framework-free so it can be unit-tested."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

log = logging.getLogger("scoring")


class SchemaError(ValueError):
    """Raised when a request violates the input contract. Blocking, by design."""


class Scorer:
    def __init__(self, artifact_dir: str | Path):
        self.dir = Path(artifact_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.features = self.manifest["training"]["feature_order"]
        self.schema = self.manifest["schema"]
        self.threshold = self.manifest["decision"]["threshold"]
        self._verify_artifact()
        self.model = joblib.load(self.dir / "model.joblib")
        self._verify_environment()
        self._verify_reference_batch()

    # -- startup checks (all fatal) ---------------------------------------
    def _verify_artifact(self):
        digest = hashlib.sha256((self.dir / "model.joblib").read_bytes()).hexdigest()
        if digest != self.manifest["model_sha256"]:
            raise RuntimeError("model.joblib does not match the manifest checksum")

    def _verify_environment(self):
        import sklearn
        want = self.manifest["environment"]["scikit_learn"]
        if sklearn.__version__ != want:
            raise RuntimeError(
                f"scikit-learn {sklearn.__version__} != manifest {want}. "
                "Refusing to start: a version mismatch can change predictions silently."
            )

    def _verify_reference_batch(self, tol: float = 1e-9):
        ref = pd.read_json(self.dir / "reference_inputs.json", orient="table")
        want = pd.read_csv(self.dir / "reference_predictions.csv")["expected_proba"].to_numpy()
        got = self.model.predict_proba(ref[self.features])[:, 1]
        worst = float(abs(got - want).max())
        if worst > tol:
            raise RuntimeError(f"reference batch mismatch: max |diff| {worst:.3e} > {tol:.0e}")
        log.info("reference batch verified, max diff %.2e", worst)

    # -- request validation (blocking) ------------------------------------
    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise SchemaError(f"missing required fields: {missing}")
        for col in self.features:
            spec = self.schema[col]
            s = df[col]
            if "min" in spec:
                if not pd.api.types.is_numeric_dtype(s):
                    raise SchemaError(f"{col}: expected numeric, got {s.dtype}")
                lo = spec["p01"] - 3 * spec["std"]
                hi = spec["p99"] + 3 * spec["std"]
                bad = float(((s < lo) | (s > hi)).mean())
                if bad > 0.01:
                    raise SchemaError(
                        f"{col}: {bad:.1%} of values outside [{lo:.1f}, {hi:.1f}]. "
                        "Rejecting the batch rather than scoring it."
                    )
            else:
                unseen = set(s.dropna().astype(str)) - set(spec["categories"])
                if unseen:
                    log.warning("%s: unseen categories %s", col, sorted(unseen)[:5])

    # -- scoring -----------------------------------------------------------
    def score(self, payload: list[dict]) -> list[dict]:
        t0 = perf_counter()
        df = pd.DataFrame(payload)
        self.validate(df)
        proba = self.model.predict_proba(df[self.features])[:, 1]
        elapsed_ms = (perf_counter() - t0) * 1000
        out = []
        for i, p in enumerate(proba):
            rec = {
                "request_id": str(uuid.uuid4()),
                "scored_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "model_version": self.manifest["version"],
                "model_sha256": self.manifest["model_sha256"][:16],
                "probability": round(float(p), 6),
                "threshold": self.threshold,
                "decision": "refer" if p >= self.threshold else "approve",
                "latency_ms": round(elapsed_ms / len(proba), 3),
            }
            # Structured log of EVERY prediction. Without this there is no
            # monitoring, only a monitoring plan.
            log.info(json.dumps(rec | {"features": payload[i]}))
            out.append(rec)
        return out
