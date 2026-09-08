# Consumer default risk — scoring service

Model version **1.0.0**, built 2026-09-07T21:45:44+00:00
against scikit-learn 1.8.0.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The service performs three fatal checks at startup and **will not boot** if any
fails: the model file must match the manifest checksum, the scikit-learn version
must match the one it was built with, and the 200-row reference batch must
reproduce to 1e-9. A service that starts is a service whose model is the one
that was validated.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health` | liveness + model version |
| `GET /metadata` | manifest: version, checksum, environment, threshold and its basis, holdout metrics |
| `POST /score` | score a batch; `422` if the request violates the input contract |

```bash
curl -s localhost:8000/score -H 'content-type: application/json' \
  -d '{"applications": [{"age": 41, "credit_score": 655, "annual_income": 38000, "...": "..."}]}'
```

## The input contract is blocking, on purpose

A batch with values far outside the training range is **rejected with a 422**,
not scored. This is the control identified in Module 13.5: the incident there
was a unit change on `credit_score` that produced ten months of confidently
wrong declines without a single exception. A warning in a log would not have
stopped it. Refusing the batch would have.

## Every prediction is logged

`predictions.log` receives one JSON line per prediction: request id, timestamp,
model version and checksum, probability, threshold, decision, latency, and the
input features. `monitor.py` reads that log and emits the alerts the model memo
promised — refer-rate shift, PSI on inputs, p95 latency — exiting non-zero on a
critical breach so it can be wired to a scheduler.

```bash
python monitor.py predictions.log
```

## Tests

```bash
pytest -q
```

Covering: reference-prediction equality, the manifest threshold matching its own
documented basis, schema rejection (missing field and out-of-range batch), the
latency budget, decision/threshold consistency, prediction logging, and a
fairness regression test against the documented action level.

## Decision threshold

`0.2308` — C_FP/(C_FP+C_FN) with C_FP=1800.0, C_FN=6000.0.
Changing it is a business decision with a fairness consequence (see the model
documentation), not a configuration tweak.
