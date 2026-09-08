"""FastAPI wrapper. All the logic lives in scoring.py."""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.scoring import SchemaError, Scorer

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.FileHandler("predictions.log"),
                              logging.StreamHandler()])

ARTIFACT = Path(__file__).resolve().parents[1] / "artifact"
app = FastAPI(title="Consumer default risk", version="1.0.0")
scorer = Scorer(ARTIFACT)          # startup checks run here; a bad artifact fails to boot


class ScoreRequest(BaseModel):
    applications: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": scorer.manifest["version"]}


@app.get("/metadata")
def metadata():
    m = scorer.manifest
    return {k: m[k] for k in ("name", "version", "created_utc", "model_sha256",
                              "environment", "decision", "holdout")}


@app.post("/score")
def score(req: ScoreRequest):
    try:
        return {"results": scorer.score(req.applications)}
    except SchemaError as e:
        # 422: the request violated the input contract. This is the control that
        # Module 13.5 identified as the one that would have caught the incident.
        raise HTTPException(status_code=422, detail=str(e))
