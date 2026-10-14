import math
import random

import pytest

from gate.calibrator import (
    LogisticRegression,
    TemperatureScaling,
    logit,
    raw_verbalized,
)
from eval.metrics import brier, ece


def _rows(n=4000, seed=0, miscal_T=3.0, agreement_informative=True):
    """verbalized is a temperature-miscalibrated view of the true success
    probability; agreement carries extra signal for variant B."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        p_true = 1 / (1 + math.exp(-(z)))
        # miscalibrated prediction: sharper than truth
        v = 1 / (1 + math.exp(-(miscal_T * z)))
        if agreement_informative:
            agreement = min(max(p_true + rng.gauss(0, 0.15), 0), 1)
        else:
            agreement = rng.random()
        correct = rng.random() < p_true
        rows.append(
            {
                "verbalized": min(max(v, 1e-4), 1 - 1e-4),
                "agreement": agreement,
                "tier": "write" if rng.random() < 0.7 else "irreversible",
                "step_index": rng.randrange(10),
                "correct": correct,
            }
        )
    return rows


def test_temperature_scaling_reduces_ece():
    train, test = _rows(4000, seed=1), _rows(4000, seed=2)
    raw = raw_verbalized(test)
    ts = TemperatureScaling().fit(train)
    y = [1.0 if r["correct"] else 0.0 for r in test]
    ece_raw, ece_ts = ece(y, raw), ece(y, ts.predict_proba(test))
    assert ece_ts < ece_raw
    # the recovered temperature should be near the miscalibration factor
    assert 2.0 < ts.T < 5.0


def test_temperature_scaling_identity_when_calibrated():
    train, test = _rows(2000, seed=3, miscal_T=1.0), _rows(2000, seed=6, miscal_T=1.0)
    ts = TemperatureScaling().fit(train)
    assert 0.7 < ts.T < 1.3


def test_logistic_regression_beats_raw_brier():
    train, test = _rows(4000, seed=4), _rows(4000, seed=5)
    lr = LogisticRegression().fit(train)
    y = [1.0 if r["correct"] else 0.0 for r in test]
    b_raw = brier(y, raw_verbalized(test))
    b_lr = brier(y, lr.predict_proba(test))
    assert b_lr < b_raw


def test_saturated_verbalized_does_not_crash():
    rows = [
        {
            "verbalized": 1.0,
            "agreement": 1.0,
            "tier": "write",
            "step_index": 0,
            "correct": True,
        }
    ] * 50 + [
        {
            "verbalized": 1.0,
            "agreement": 0.2,
            "tier": "write",
            "step_index": 1,
            "correct": False,
        }
    ] * 10
    ts = TemperatureScaling().fit(rows)
    lr = LogisticRegression().fit(rows)
    assert len(ts.predict_proba(rows)) == 60
    assert len(lr.predict_proba(rows)) == 60
    # saturated verbalized alone cannot rank; agreement can, via variant B
    p = lr.predict_proba(rows)
    assert p[0] > p[-1]


def test_logit_clip():
    assert logit(0.5) == 0
    assert logit(0.0) < -10
    assert logit(1.0) > 10
