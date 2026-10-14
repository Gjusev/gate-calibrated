"""Calibrators (plan section 4.5). Both variants expose fit(X, y) /
predict_proba(X).

Features per gated action (dict keys, produced from collected rows):
  verbalized: float in [0, 1]
  agreement: float in [0, 1]
  tier: "write" | "irreversible"
  step_index: int >= 0

Variant A recalibrates the verbalized probability with temperature scaling
(one parameter T). Variant B is a logistic regression on
[logit(verbalized), agreement, tier one-hot, log1p(step_index)].
Implemented with numpy only (no scipy/sklearn in the dependency tree).
"""

import math
from typing import Dict, List, Sequence

import numpy as np

EPS = 1e-6
TIER_VALUES = ("write", "irreversible")


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def features_matrix(rows: Sequence[Dict]) -> np.ndarray:
    """[logit(verbalized), agreement, is_irreversible, log1p(step_index)]."""
    return np.array(
        [
            [
                logit(r["verbalized"]),
                r["agreement"],
                1.0 if r["tier"] == "irreversible" else 0.0,
                math.log1p(r["step_index"]),
            ]
            for r in rows
        ]
    )


def labels_vector(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([1.0 if r["correct"] else 0.0 for r in rows])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class TemperatureScaling:
    """p_hat = sigmoid(logit(verbalized) / T)."""

    name = "A_temperature_scaling"

    def __init__(self) -> None:
        self.T = 1.0

    def fit(self, rows: Sequence[Dict]):
        p = np.clip(
            np.array([r["verbalized"] for r in rows], dtype=float), EPS, 1 - EPS
        )
        y = labels_vector(rows)
        z = np.log(p / (1 - p))

        def nll(T: float) -> float:
            q = _sigmoid(z / T)
            return -np.mean(y * np.log(q + 1e-12) + (1 - y) * np.log(1 - q + 1e-12))

        # coarse grid on log(T), then local refinement
        Ts = np.exp(np.linspace(math.log(0.05), math.log(20.0), 200))
        best = Ts[int(np.argmin([nll(T) for T in Ts]))]
        lo, hi = best / 1.1, best * 1.1
        for _ in range(60):  # golden-section-ish bisection refine
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if nll(m1) < nll(m2):
                hi = m2
            else:
                lo = m1
        self.T = 0.5 * (lo + hi)
        return self

    def predict_proba(self, rows: Sequence[Dict]) -> np.ndarray:
        p = np.clip(
            np.array([r["verbalized"] for r in rows], dtype=float), EPS, 1 - EPS
        )
        return _sigmoid(np.log(p / (1 - p)) / self.T)

    def to_json(self) -> Dict:
        return {"variant": self.name, "T": float(self.T)}

    @classmethod
    def from_json(cls, blob: Dict) -> "TemperatureScaling":
        ts = cls()
        ts.T = float(blob["T"])
        return ts


class LogisticRegression:
    """IRLS with L2 regularization and standardized features."""

    name = "B_logistic_regression"
    l2 = 1.0

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None

    def fit(self, rows: Sequence[Dict]):
        X = features_matrix(rows)
        y = labels_vector(rows)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        Z = np.hstack([Z, np.ones((len(Z), 1))])  # bias column
        w = np.zeros(Z.shape[1])
        lam = np.diag([self.l2] * (Z.shape[1] - 1) + [0.0])  # no penalty on bias
        for _ in range(50):
            q = _sigmoid(Z @ w)
            grad = Z.T @ (q - y) + lam @ w
            W = np.clip(q * (1 - q), 1e-6, None)
            H = Z.T @ (Z * W[:, None]) + lam
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            w_new = w - step
            if np.max(np.abs(w_new - w)) < 1e-8:
                w = w_new
                break
            w = w_new
        self.w = w
        return self

    def predict_proba(self, rows: Sequence[Dict]) -> np.ndarray:
        X = features_matrix(rows)
        Z = (X - self.mu) / self.sd
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        return _sigmoid(Z @ self.w)

    def to_json(self) -> Dict:
        return {
            "variant": self.name,
            "w": [float(x) for x in self.w],
            "mu": [float(x) for x in self.mu],
            "sd": [float(x) for x in self.sd],
            "l2": self.l2,
        }

    @classmethod
    def from_json(cls, blob: Dict) -> "LogisticRegression":
        lr = cls(l2=blob.get("l2", 1.0))
        lr.w = np.array(blob["w"])
        lr.mu = np.array(blob["mu"])
        lr.sd = np.array(blob["sd"])
        return lr


def calibrator_from_json(blob: Dict):
    if blob["variant"] == TemperatureScaling.name:
        return TemperatureScaling.from_json(blob)
    if blob["variant"] == LogisticRegression.name:
        return LogisticRegression.from_json(blob)
    raise ValueError(f"unknown calibrator variant {blob['variant']!r}")


def raw_verbalized(rows: Sequence[Dict]) -> np.ndarray:
    """Uncalibrated baseline."""
    return np.array([r["verbalized"] for r in rows], dtype=float)
