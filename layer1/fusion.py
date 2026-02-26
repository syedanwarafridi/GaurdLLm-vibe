"""
Layer-1 Signal Fusion & Calibration
=====================================
Implements:

  1. LogisticFusion  — combines the three feature scores via a logistic stacker
                       z(x) = w0 + w1*fcurv + w2*frank + w3*g(x)
                       s_raw(x) = sigmoid(z(x))

  2. TemperatureScaler — post-hoc calibration on a held-out set
                         s(x) = z(x) / T,  T chosen by validation NLL

  3. ThresholdRouter  — routes the final score to one of three decisions:
                         allow     if  s(x) < τ_low
                         gray-zone if  τ_low ≤ s(x) ≤ τ_high   → send to Judge
                         mitigate  if  s(x) > τ_high

References:
  [Mitchell et al., 2023] DetectGPT
  [Su et al., 2023]       DetectLLM
  [Dugan et al., 2024]    RAID
"""

from __future__ import annotations

import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from typing import List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger("layer1.fusion")

DECISION_ALLOW     = "allow"
DECISION_GRAY_ZONE = "gray_zone"
DECISION_MITIGATE  = "mitigate"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logistic Fusion
# ─────────────────────────────────────────────────────────────────────────────

class LogisticFusion:
    """
    Logistic stacker that combines the three Layer-1 signals.

    Input features:  [fcurv(x), frank(x), g(x)]  each ∈ [0, 1]
    Output:          s_raw(x) ∈ [0, 1]  (uncalibrated fusion score)

    Fitting uses scikit-learn LogisticRegression on (features, labels) from
    the distillation pool.  Weights are saved/loaded from a JSON file.
    """

    def __init__(self):
        self._clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        self._fitted = False
        # Default weights before fitting (equal weighting)
        self._w = np.array([0.0, 1.0, 1.0, 1.0])   # [bias, w1, w2, w3]

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the stacker.

        Args:
            features: shape (N, 3) — columns: [fcurv, frank, g_classifier]
            labels:   shape (N,)   — 1=allow, 0=mitigate
        """
        logger.info(f"Fitting logistic stacker on {len(labels)} samples.")
        self._clf.fit(features, labels)
        self._fitted = True

        # Store weights for manual scoring
        self._w = np.concatenate(
            [self._clf.intercept_, self._clf.coef_[0]]
        )
        logger.info(f"Stacker weights: bias={self._w[0]:.4f}, "
                    f"w_curv={self._w[1]:.4f}, w_rank={self._w[2]:.4f}, "
                    f"w_clf={self._w[3]:.4f}")

    def raw_score(self, fcurv: float, frank: float, g: float) -> float:
        """
        z(x) = w0 + w1*fcurv + w2*frank + w3*g(x)
        s_raw(x) = sigmoid(z(x))

        Returns s_raw ∈ [0, 1]. Higher → more likely harmful.
        """
        if self._fitted:
            features = np.array([[fcurv, frank, g]])
            prob = self._clf.predict_proba(features)[0]
            # class 0 = mitigate, class 1 = allow  (depends on fit order)
            # Return P(mitigate) = 1 - P(allow)
            allow_idx = list(self._clf.classes_).index(1)
            return float(1.0 - prob[allow_idx])
        else:
            # Fall back to simple equal-weight average before fitting
            z = self._w[0] + self._w[1] * fcurv + self._w[2] * frank + self._w[3] * g
            return float(torch.sigmoid(torch.tensor(z)).item())

    def feature_vector(self, fcurv: float, frank: float, g: float) -> np.ndarray:
        return np.array([fcurv, frank, g])

    def save(self, path: str) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)
        logger.info(f"Logistic stacker saved → {path}")

    def load(self, path: str) -> None:
        import joblib
        self._clf = joblib.load(path)
        self._fitted = True
        self._w = np.concatenate(
            [self._clf.intercept_, self._clf.coef_[0]]
        )
        logger.info(f"Logistic stacker loaded ← {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temperature Scaler (Calibration)
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler:
    """
    Post-hoc calibration via temperature scaling.

    Finds T* that minimises NLL on the held-out validation set:
        s(x) = sigmoid(logit(s_raw(x)) / T)

    Reference: Guo et al., 2017 — "On Calibration of Modern Neural Networks"
    """

    def __init__(self, T: float = 1.0):
        self.T = T
        self._optimised = False

    def calibrate(self, raw_scores: List[float], labels: List[int]) -> float:
        """
        Optimise temperature T on (raw_scores, labels).

        Args:
            raw_scores: s_raw(x) ∈ [0, 1] from LogisticFusion
            labels:     ground-truth (1=allow, 0=mitigate)

        Returns:
            Optimal T.
        """
        eps = 1e-7
        logits = torch.tensor(
            [math.log((s + eps) / (1 - s + eps)) for s in raw_scores],
            dtype=torch.float32,
        )
        targets = torch.tensor(labels, dtype=torch.float32)

        T_param = torch.nn.Parameter(torch.ones(1))
        optimiser = torch.optim.LBFGS([T_param], lr=0.01, max_iter=500)

        def closure():
            optimiser.zero_grad()
            scaled = torch.sigmoid(logits / T_param)
            loss = F.binary_cross_entropy(scaled, targets)
            loss.backward()
            return loss

        optimiser.step(closure)
        self.T = float(T_param.item())
        self._optimised = True
        logger.info(f"Temperature scaling: T* = {self.T:.4f}")
        return self.T

    def scale(self, raw_score: float) -> float:
        """Apply temperature scaling to a raw fusion score."""
        eps = 1e-7
        logit = math.log((raw_score + eps) / (1 - raw_score + eps))
        return float(torch.sigmoid(torch.tensor(logit / self.T)).item())

    def scale_batch(self, raw_scores: List[float]) -> List[float]:
        return [self.scale(s) for s in raw_scores]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"T": self.T, "optimised": self._optimised}, f)
        logger.info(f"Temperature scaler saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.T = data["T"]
        self._optimised = data["optimised"]
        logger.info(f"Temperature scaler loaded ← {path} (T={self.T:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Threshold Router
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdRouter:
    """
    Routes a calibrated score s(x) to one of three decisions.

        allow     if  s(x) < τ_low
        gray_zone if  τ_low ≤ s(x) ≤ τ_high   → escalate to Layer-2 Judge
        mitigate  if  s(x) > τ_high

    τ_low and τ_high are set from config (TBD from experiments; defaults: 0.3, 0.7).
    """

    def __init__(self, tau_low: float = 0.3, tau_high: float = 0.7):
        self.tau_low = tau_low
        self.tau_high = tau_high

    def route(self, score: float) -> str:
        """Returns DECISION_ALLOW, DECISION_GRAY_ZONE, or DECISION_MITIGATE."""
        if score < self.tau_low:
            return DECISION_ALLOW
        elif score > self.tau_high:
            return DECISION_MITIGATE
        else:
            return DECISION_GRAY_ZONE

    def route_batch(self, scores: List[float]) -> List[str]:
        return [self.route(s) for s in scores]

    def compute_fpr_at_tpr(
        self,
        scores: List[float],
        labels: List[int],
        target_tpr: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Find threshold τ that achieves target TPR, report FPR at that threshold.
        (Used for RAID-style FPR@95%TPR metric.)

        Args:
            scores: calibrated scores (higher → mitigate)
            labels: 0=mitigate (positive), 1=allow (negative)
            target_tpr: desired true-positive rate on mitigate class

        Returns:
            (threshold, fpr_at_target_tpr)
        """
        from sklearn.metrics import roc_curve
        # Flip labels: mitigate=1 (positive class), allow=0 (negative)
        binary_labels = [1 - l for l in labels]
        fpr, tpr, thresholds = roc_curve(binary_labels, scores)

        # Find threshold closest to target TPR
        idx = np.argmin(np.abs(tpr - target_tpr))
        return float(thresholds[idx]), float(fpr[idx])

    def update_thresholds(self, tau_low: float, tau_high: float) -> None:
        self.tau_low = tau_low
        self.tau_high = tau_high
        logger.info(f"Thresholds updated: τ_low={tau_low:.3f}, τ_high={tau_high:.3f}")
