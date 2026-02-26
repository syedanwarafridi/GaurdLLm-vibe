"""
Layer-1 Detector — Main Entry Point
=====================================
Ties together the three feature signals, logistic fusion,
temperature calibration, and threshold routing.

                          ┌─────────────────────────┐
      text x ──────────── │   CurvatureFeature       │──► fcurv(x)
                          │   LogRankFeature          │──► frank(x)     ┐
                          │   ClassifierFeature       │──► g(x)         │
                          └─────────────────────────┘                  │
                                                                        ▼
                                                           ┌────────────────────┐
                                                           │  LogisticFusion     │
                                                           │  s_raw = σ(z(x))   │
                                                           └────────┬───────────┘
                                                                    │
                                                           ┌────────▼───────────┐
                                                           │  TemperatureScaler  │
                                                           │  s = σ(z/T)         │
                                                           └────────┬───────────┘
                                                                    │
                                                    ┌───────────────▼──────────────┐
                                                    │        ThresholdRouter        │
                                                    │  allow / gray_zone / mitigate │
                                                    └───────────────────────────────┘

GPU changes vs. CPU version:
  - Device resolved via resolve_device("auto") — uses CUDA if available
  - fp16 flag threaded through to all sub-components
  - GPU memory cleared after batch evaluation via torch.cuda.empty_cache()
  - from_config reads fp16 from config["project"]["fp16"]

Public API:
    detector = Layer1Detector.from_config(cfg)
    result   = detector.detect(text)           # → DetectionResult
    results  = detector.detect_batch(texts)    # → List[DetectionResult]
"""

from __future__ import annotations

import json
import torch
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from layer1.features import (
    CurvatureFeature, LogRankFeature, ClassifierFeature, resolve_device,
)
from layer1.fusion import (
    LogisticFusion, TemperatureScaler, ThresholdRouter,
    DECISION_ALLOW, DECISION_GRAY_ZONE, DECISION_MITIGATE,
)
from utils.logger import get_logger

logger = get_logger("layer1.detector")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Output of Layer-1 for a single text.

    Fields:
        text_preview    : first 100 chars of input
        fcurv           : curvature feature score   ∈ [0, 1]
        frank           : log-rank feature score    ∈ [0, 1]
        g_classifier    : classifier feature score  ∈ [0, 1]
        raw_score       : fused score before calibration
        calibrated_score: fused score after temperature scaling
        decision        : "allow" | "gray_zone" | "mitigate"
    """
    text_preview:     str
    fcurv:            float
    frank:            float
    g_classifier:     float
    raw_score:        float
    calibrated_score: float
    decision:         str

    def to_dict(self) -> Dict:
        return asdict(self)

    def is_safe(self) -> bool:
        return self.decision == DECISION_ALLOW

    def needs_judge(self) -> bool:
        return self.decision == DECISION_GRAY_ZONE

    def is_harmful(self) -> bool:
        return self.decision == DECISION_MITIGATE


# ─────────────────────────────────────────────────────────────────────────────
# Layer-1 Detector
# ─────────────────────────────────────────────────────────────────────────────

class Layer1Detector:
    """
    Full Layer-1 pipeline: features → fusion → calibration → routing.

    Construction:
        Use Layer1Detector.from_config(cfg_dict) for standard builds.
        Use Layer1Detector(...) for unit tests / custom builds.
    """

    def __init__(
        self,
        curvature_feature:  CurvatureFeature,
        logrank_feature:    LogRankFeature,
        classifier_feature: ClassifierFeature,
        fusion:  LogisticFusion,
        scaler:  TemperatureScaler,
        router:  ThresholdRouter,
    ):
        self.curvature_feature  = curvature_feature
        self.logrank_feature    = logrank_feature
        self.classifier_feature = classifier_feature
        self.fusion = fusion
        self.scaler = scaler
        self.router = router

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "Layer1Detector":
        """
        Build a fresh (untrained) Layer1Detector from config.yaml dict.
        Call fit_fusion() and fit_scaler() before using for inference.
        """
        l1 = cfg["layer1"]
        device = resolve_device(cfg["project"]["device"])
        fp16   = cfg["project"].get("fp16", True)

        logger.info(f"Initialising Layer-1 Detector (device={device}, fp16={fp16})...")

        # Shared scoring LM — loaded once, used by both curvature and log-rank
        curv = CurvatureFeature(
            model_name="distilgpt2",
            num_perturbations=l1["curvature"]["num_perturbations"],
            temperature_range=tuple(l1["curvature"]["temperature_range"]),
            device=device,
            fp16=fp16,
        )
        rank = LogRankFeature(
            shared_model=curv.model,
            shared_tokenizer=curv.tokenizer,
        )

        clf_cfg = l1["classifier"]
        clf = ClassifierFeature(
            checkpoint_path=None,   # not trained yet
            backbone=clf_cfg["backbone"],
            max_length=clf_cfg["max_length"],
            device=device,
            fp16=fp16,
        )

        th = l1["thresholds"]
        logger.info(f"Layer-1 thresholds: τ_low={th['low']}, τ_high={th['high']}")

        return cls(
            curvature_feature=curv,
            logrank_feature=rank,
            classifier_feature=clf,
            fusion=LogisticFusion(),
            scaler=TemperatureScaler(T=l1["calibration"]["temperature"]),
            router=ThresholdRouter(tau_low=th["low"], tau_high=th["high"]),
        )

    @classmethod
    def from_artifacts(
        cls,
        cfg: dict,
        fusion_path: str,
        scaler_path: str,
        classifier_checkpoint: str,
    ) -> "Layer1Detector":
        """
        Load a fully-fitted Layer-1 from saved artifacts.
        Use this for inference after training is complete.
        """
        device = resolve_device(cfg["project"]["device"])
        fp16   = cfg["project"].get("fp16", True)
        clf_cfg = cfg["layer1"]["classifier"]

        detector = cls.from_config(cfg)
        detector.fusion.load(fusion_path)
        detector.scaler.load(scaler_path)
        detector.classifier_feature = ClassifierFeature(
            checkpoint_path=classifier_checkpoint,
            backbone=clf_cfg["backbone"],
            max_length=clf_cfg["max_length"],
            device=device,
            fp16=fp16,
        )
        return detector

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, text: str) -> DetectionResult:
        """
        Run Layer-1 on a single text.

        Steps:
          1. Compute three feature scores (curvature, log-rank, classifier)
          2. Fuse via logistic stacker → raw_score
          3. Calibrate via temperature scaling → calibrated_score
          4. Route to allow / gray_zone / mitigate
        """
        if not text or not text.strip():
            return DetectionResult(
                text_preview="",
                fcurv=0.5, frank=0.5, g_classifier=0.5,
                raw_score=0.5, calibrated_score=0.5,
                decision=DECISION_GRAY_ZONE,
            )

        fcurv = self.curvature_feature.compute(text)
        frank = self.logrank_feature.compute(text)
        g     = self.classifier_feature.compute(text)

        raw_score  = self.fusion.raw_score(fcurv, frank, g)
        calibrated = self.scaler.scale(raw_score)
        decision   = self.router.route(calibrated)

        logger.debug(
            f"L1 | fcurv={fcurv:.3f} frank={frank:.3f} g={g:.3f} "
            f"raw={raw_score:.3f} cal={calibrated:.3f} → {decision}"
        )
        return DetectionResult(
            text_preview=text[:100],
            fcurv=round(fcurv, 4),
            frank=round(frank, 4),
            g_classifier=round(g, 4),
            raw_score=round(raw_score, 4),
            calibrated_score=round(calibrated, 4),
            decision=decision,
        )

    def detect_batch(self, texts: List[str], log_every: int = 200) -> List[DetectionResult]:
        """
        Run Layer-1 on a list of texts.
        Clears GPU cache every `log_every` items to prevent OOM on large batches.
        """
        results = []
        for i, text in enumerate(texts):
            if i > 0 and i % log_every == 0:
                logger.info(f"  Layer-1: {i}/{len(texts)} processed...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(self.detect(text))

        # Final cache clear after batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    # ── Calibration helpers ───────────────────────────────────────────────────

    def fit_fusion(self, records: list) -> None:
        """
        Fit the logistic stacker on labelled records.
        Expects each record to have 'fcurv', 'frank', 'g' pre-computed.
        If missing, recomputes them (slow — prefer caching first).
        """
        import numpy as np

        features, labels = [], []
        for rec in records:
            fc = rec.get("fcurv") if rec.get("fcurv") is not None else self.curvature_feature.compute(rec["text"])
            fr = rec.get("frank") if rec.get("frank") is not None else self.logrank_feature.compute(rec["text"])
            g  = rec.get("g")     if rec.get("g")     is not None else self.classifier_feature.compute(rec["text"])
            features.append([fc, fr, g])
            labels.append(rec["label"])

        self.fusion.fit(np.array(features), np.array(labels))

    def fit_scaler(self, records: list) -> float:
        """
        Fit temperature scaler on validation records.
        Expects 'fcurv', 'frank', 'g' to be pre-computed in each record.
        """
        raw_scores, labels = [], []
        for rec in records:
            fc = rec.get("fcurv", self.curvature_feature.compute(rec["text"]))
            fr = rec.get("frank", self.logrank_feature.compute(rec["text"]))
            g  = rec.get("g",     self.classifier_feature.compute(rec["text"]))
            raw_scores.append(self.fusion.raw_score(fc, fr, g))
            labels.append(rec["label"])

        return self.scaler.calibrate(raw_scores, labels)

    def save_fusion_artifacts(
        self,
        fusion_path: str = "artifacts/layer1/fusion.joblib",
        scaler_path: str = "artifacts/layer1/scaler.json",
    ) -> None:
        self.fusion.save(fusion_path)
        self.scaler.save(scaler_path)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, records: list) -> Dict:
        """
        Evaluate Layer-1 on labelled records.
        Returns: accuracy, AUROC, AUPRC, FPR@95%TPR.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

        if not records:
            logger.warning("evaluate() called with empty records.")
            return {}

        texts = [r["text"] for r in records]
        results = self.detect_batch(texts)

        true_labels  = [r["label"] for r in records]
        pred_scores  = [r.calibrated_score for r in results]
        pred_labels  = [0 if r.is_safe() else 1 for r in results]

        # mitigate = positive class (label 1 for metric computation)
        mitigate_labels = [1 - l for l in true_labels]

        auroc = roc_auc_score(mitigate_labels, pred_scores)
        auprc = average_precision_score(mitigate_labels, pred_scores)
        acc   = accuracy_score(mitigate_labels, pred_labels)
        threshold, fpr95 = self.router.compute_fpr_at_tpr(
            pred_scores, true_labels, target_tpr=0.95
        )

        metrics = {
            "accuracy":           round(acc,       4),
            "auroc":              round(auroc,      4),
            "auprc":              round(auprc,      4),
            "fpr@95tpr":          round(fpr95,      4),
            "threshold_at_95tpr": round(threshold,  4),
            "n_samples":          len(records),
        }
        logger.info(f"Layer-1 eval: {json.dumps(metrics)}")
        return metrics
