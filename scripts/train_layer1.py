"""
Phase 2 — Train Layer-1 Classifier
=====================================
Steps:
  1. Load the distillation pool (train / val / test splits)
  2. Compute curvature + log-rank features (cached to avoid recomputation)
  3. Fit LogisticFusion stacker on train features (placeholder g=0.5)
  4. Fine-tune RoBERTa+LoRA classifier on train records (GPU + AMP)
  5. Re-compute g(x) with trained classifier; re-fit fusion
  6. Calibrate TemperatureScaler on validation set
  7. Evaluate on test set: AUROC, AUPRC, FPR@95%TPR, accuracy
  8. Save all artifacts to artifacts/layer1/

GPU notes:
  - Device auto-detected from config ("auto" → cuda if available)
  - Mixed-precision training via AMP (float16 on GPU)
  - GPU memory logged at key steps via torch.cuda.memory_allocated()
  - torch.cuda.empty_cache() called between heavy phases
  - Feature computation cached so reruns skip the slow step

Usage (activate your conda env first):
    conda activate AI
    python scripts/train_layer1.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.config_loader import load_config
from utils.data_utils import load_jsonl, set_seed, save_jsonl
from utils.logger import get_logger

from layer1.features   import CurvatureFeature, LogRankFeature, ClassifierFeature, resolve_device
from layer1.fusion     import LogisticFusion, TemperatureScaler, ThresholdRouter
from layer1.classifier import build_lora_classifier, ClassifierTrainer
from layer1.detector   import Layer1Detector

logger = get_logger("train_layer1")


# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_gpu_memory(tag: str = "") -> None:
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
        logger.info(
            f"[GPU mem{' ' + tag if tag else ''}] "
            f"allocated={allocated:.2f}GB  reserved={reserved:.2f}GB"
        )


def clear_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation with caching
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    records: list,
    curv_feat: CurvatureFeature,
    rank_feat: LogRankFeature,
    cache_path: str,
) -> list:
    """
    Compute curvature + log-rank features for all records.
    Results are cached to a .jsonl file — if the cache exists, it is loaded
    directly to skip expensive recomputation.
    """
    if Path(cache_path).exists():
        logger.info(f"Feature cache found, loading: {cache_path}")
        return load_jsonl(cache_path)

    logger.info(f"Computing features for {len(records)} records → {cache_path}")
    enriched = []

    for i, rec in enumerate(tqdm(records, desc="Features", unit="sample")):
        rec = rec.copy()
        try:
            rec["fcurv"] = curv_feat.compute(rec["text"])
            rec["frank"] = rank_feat.compute(rec["text"])
        except Exception as e:
            logger.warning(f"  Feature error on record {i}: {e} — using 0.5 fallback")
            rec["fcurv"] = 0.5
            rec["frank"] = 0.5
        rec["g"] = 0.5   # placeholder; updated after classifier is trained
        enriched.append(rec)

        # Periodic GPU cache clear to avoid OOM on large datasets
        if (i + 1) % 500 == 0:
            clear_gpu()
            log_gpu_memory(f"after {i+1} samples")

    save_jsonl(enriched, cache_path)
    logger.info(f"Features cached → {cache_path}")
    return enriched


def update_classifier_scores(
    records: list,
    clf_feat: ClassifierFeature,
    batch_size: int = 64,
) -> list:
    """
    Replace the g=0.5 placeholder in each record with the trained
    classifier score. Batched for GPU efficiency.
    """
    texts = [r["text"] for r in records]
    scores = clf_feat.compute_batch(texts, batch_size=batch_size)
    updated = []
    for rec, g in zip(records, scores):
        rec = rec.copy()
        rec["g"] = g
        updated.append(rec)
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg    = load_config()
    seed   = cfg["project"]["seed"]
    set_seed(seed)

    device = resolve_device(cfg["project"]["device"])
    fp16   = cfg["project"].get("fp16", True)
    l1_cfg = cfg["layer1"]
    clf_cfg = l1_cfg["classifier"]

    artifact_dir = Path("artifacts/layer1")
    cache_dir    = Path("data/processed/layer1_features")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pool_dir = Path(cfg["data"]["distillation_pool_dir"])

    logger.info("=" * 60)
    logger.info(f"LLM-Guard — Layer-1 Training")
    logger.info(f"  Device : {device}")
    logger.info(f"  FP16   : {fp16 and device == 'cuda'}")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading distillation pool...")
    train_records = load_jsonl(str(pool_dir / "train.jsonl"))
    val_records   = load_jsonl(str(pool_dir / "val.jsonl"))
    test_records  = load_jsonl(str(pool_dir / "test.jsonl"))
    logger.info(
        f"  Train={len(train_records)}  Val={len(val_records)}  Test={len(test_records)}"
    )

    # ── 2. Load scoring LM (shared between curvature + log-rank) ─────────────
    logger.info("Loading scoring LM (distilgpt2)...")
    curv_feat = CurvatureFeature(
        model_name="distilgpt2",
        num_perturbations=l1_cfg["curvature"]["num_perturbations"],
        temperature_range=tuple(l1_cfg["curvature"]["temperature_range"]),
        device=device,
        fp16=fp16,
    )
    rank_feat = LogRankFeature(
        shared_model=curv_feat.model,
        shared_tokenizer=curv_feat.tokenizer,
    )
    log_gpu_memory("after scoring LM load")

    # ── 3. Compute curvature + log-rank features (cached) ────────────────────
    train_enriched = compute_features(
        train_records, curv_feat, rank_feat,
        cache_path=str(cache_dir / "train_features.jsonl"),
    )
    val_enriched = compute_features(
        val_records, curv_feat, rank_feat,
        cache_path=str(cache_dir / "val_features.jsonl"),
    )
    test_enriched = compute_features(
        test_records, curv_feat, rank_feat,
        cache_path=str(cache_dir / "test_features.jsonl"),
    )

    # ── 4. Fit initial logistic fusion (g=0.5 placeholder) ───────────────────
    logger.info("Fitting initial logistic fusion (g=placeholder)...")
    fusion = LogisticFusion()
    train_feats  = np.array([[r["fcurv"], r["frank"], r["g"]] for r in train_enriched])
    train_labels = np.array([r["label"] for r in train_enriched])
    fusion.fit(train_feats, train_labels)
    fusion.save(str(artifact_dir / "fusion_v1.joblib"))

    # ── 5. Train RoBERTa+LoRA Classifier ─────────────────────────────────────
    # Free the scoring LM from GPU before loading the classifier
    del curv_feat.model
    clear_gpu()
    log_gpu_memory("after scoring LM freed")

    logger.info("Building RoBERTa+LoRA classifier...")
    model, tokenizer = build_lora_classifier(
        backbone=clf_cfg["backbone"],
        lora_r=clf_cfg["lora_r"],
        lora_alpha=clf_cfg["lora_alpha"],
        lora_dropout=clf_cfg["lora_dropout"],
        device=device,
        fp16=fp16,
    )
    log_gpu_memory("after classifier model load")

    trainer = ClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        backbone=clf_cfg["backbone"],
        device=device,
        fp16=fp16,
        lr=clf_cfg["learning_rate"],
        batch_size=clf_cfg["batch_size"],
        num_workers=clf_cfg.get("num_workers", 0),
        pin_memory=clf_cfg.get("pin_memory", False),
        max_epochs=clf_cfg["epochs"],
        patience=clf_cfg["early_stopping_patience"],
        max_length=clf_cfg["max_length"],
        output_dir=str(artifact_dir / "classifier"),
    )

    logger.info("Training classifier...")
    history = trainer.train(train_records, val_records)
    trainer.load_best()
    log_gpu_memory("after classifier training")

    # ── 6. Re-compute g(x) with trained classifier ───────────────────────────
    logger.info("Re-computing g(x) with trained classifier...")
    clf_feature = ClassifierFeature(
        checkpoint_path=str(artifact_dir / "classifier" / "checkpoint_best"),
        backbone=clf_cfg["backbone"],
        max_length=clf_cfg["max_length"],
        device=device,
        fp16=fp16,
    )

    bs = clf_cfg["batch_size"]
    train_final = update_classifier_scores(train_enriched, clf_feature, batch_size=bs)
    val_final   = update_classifier_scores(val_enriched,   clf_feature, batch_size=bs)
    test_final  = update_classifier_scores(test_enriched,  clf_feature, batch_size=bs)

    # ── 7. Re-fit fusion with real g(x) scores ───────────────────────────────
    logger.info("Re-fitting logistic fusion with trained classifier scores...")
    train_feats_v2 = np.array([[r["fcurv"], r["frank"], r["g"]] for r in train_final])
    fusion.fit(train_feats_v2, train_labels)
    fusion.save(str(artifact_dir / "fusion_v2.joblib"))

    # ── 8. Calibrate temperature scaler on validation set ────────────────────
    logger.info("Calibrating temperature scaler...")
    val_raw_scores = [
        fusion.raw_score(r["fcurv"], r["frank"], r["g"]) for r in val_final
    ]
    val_labels = [r["label"] for r in val_final]
    scaler = TemperatureScaler()
    T_star = scaler.calibrate(val_raw_scores, val_labels)
    scaler.save(str(artifact_dir / "scaler.json"))

    # ── 9. Evaluate on test set ───────────────────────────────────────────────
    logger.info("Evaluating on test set...")

    # Reload scoring LM for evaluation (needed by detector.detect())
    curv_feat_eval = CurvatureFeature(
        model_name="distilgpt2",
        num_perturbations=l1_cfg["curvature"]["num_perturbations"],
        temperature_range=tuple(l1_cfg["curvature"]["temperature_range"]),
        device=device,
        fp16=fp16,
    )
    rank_feat_eval = LogRankFeature(
        shared_model=curv_feat_eval.model,
        shared_tokenizer=curv_feat_eval.tokenizer,
    )

    router = ThresholdRouter(
        tau_low=l1_cfg["thresholds"]["low"],
        tau_high=l1_cfg["thresholds"]["high"],
    )
    detector = Layer1Detector(
        curvature_feature=curv_feat_eval,
        logrank_feature=rank_feat_eval,
        classifier_feature=clf_feature,
        fusion=fusion,
        scaler=scaler,
        router=router,
    )

    test_metrics = detector.evaluate(test_records)
    clear_gpu()

    # ── 10. Save results ──────────────────────────────────────────────────────
    results = {
        "test_metrics": test_metrics,
        "temperature_T": T_star,
        "training_history": history,
        "config_snapshot": {
            "device": device,
            "fp16": fp16 and device == "cuda",
            "thresholds": l1_cfg["thresholds"],
            "classifier_backbone": clf_cfg["backbone"],
            "lora_r": clf_cfg["lora_r"],
            "batch_size": clf_cfg["batch_size"],
        },
    }
    with open(artifact_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Layer-1 Training Complete")
    logger.info(f"  AUROC       : {test_metrics.get('auroc', 'N/A')}")
    logger.info(f"  AUPRC       : {test_metrics.get('auprc', 'N/A')}")
    logger.info(f"  FPR@95%TPR  : {test_metrics.get('fpr@95tpr', 'N/A')}")
    logger.info(f"  Accuracy    : {test_metrics.get('accuracy', 'N/A')}")
    logger.info(f"  T*          : {T_star:.4f}")
    logger.info(f"  Artifacts   → {artifact_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
