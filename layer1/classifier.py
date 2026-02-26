"""
Layer-1 Distilled Classifier
==============================
RoBERTa-base + LoRA fine-tuned for binary safety classification.

  Label 1 → allow    (safe content)
  Label 0 → mitigate (harmful / policy-violating content)

Training uses labels distilled from the Layer-2 Judge, making this
classifier a compressed approximation of the Judge's decision surface.

GPU improvements vs. CPU version:
  - Mixed-precision training (torch.cuda.amp.GradScaler + autocast)
  - num_workers and pin_memory on DataLoader for fast GPU transfers
  - float16 model loading on CUDA
  - load_best() receives the backbone name as a parameter (no hardcode)
  - GradScaler is a no-op on CPU so the same code runs on both

References:
  [Manakul et al., 2023] SelfCheckGPT — distillation from larger models
  PEFT / LoRA: Hu et al., 2022
"""

from __future__ import annotations

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from pathlib import Path
from typing import List, Dict, Optional
from utils.logger import get_logger

logger = get_logger("layer1.classifier")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    """Wraps a list of {'text': str, 'label': 0/1} records."""

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        # Filter out records missing text or label
        self.records = [
            r for r in records
            if r.get("text") and r.get("label") is not None
        ]
        if len(self.records) < len(records):
            logger.warning(
                f"PromptDataset: dropped {len(records) - len(self.records)} "
                "records with missing text or label."
            )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        enc = self.tokenizer(
            str(rec["text"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(int(rec["label"]), dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_classifier(
    backbone: str = "roberta-base",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    num_labels: int = 2,
    device: str = "cpu",
    fp16: bool = True,
) -> tuple:
    """
    Returns (model, tokenizer) — RoBERTa-base with LoRA adapters.

    LoRA targets query + value projections, reducing trainable params
    from ~125M to ~1M — feasible on both CPU and GPU.

    On GPU: loads in float16 to reduce VRAM usage.
    On CPU: loads in float32.
    """
    from layer1.features import model_dtype

    logger.info(f"Building LoRA classifier: backbone={backbone}, device={device}")
    dtype = model_dtype(device, fp16)

    tokenizer = AutoTokenizer.from_pretrained(backbone)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=num_labels,
        torch_dtype=dtype,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],  # RoBERTa attention projections
        bias="none",
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierTrainer:
    """
    Trains the RoBERTa+LoRA classifier with:
      - Mixed-precision (AMP) on GPU
      - Early stopping on validation loss
      - Gradient clipping

    Loss: CrossEntropy( g(x), y )  where y ∈ {0, 1}
    """

    def __init__(
        self,
        model,
        tokenizer,
        backbone: str = "roberta-base",
        device: str = "cpu",
        fp16: bool = True,
        lr: float = 2e-4,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_epochs: int = 5,
        patience: int = 2,
        max_length: int = 512,
        output_dir: str = "artifacts/layer1_classifier",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.device = device
        self.fp16 = fp16
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DataLoader workers: on Windows with CUDA, set num_workers=0 if errors
        self._num_workers = num_workers if device == "cuda" else 0
        self._pin_memory = pin_memory and (device == "cuda")

        # GradScaler is a no-op on CPU (enabled=False)
        self._scaler = GradScaler(enabled=(device == "cuda" and fp16))

        logger.info(
            f"ClassifierTrainer: device={device}, fp16={fp16 and device=='cuda'}, "
            f"batch_size={batch_size}, num_workers={self._num_workers}"
        )

    def train(
        self,
        train_records: List[Dict],
        val_records: List[Dict],
    ) -> Dict:
        """
        Full training loop with mixed-precision and early stopping.
        Returns training history dict.
        """
        train_ds = PromptDataset(train_records, self.tokenizer, self.max_length)
        val_ds   = PromptDataset(val_records,   self.tokenizer, self.max_length)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )
        total_steps = len(train_loader) * self.max_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimiser,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps,
        )

        criterion = nn.CrossEntropyLoss()
        history: Dict = {"train_loss": [], "val_loss": [], "val_acc": []}

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            # ── Train ──────────────────────────────────────────────────────
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels         = batch["label"].to(self.device, non_blocking=True)

                optimiser.zero_grad(set_to_none=True)   # faster than zero_grad()

                with autocast(enabled=(self.device == "cuda" and self.fp16)):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = criterion(outputs.logits, labels)

                self._scaler.scale(loss).backward()
                self._scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self._scaler.step(optimiser)
                self._scaler.update()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # ── Validate ───────────────────────────────────────────────────
            val_loss, val_acc = self._evaluate(val_loader, criterion)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch}/{self.max_epochs} — "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}"
            )

            # ── Early stopping ─────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint("best")
                logger.info(f"  → New best (val_loss={val_loss:.4f}) saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best epoch={best_epoch}, val_loss={best_val_loss:.4f})"
                    )
                    break

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training complete. Best epoch: {best_epoch}")
        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels         = batch["label"].to(self.device, non_blocking=True)

            with autocast(enabled=(self.device == "cuda" and self.fp16)):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def _save_checkpoint(self, tag: str = "best") -> None:
        save_path = self.output_dir / f"checkpoint_{tag}"
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        # Save backbone name so load_best can reconstruct without hardcoding
        with open(save_path / "backbone.txt", "w") as f:
            f.write(self.backbone)
        logger.info(f"Checkpoint saved → {save_path}")

    def load_best(self) -> None:
        """Reload the best checkpoint into self.model (replaces the current model)."""
        from layer1.features import model_dtype

        best_path = self.output_dir / "checkpoint_best"
        if not best_path.exists():
            logger.warning("No best checkpoint found. Model unchanged.")
            return

        # Read backbone name saved during _save_checkpoint
        backbone_file = best_path / "backbone.txt"
        backbone = self.backbone
        if backbone_file.exists():
            backbone = backbone_file.read_text().strip()

        dtype = model_dtype(self.device, self.fp16)
        base = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=2, torch_dtype=dtype
        )
        self.model = PeftModel.from_pretrained(base, str(best_path)).to(self.device)
        self.model.eval()
        logger.info(f"Best checkpoint loaded ← {best_path} (backbone={backbone})")
