import logging
import json
import time
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


class AuditLogger:
    """
    Logs every L1/L2/L3 decision as a JSON line for auditability.
    Aligns with NIST AI RMF artifact logging requirements.
    """

    def __init__(self, log_path: str = "logs/audit.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_layer1(self, text: str, scores: dict, decision: str) -> None:
        self.log({
            "layer": 1,
            "text_preview": text[:100],
            "scores": scores,
            "decision": decision,
        })

    def log_layer2(self, text: str, judge_decision: str,
                   severity: float, rationale: str, confidence: float) -> None:
        self.log({
            "layer": 2,
            "text_preview": text[:100],
            "judge_decision": judge_decision,
            "severity": severity,
            "rationale": rationale,
            "confidence": confidence,
        })

    def log_layer3(self, text: str, mode: str, output: str,
                   constraints_applied: list) -> None:
        self.log({
            "layer": 3,
            "text_preview": text[:100],
            "mitigation_mode": mode,
            "output_preview": output[:100],
            "constraints_applied": constraints_applied,
        })
