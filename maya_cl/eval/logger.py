# logger.py — per-batch CSV and per-task JSON logging

import os
import csv
import json
from datetime import datetime
from maya_cl.utils.config import RESULTS_DIR


class RunLogger:

    def __init__(self, run_name: str):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id  = f"{run_name}_{timestamp}"

        self.csv_path = os.path.join(RESULTS_DIR, f"{self.run_id}_batches.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            "task", "epoch", "batch",
            "loss", "confidence", "pain_fired",
            "lability_mean", "vairagya_protection_fc1",
            "shraddha", "bhaya", "vairagya", "spanda", "buddhi",
        ])
        self.csv_writer.writeheader()

        self.json_path = os.path.join(RESULTS_DIR, f"{self.run_id}_summary.json")
        self.summary   = {"run_id": self.run_id, "tasks": [], "final_metrics": {}}

    def log_batch(self, task: int, epoch: int, batch: int,
                  loss: float, confidence: float, pain_fired: bool,
                  lability_mean: float, vairagya_protection: float,
                  affective: dict) -> None:
        self.csv_writer.writerow({
            "task":                   task,
            "epoch":                  epoch,
            "batch":                  batch,
            "loss":                   round(loss, 6),
            "confidence":             round(confidence, 4),
            "pain_fired":             int(pain_fired),
            "lability_mean":          round(lability_mean, 4),
            "vairagya_protection_fc1": round(vairagya_protection, 4),
            "shraddha": round(affective["shraddha"], 4),
            "bhaya":    round(affective["bhaya"],    4),
            "vairagya": round(affective["vairagya"], 4),
            "spanda":   round(affective["spanda"],   4),
            "buddhi":   round(affective.get("buddhi", 0.0), 4),
        })
        self.csv_file.flush()

    def log_task_summary(self, task: int, accuracy_dict: dict,
                         cl_metrics: dict) -> None:
        self.summary["tasks"].append({
            "task": task,
            "per_task_accuracy": accuracy_dict,
            "cl_metrics_so_far": cl_metrics,
        })
        self._save_json()

    def log_final(self, cl_metrics: dict) -> None:
        self.summary["final_metrics"] = cl_metrics
        self._save_json()
        print(f"\nResults saved to:\n  {self.csv_path}\n  {self.json_path}")

    def _save_json(self) -> None:
        with open(self.json_path, "w") as f:
            json.dump(self.summary, f, indent=2)

    def close(self) -> None:
        self.csv_file.close()