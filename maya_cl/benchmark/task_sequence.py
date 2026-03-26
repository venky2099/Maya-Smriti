# task_sequence.py — task state tracking and pain signal computation

import torch
from maya_cl.utils.config import NUM_TASKS, PAIN_CONFIDENCE_THRESHOLD


class TaskSequencer:

    def __init__(self):
        self.current_task = 0
        self.confidence_history = []
        self.pain_fired = False

    def update_confidence(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits.detach(), dim=1)
        mean_conf = probs.max(dim=1).values.mean().item()
        self.confidence_history.append(mean_conf)
        if len(self.confidence_history) > 20:
            self.confidence_history.pop(0)
        return mean_conf

    def check_pain_signal(self, cur_loss: float, prev_loss: float,
                          current_conf: float,
                          replay_conf: float = None) -> bool:
        # Condition 1 — loss spike (works in Paper 3, less reliable with replay)
        if prev_loss is not None:
            if (cur_loss / (prev_loss + 1e-8)) > 1.5:
                self.pain_fired = True
                return True

        # Condition 2 — replay confidence collapse
        # When old class samples produce low confidence, the network
        # is failing to retain prior knowledge. This is the CIL pain signal.
        if replay_conf is not None and replay_conf < 0.30:
            self.pain_fired = True
            return True

        self.pain_fired = False
        return False

    def next_task(self):
        assert self.current_task < NUM_TASKS - 1
        self.current_task += 1
        self.confidence_history.clear()