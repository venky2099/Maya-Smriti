# affective_state.py — five-dimensional affective state tracker
# Shraddha / Bhaya / Vairagya / Spanda from Papers 1-3.
# Buddhi introduced in Paper 4 — discriminative intellect,
# modelled after the Vedantic concept of Viparita Buddhi:
# intellect corrupts under fear and inexperience, recovers through observation.

import math
import torch
from maya_cl.utils.config import (
    TAU_SHRADDHA, TAU_BHAYA, TAU_VAIRAGYA, TAU_SPANDA, TAU_BUDDHI
)


class AffectiveState:

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.shraddha = torch.tensor(0.5, device=self.device)
        self.bhaya    = torch.tensor(0.0, device=self.device)
        self.vairagya = torch.tensor(0.5, device=self.device)
        self.spanda   = torch.tensor(0.0, device=self.device)
        self.buddhi   = torch.tensor(0.0, device=self.device)
        self._batch_count = 0

    def reset_experience(self):
        # Called at every task boundary.
        # Experience resets to zero — Viparita Buddhi begins.
        self._batch_count = 0
        self.buddhi = torch.tensor(0.0, device=self.device)

    def update(self, mean_confidence: float, pain_fired: bool,
               mean_spike_rate: float):
        dt = 1.0

        self.shraddha += dt * (mean_confidence - self.shraddha) / TAU_SHRADDHA
        self.shraddha.clamp_(0.0, 1.0)

        if pain_fired:
            self.bhaya = torch.tensor(1.0, device=self.device)
        else:
            self.bhaya += dt * (0.0 - self.bhaya) / TAU_BHAYA
        self.bhaya.clamp_(0.0, 1.0)

        self.vairagya += dt * (self.shraddha - self.vairagya) / TAU_VAIRAGYA
        self.vairagya.clamp_(0.0, 1.0)

        self.spanda += dt * (mean_spike_rate - self.spanda) / TAU_SPANDA
        self.spanda.clamp_(0.0, 1.0)

        self._update_buddhi()

    def _update_buddhi(self):
        self._batch_count += 1
        experience = 1.0 - math.exp(-self._batch_count / TAU_BUDDHI)
        buddhi_val = experience * (1.0 - self.bhaya.item())
        self.buddhi = torch.tensor(buddhi_val, device=self.device).clamp(0.0, 1.0)

    def as_dict(self) -> dict:
        return {
            "shraddha": self.shraddha.item(),
            "bhaya":    self.bhaya.item(),
            "vairagya": self.vairagya.item(),
            "spanda":   self.spanda.item(),
            "buddhi":   self.buddhi.item(),
        }