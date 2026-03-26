# vairagya_decay.py — affectively-gated heterosynaptic decay
# V[i][j] per-synapse Vairagya score. High score = protected from decay.
#
# Paper 4 addition: Viparita Buddhi erosion.
# When Buddhi is low (fear high, experience low), Vairagya scores erode.
# Erosion gates to zero as Buddhi rises — wisdom stabilises with experience.
# This prevents the score from becoming a one-way ratchet, which caused
# fc_out to hit 100% protection and block CIL gradient flow entirely.

import torch
from maya_cl.utils.config import (
    VAIRAGYA_DECAY_RATE,
    VAIRAGYA_PROTECTION_THRESHOLD,
    VAIRAGYA_ACCUMULATE_RATE,
    VAIRAGYA_PAIN_EROSION_RATE,
)


class VairagyadDecay:

    def __init__(self, shape: tuple, device: torch.device):
        self.scores = torch.zeros(shape, device=device)
        self.device = device

    def accumulate(self, active_mask: torch.Tensor,
                   pain_mask: torch.Tensor,
                   bhaya: float = 0.0,
                   buddhi: float = 1.0) -> None:
        with torch.no_grad():
            self.scores[active_mask] += VAIRAGYA_ACCUMULATE_RATE * buddhi
            self.scores[pain_mask] += VAIRAGYA_ACCUMULATE_RATE * 5.0 * buddhi

            # Viparita Buddhi erosion — only meaningful when Buddhi is collapsed.
            # viparita_multiplier approaches 0 as Buddhi → 1.
            viparita = bhaya * (1.0 - buddhi)
            if viparita > 0.01:
                self.scores -= viparita * VAIRAGYA_PAIN_EROSION_RATE

            self.scores.clamp_(0.0, 1.0)

    def apply_decay(self, weight: torch.Tensor) -> None:
        with torch.no_grad():
            unprotected = self.scores < VAIRAGYA_PROTECTION_THRESHOLD
            weight[unprotected] *= (1.0 - VAIRAGYA_DECAY_RATE)

    def get_scores(self) -> torch.Tensor:
        return self.scores

    def protection_fraction(self) -> float:
        return (self.scores >= VAIRAGYA_PROTECTION_THRESHOLD).float().mean().item()