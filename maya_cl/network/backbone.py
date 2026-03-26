# backbone.py — full Maya-CL SNN architecture
# Conv(32) → MaxPool → Conv(64) → MaxPool → FC(2048) → FC(10)
# Single unified CIL output head — no task oracle
# FC1_SIZE=2048 provides sufficient capacity for 5 sequential tasks
# with ~40-60% Vairagya protection active

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron
from maya_cl.network.lif_layers import ConvLIFBlock, FCLIFBlock
from maya_cl.utils.config import (
    CONV1_CHANNELS, CONV2_CHANNELS, FC1_SIZE, NUM_CLASSES, T_STEPS
)


class MayaCLNet(nn.Module):
    """
    Fixed-size SNN backbone. No architectural expansion across tasks.
    Plasticity is handled externally by the plasticity modules.
    FC1_SIZE=2048 ensures sufficient free capacity even with
    Vairagya protection active on prior-task synapses.
    """

    def __init__(self):
        super().__init__()

        # ── Convolutional blocks ──────────────────────────────────
        self.conv1 = ConvLIFBlock(3, CONV1_CHANNELS)
        self.pool1 = layer.MaxPool2d(2, 2)          # 32x32 → 16x16

        self.conv2 = ConvLIFBlock(CONV1_CHANNELS, CONV2_CHANNELS)
        self.pool2 = layer.MaxPool2d(2, 2)          # 16x16 → 8x8

        # ── Fully connected blocks ────────────────────────────────
        self.flatten = layer.Flatten()
        fc1_input = CONV2_CHANNELS * 8 * 8          # 64 * 8 * 8 = 4096
        self.fc1 = FCLIFBlock(fc1_input, FC1_SIZE)  # 4096 → 2048

        # output layer — standard linear, no LIF
        self.fc_out = layer.Linear(FC1_SIZE, NUM_CLASSES, bias=True)

        # set step mode to multi-step
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [T, B, C, H, W]
        returns: [B, NUM_CLASSES]
        """
        out = self.conv1(x_seq)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc_out(out)          # [T, B, NUM_CLASSES]
        out = out.mean(dim=0)           # [B, NUM_CLASSES]
        return out

    def reset(self):
        # reset only LIF nodes — avoids infinite recursion from reset_net
        for m in self.modules():
            if isinstance(m, neuron.LIFNode):
                m.reset()