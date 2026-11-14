"""The Phase 1 network. Deliberately tiny and hardware-friendly:

  - two Conv1d layers (kernel 5, no padding, stride 1), ReLU, MaxPool(2)
  - one Linear layer, six outputs
  - no batch norm, no dropout, ReLU only, channel counts 8 and 16

Input standardisation (per-channel mean/std from the training set) is done
outside the network and stored next to the weights in weights.npz.
"""
import numpy as np
import torch
from torch import nn

IN_CHANNELS, WINDOW, NUM_CLASSES = 9, 128, 6
C1, C2, KERNEL = 8, 16, 5


class HarCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(IN_CHANNELS, C1, KERNEL)   # (9,128)  -> (8,124)
        self.conv2 = nn.Conv1d(C1, C2, KERNEL)            # (8,62)   -> (16,58)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        len1 = (WINDOW - KERNEL + 1) // 2                 # 62
        len2 = (len1 - KERNEL + 1) // 2                   # 29
        self.fc = nn.Linear(C2 * len2, NUM_CLASSES)       # 464 -> 6

    def forward(self, x):                                 # x: (N, 9, 128), standardised
        x = self.pool(self.relu(self.conv1(x)))           # (N, 8, 62)
        x = self.pool(self.relu(self.conv2(x)))           # (N, 16, 29)
        # Flatten is channel-major: index = channel * 29 + position.
        return self.fc(x.flatten(1))                      # (N, 6) logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_npz(path, model, mean, std):
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    arrays["input_mean"] = np.asarray(mean, dtype=np.float32)
    arrays["input_std"] = np.asarray(std, dtype=np.float32)
    np.savez(path, **arrays)


def load_npz(path):
    """Return (model in eval mode, input_mean, input_std) from a weights .npz."""
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    mean, std = arrays.pop("input_mean"), arrays.pop("input_std")
    model = HarCnn()
    model.load_state_dict({k: torch.from_numpy(v) for k, v in arrays.items()})
    model.eval()
    return model, mean, std


def standardise(x, mean, std):
    """x: (N, 9, 128) raw signals; mean/std: (9,) training-set statistics."""
    return (x - mean[None, :, None]) / std[None, :, None]
