"""XFR residual classifiers used by the released WiSig and LTE-V fits.

Inputs are float32 elementary XFR rows shaped ``[N, 2, m]``. The ``core``
module names and construction order retain the historical checkpoint layout.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


class WiSigResidualUnit(nn.Module):
    def __init__(self, in_width: int, out_width: int, kernel: int):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv1d(in_width, out_width, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_width), nn.ReLU(),
            nn.Conv1d(out_width, out_width, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_width),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_width, out_width, 1, bias=False), nn.BatchNorm1d(out_width),
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.path(x) + self.shortcut(x))


class WiSigCore(nn.Module):
    def __init__(self, m: int, n_classes: int):
        super().__init__()
        if m < 16 or m % 16:
            raise ValueError("WiSig m must be a positive multiple of 16")
        widths = [2, 32, 64, 128, 256]
        self.stages = nn.ModuleList([
            WiSigResidualUnit(widths[i], widths[i + 1], kernel)
            for i, kernel in enumerate([7, 5, 5, 3])
        ])
        self.pool = nn.MaxPool1d(2)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * (m // 16), 512), nn.ReLU(),
            nn.Dropout(.3), nn.Linear(512, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = self.pool(stage(x))
        return self.head(x)


class LTEResidualUnit(nn.Module):
    def __init__(self, in_width: int, out_width: int, stride: int):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv1d(in_width, out_width, 3, stride, padding=1, bias=False),
            nn.BatchNorm1d(out_width), nn.ReLU(), nn.Dropout(.5),
            nn.Conv1d(out_width, out_width, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_width),
        )
        self.shortcut = nn.Identity() if in_width == out_width and stride == 1 else nn.Sequential(
            nn.Conv1d(in_width, out_width, 1, stride, bias=False), nn.BatchNorm1d(out_width),
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.path(x) + self.shortcut(x))


class LTECore(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(3, 2, 1),
        )
        self.levels = nn.ModuleList()
        previous_width = 64
        for i, width in enumerate([64, 128, 256, 512]):
            self.levels.append(nn.Sequential(
                LTEResidualUnit(previous_width, width, 1 if i == 0 else 2),
                LTEResidualUnit(width, width, 1),
            ))
            previous_width = width
        self.output = nn.Linear(512, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for level in self.levels:
            x = level(x)
        return self.output(x.mean(-1))


class XFRClassifier(nn.Module):
    """An elementary-row classifier; block fusion is performed by evaluation."""
    log_probability_output = False

    def __init__(self, dataset: str, m: int, n_classes: int):
        super().__init__()
        if dataset not in {"wisig", "ltev"}:
            raise ValueError("dataset must be 'wisig' or 'ltev'")
        if m < 1 or n_classes < 2:
            raise ValueError("m must be positive and n_classes must be at least two")
        self.dataset = dataset
        self.input_length = m
        self.n_classes = n_classes
        self.core = WiSigCore(m, n_classes) if dataset == "wisig" else LTECore(n_classes)

    def forward(self, rows: Tensor) -> Tensor:
        if rows.ndim != 3 or rows.shape[1:] != (2, self.input_length):
            raise ValueError(f"Expected [N, 2, {self.input_length}] elementary XFR rows")
        return self.core(rows)


def create_model(config: Mapping) -> XFRClassifier:
    """Construct the historical backbone without loading pretrained weights.

    ``dataset`` is required. Optional ``m`` and ``n_classes`` default to
    ``240, 6`` for WiSig and ``256, 9`` for LTE-V, respectively.
    """
    dataset = config["dataset"]
    if dataset not in {"wisig", "ltev"}:
        raise ValueError("dataset must be 'wisig' or 'ltev'")
    m = config.get("m", 240 if dataset == "wisig" else 256)
    n_classes = config.get("n_classes", 6 if dataset == "wisig" else 9)
    return XFRClassifier(dataset, int(m), int(n_classes))


def load_weights(model: nn.Module, path: str | Path) -> nn.Module:
    """Strictly load a tensor state dictionary and return ``model``.

    Released weights are plain state dictionaries. Safe training dictionaries
    with a ``model`` or ``state_dict`` field are accepted as well. Loading is
    always CPU-mapped and uses ``weights_only=True``; there is no pickle fallback.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(state, Mapping):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise ValueError("Expected a nonempty tensor state dictionary")
    model.load_state_dict(state, strict=True)
    return model


__all__ = ["XFRClassifier", "create_model", "load_weights"]
