"""Evaluate the WiSig date-pair and 20 dB XFR checkpoints."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn

from xfr.wisig_data import load_dataset as load_wisig_dataset

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TX = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
RX = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1", "2-19", "20-1", "3-19", "7-14", "7-7", "8-8")


class ResidualBlock(nn.Module):
    def __init__(self, width_in, width_out, kernel):
        super().__init__()
        self.conv1 = nn.Conv1d(width_in, width_out, kernel, padding=kernel // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(width_out)
        self.conv2 = nn.Conv1d(width_out, width_out, kernel, padding=kernel // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(width_out)
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(nn.Conv1d(width_in, width_out, 1, bias=False), nn.BatchNorm1d(width_out))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.downsample(x))


class XFRNet(nn.Module):
    """Four residual stages and the length-240 flatten/MLP classifier."""
    def __init__(self):
        super().__init__()
        self.layer1 = ResidualBlock(2, 32, 7)
        self.layer2 = ResidualBlock(32, 64, 5)
        self.layer3 = ResidualBlock(64, 128, 5)
        self.layer4 = ResidualBlock(128, 256, 3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 15, 512),
                                nn.ReLU(), nn.Dropout(.3), nn.Linear(512, 6))

    def forward(self, x):
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = self.pool(layer(x))
        return self.fc(x)


def _settings(args):
    names = {"wisig-date": "wisig_dates.json", "wisig-20db": "wisig_snr20.json"}
    if args.experiment not in names:
        raise ValueError("Expected wisig-date or wisig-20db")
    config = json.loads((CONFIGS / names[args.experiment]).read_text(encoding="utf-8"))
    if args.experiment == "wisig-date" and getattr(args, "all_pairs", False):
        return config, None
    source = getattr(args, "source_date", None) or "2021-03-15"
    target = getattr(args, "target_date", None) or "2021-03-01"
    selected = [run for run in config["runs"]
                if run["source_date"] == source and run["target_date"] == target]
    if not selected:
        raise ValueError(f"No {args.experiment} weights for {source} to {target}")
    return config, selected[0]


def _build_blocks(dataset, date, config, rng):
    """Assemble cyclic receiver groups into [block, position, IQ, frame]."""
    di = dataset["capture_date_list"].index(date.replace("-", "_"))
    ei = dataset["equalized_list"].index(0)
    size, chunk = config["m"], config["rx_chunk"]
    blocks, labels = [], []
    for tx in range(6):
        pools, indices = [], []
        for rx in range(12):
            raw = dataset["data"][tx][rx][di][ei]
            if raw.shape != (1000, 256, 2):
                raise ValueError("Expected 1000 length-256 references per transmitter, receiver, and date")
            pools.append(raw)
            order = np.arange(1000)
            if config["shuffle_rx_pools"]:
                rng.shuffle(order)
            indices.append(order)
        schedule = [(rx, int(packet)) for start in range(0, 1000, chunk)
                    for rx in range(12) for packet in indices[rx][start:start + chunk]]
        for start in range(0, len(schedule) - size + 1, size):
            block = np.stack([pools[rx][packet] for rx, packet in schedule[start:start + size]])
            blocks.append(block.transpose(1, 2, 0).astype(np.float32))
            labels.append(tx)
    return np.stack(blocks), np.asarray(labels, dtype=np.int64)


def _condition(blocks, config, rng):
    if config["conditioning"] == "iq_standardization":
        iq = np.ascontiguousarray(blocks.transpose(0, 1, 3, 2))
        mu, sigma = iq.mean(2, keepdims=True), iq.std(2, keepdims=True)
        sigma[sigma < 1e-8] = 1
        return np.ascontiguousarray(((iq - mu) / sigma).transpose(0, 1, 3, 2))
    result = np.empty_like(blocks)
    rotation = np.exp(2j * np.pi * config["fd_hz"] * np.arange(config["m"]) / config["fs_hz"])
    for index, block in enumerate(blocks):
        signal = block[:, 0] + 1j * block[:, 1]
        signal /= np.sqrt(np.mean(np.abs(signal) ** 2, axis=1, keepdims=True)) + 1e-12
        signal = signal * rotation
        power = np.mean(np.abs(signal) ** 2, axis=1, keepdims=True)
        # Each position draws its real samples followed by its imaginary samples.
        normals = rng.randn(256, 2, config["m"])
        noise = normals[:, 0] + 1j * normals[:, 1]
        signal = signal + np.sqrt(power / (2 * 10 ** (config["snr_db"] / 10))) * noise
        iq = np.stack([signal.real, signal.imag], axis=1).astype(np.float32)
        mu, sigma = iq.mean(-1, keepdims=True), iq.std(-1, keepdims=True)
        sigma[sigma < 1e-8] = 1
        result[index] = (iq - mu) / sigma
    return result


def _load_dataset(path, config):
    return load_wisig_dataset(path, config["data"]["source_sha256"])


def _prepare_pair(dataset, config, source_date, target_date):
    # Each date pair starts the same independent source-then-target RNG stream.
    rng = np.random.RandomState(config["seed"])
    source, _ = _build_blocks(dataset, source_date, config, rng)
    target, labels = _build_blocks(dataset, target_date, config, rng)
    # Source construction and noise draws precede target conditioning.
    _condition(source, config, rng)
    del source
    return _condition(target, config, rng), labels


def _prepare(path, config, source_date, target_date):
    return _prepare_pair(_load_dataset(path, config), config, source_date, target_date)


def _load_model(archive, entry, device):
    with zipfile.ZipFile(archive) as weights:
        payload = weights.read(entry["path"])
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {entry['path']}")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    model = XFRNet().to(device)
    model.load_state_dict(state, strict=True)
    return model


@torch.inference_mode()
def _predict(model, blocks, device):
    model.eval()
    rows = blocks.reshape(-1, 2, blocks.shape[-1])
    predictions = []
    for start in range(0, len(rows), 256):
        batch = torch.from_numpy(np.ascontiguousarray(rows[start:start + 256])).to(device)
        predictions.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def evaluate(args):
    """Read ManySig and a checkpoint ZIP, returning row accuracy for each fold."""
    config, run = _settings(args)
    if args.fold not in {"all", "1", "2", "3", "4", "5"}:
        raise ValueError("fold must be all or an integer from 1 through 5")
    if str(args.device) == "cpu":
        torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    all_pairs = run is None
    runs = config["runs"] if all_pairs else [run]
    dataset = _load_dataset(args.data, config) if all_pairs else None
    folds = range(1, 6) if args.fold == "all" else [int(args.fold)]
    results = []
    for run in runs:
        if all_pairs:
            blocks, labels = _prepare_pair(dataset, config, run["source_date"], run["target_date"])
        else:
            blocks, labels = _prepare(args.data, config, run["source_date"], run["target_date"])
        actual = np.repeat(labels, 256)
        for fold in folds:
            model = _load_model(args.weights, run["weights"][fold - 1], args.device)
            predictions = _predict(model, blocks, args.device)
            correct, total = int(np.count_nonzero(predictions == actual)), int(len(actual))
            row = dict(fold=fold, correct=correct, total=total,
                       accuracy_pct=100 * correct / total, unit="row")
            if all_pairs:
                row.update(source_date=run["source_date"], target_date=run["target_date"])
            results.append(row)
            del model
        del blocks, labels, actual
    return results
