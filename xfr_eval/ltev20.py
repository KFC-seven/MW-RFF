"""LTE-V 20 dB evaluation from registered MAT recordings and native weights.

The default ``legacy`` profile uses raw256 for SD and legacy288 for CD.
``paper256`` uses the separate IQ256 extraction for both constructions.
Each profile requires its matching data version; all use the same released
checkpoint states. Accuracy counts one XFR row, without block voting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

import h5py
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch
from torch import nn

CONFIGS = Path(__file__).resolve().parents[1] / "configs/ltev20"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _mat_text(dataset):
    values = np.asarray(dataset[()]).ravel()
    if values.dtype.kind in "SUO":
        return "".join(x.decode() if isinstance(x, bytes) else str(x) for x in values)
    return "".join(chr(int(x)) for x in values if x) if len(values) > 1 else str(int(values[0]))


def _read_record(path):
    with h5py.File(path, "r") as file:
        group = file["rfDataset"]
        data = group["dmrs"][:]
        if data.ndim != 2 or not data.dtype.names or not {"real", "imag"} <= set(data.dtype.names):
            raise ValueError("Expected compound-complex DMRS shaped [frames, positions]")
        values = np.ascontiguousarray(data["real"] + 1j * data["imag"], dtype="<c16")
        tx = re.sub(r"^TX", "", _mat_text(group["txID"]), flags=re.I).zfill(3)
        rx = re.sub(r"^RX", "", _mat_text(group["rxID"]), flags=re.I).zfill(3)
        speed = int(_mat_text(group["velocity"]))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite DMRS values")
    identity = dict(tx=tx, receiver=rx, speed_kmh=speed, n=len(values), L=values.shape[1],
                    complex128_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    return values, identity


def _build_membership(records, construction, m):
    references, labels = [], []
    transmitters = sorted({r["tx"] for r in records})
    if len(transmitters) != 9:
        raise ValueError("Expected nine transmitters")
    for label, tx in enumerate(transmitters):
        group = [r for r in records if r["tx"] == tx]
        if len(group) != 8:
            raise ValueError("Each transmitter requires eight receiver/speed recordings")
        if construction == "SD":
            for record in group:
                for block in range(record["n"] // m):
                    references.append(record["offset"] + np.arange(block * m, (block + 1) * m))
                    labels.append(label)
        elif construction == "CD":
            if m % len(group):
                raise ValueError("CD block length must be divisible by eight")
            take = m // len(group)
            for block in range(min(r["n"] for r in group) // take):
                references.append(np.concatenate([
                    r["offset"] + np.arange(block * take, (block + 1) * take) for r in group
                ]))
                labels.append(label)
        else:
            raise ValueError("Expected SD or CD")
    references = np.asarray(references, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(references)) != references.size:
        raise ValueError("Physical reference shared between blocks")
    return references, labels


def _split_blocks(labels, seed):
    train, test = train_test_split(np.arange(len(labels)), test_size=.25,
                                   stratify=labels, random_state=seed)
    folds = []
    for number, (tr, val) in enumerate(KFold(5, shuffle=True, random_state=seed).split(train), 1):
        split = dict(fold=number, train=train[tr].tolist(), val=train[val].tolist(), test=test.tolist())
        all_blocks = np.concatenate([split[k] for k in ("train", "val", "test")])
        if not np.array_equal(np.sort(all_blocks), np.arange(len(labels))):
            raise ValueError("Overlapping or missing split blocks")
        folds.append(split)
    return folds


def _condition(values, config, rng):
    values = values / (np.sqrt(np.mean(np.abs(values) ** 2, axis=1, keepdims=True)) + 1e-12)
    phase = np.exp(2j * np.pi * config["doppler_hz"] * np.arange(values.shape[1]) / config["fs_hz"])
    values = values * phase
    power = np.mean(np.abs(values) ** 2, axis=1, keepdims=True)
    noise = rng.randn(len(values), 2, values.shape[1])
    values = values + np.sqrt(power / 10 ** (config["snr_db"] / 10) / 2) * (noise[:, 0] + 1j * noise[:, 1])
    return np.stack([values.real, values.imag], axis=-1).astype(np.float32)


@dataclass
class _Prepared:
    references: np.ndarray
    block_refs: np.ndarray
    block_labels: np.ndarray
    folds: list
    config: dict


def _prepare(data_root, config):
    root = Path(data_root)
    registry = _read_json(CONFIGS / config["registry"])
    files = list(root.rglob("*.mat"))
    if len(files) != 72:
        raise ValueError("Expected precisely 72 MAT recordings for the selected data version")
    matched = {}
    for path in files:
        _, actual = _read_record(path)
        key = actual["tx"], actual["receiver"], actual["speed_kmh"]
        if key in matched:
            raise ValueError("Duplicate recording identity")
        matched[key] = path, actual
    records, offset = [], 0
    for expected in registry["records"]:
        key = expected["tx"], expected["receiver"], expected["speed_kmh"]
        if key not in matched:
            raise ValueError("Missing transmitter/receiver/speed recording")
        _, actual = matched[key]
        if actual != expected or actual["L"] != config["L"]:
            raise ValueError("MAT content differs from the selected data version; choose its matching profile")
        records.append(dict(expected, offset=offset))
        offset += expected["n"]
    references = np.empty((offset, config["L"], 2), dtype=np.float32)
    rng = np.random.RandomState(config["noise_seed"])
    for record in records:
        path, _ = matched[record["tx"], record["receiver"], record["speed_kmh"]]
        values, _ = _read_record(path)
        references[record["offset"]:record["offset"] + record["n"]] = _condition(values, config, rng)
    block_refs, labels = _build_membership(records, config["construction"], config["m"])
    return _Prepared(references, block_refs, labels, _split_blocks(labels, config["split_seed"]), config)


def _test_rows(data, fold):
    ids = np.asarray(data.folds[fold - 1]["test"], dtype=np.int64)
    positions, frames = data.config["L"], data.config["m"]
    rows = np.empty((len(ids) * positions, 2, frames), dtype=np.float32)
    for begin in range(0, len(ids), 8):
        group = ids[begin:begin + 8]
        # [block, frame, position, IQ] -> [block, position, IQ, frame].
        rows[begin * positions:(begin + len(group)) * positions] = (
            data.references[data.block_refs[group]].transpose(0, 2, 3, 1).reshape(-1, 2, frames)
        )
    return rows, np.repeat(data.block_labels[ids], positions)


def _read_test_bundle(path, config):
    """Condition the released raw test references with the original per-record RNG stream."""
    expected = config.get("test_bundle_sha256")
    if expected is None:
        raise ValueError("No portable test bundle is registered for this data profile; use its complete MAT directory.")
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(2 ** 20), b""):
            checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError("The test bundle checksum differs from the released file.")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as bundle:
            if set(bundle.files) != {"references", "reference_ids", "labels", "block_ids"}:
                raise ValueError("Expected references, reference_ids, labels, and block_ids")
            signals = bundle["references"]
            indices = bundle["reference_ids"].ravel()
            labels = bundle["labels"]
            block_ids = bundle["block_ids"]

    if (signals.dtype != np.complex128 or signals.shape != (len(labels), config["m"], config["L"])
            or labels.dtype != np.int64 or block_ids.dtype != np.int64
            or block_ids.shape != labels.shape or indices.dtype != np.int32
            or len(indices) != len(labels) * config["m"]):
        raise ValueError("Unexpected raw test-bundle array shapes or dtypes")

    length, m = config["L"], config["m"]
    signals = signals.reshape(-1, length)
    observations = np.empty((len(signals), length, 2), dtype=np.float32)
    offset = 0
    rng = np.random.RandomState(config["noise_seed"])
    for record in _read_json(CONFIGS / config["registry"])["records"]:
        count = record["n"]
        selected = (indices >= offset) & (indices < offset + count)
        positions = indices[selected] - offset
        # Conditioning is per reference. Preserve the original noise draws for
        # selected frames by also consuming the draws for omitted frames.
        source = np.zeros((count, length), dtype=signals.dtype)
        source[positions] = signals[selected]
        observations[selected] = _condition(source, config, rng)[positions]
        offset += count
    blocks = observations.reshape(len(labels), m, length, 2)
    rows = np.ascontiguousarray(blocks.transpose(0, 2, 3, 1).reshape(-1, 2, m))
    return rows, np.repeat(labels, length)


class _BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(.5)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, rows):
        identity = rows
        result = self.relu(self.bn1(self.conv1(rows)))
        result = self.bn2(self.conv2(self.dropout(result)))
        if self.downsample is not None:
            identity = self.downsample(rows)
        result += identity
        return self.relu(result)


class _ResNet18(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.input_length = input_length
        self.in_planes = 64
        self.conv1 = nn.Conv1d(2, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)
        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        self.layer4 = self._make_layer(512, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 9)

    def _make_layer(self, planes, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(nn.Conv1d(self.in_planes, planes, 1, stride, bias=False),
                                       nn.BatchNorm1d(planes))
        blocks = [_BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        blocks.append(_BasicBlock(planes, planes))
        return nn.Sequential(*blocks)

    def forward(self, rows):
        if rows.ndim != 3 or rows.shape[1:] != (2, self.input_length):
            raise ValueError(f"Expected [N, 2, {self.input_length}] XFR rows")
        result = self.maxpool(self.relu(self.bn1(self.conv1(rows))))
        result = self.layer4(self.layer3(self.layer2(self.layer1(result))))
        return self.fc(self.avgpool(result).squeeze(-1))


def _load_model(archive, checkpoint, input_length):
    content = archive.read(checkpoint["member"])
    if hashlib.sha256(content).hexdigest() != checkpoint["sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the released weight")
    state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("Expected a tensor state dictionary")
    model = _ResNet18(input_length)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.inference_mode()
def _score(model, rows, labels, device):
    model.to(device).eval()
    correct = 0
    for begin in range(0, len(labels), 256):
        logits = model(torch.from_numpy(rows[begin:begin + 256]).to(device))
        predictions = logits.argmax(1).cpu().numpy()
        correct += int(np.count_nonzero(predictions == labels[begin:begin + 256]))
    return correct


def evaluate(args) -> list[dict]:
    """Return per-fold row accuracy; the caller owns output/summary handling."""
    experiments = {"ltev-sd-20db": "sd", "ltev-cd-20db": "cd"}
    if args.experiment not in experiments:
        raise ValueError("Expected ltev-sd-20db or ltev-cd-20db")
    profile = getattr(args, "profile", "legacy")
    if profile not in {"legacy", "paper256"}:
        raise ValueError("LTE-V profile must be legacy or paper256")
    requested = str(args.fold)
    if requested not in {"all", "1", "2", "3", "4", "5"}:
        raise ValueError("fold must be all or 1..5")
    folds = list(range(1, 6)) if requested == "all" else [int(requested)]
    kind = experiments[args.experiment]
    config = _read_json(CONFIGS / f"{kind}_{profile}.json")
    weights = _read_json(CONFIGS / "weights.json")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu")
    with ZipFile(args.weights) as archive:
        expected = {c["member"] for branch in weights.values() for c in branch.values()}
        members = [m for m in archive.namelist() if not m.endswith("/")]
        if set(members) != expected or len(members) != len(expected):
            raise ValueError("Expected the LTE-V 20 dB ZIP containing ten released checkpoint files")
        path = Path(args.data)
        if path.is_file() and path.suffix.lower() == ".npz":
            rows, labels = _read_test_bundle(path, config)
        else:
            data = _prepare(path, config)
            if any(data.folds[fold - 1]["test"] != data.folds[folds[0] - 1]["test"] for fold in folds):
                raise ValueError("All model folds must use the same held-out test blocks")
            rows, labels = _test_rows(data, folds[0])
            del data
        results = []
        for fold in folds:
            model = _load_model(archive, weights[kind][str(fold)], config["m"])
            correct = _score(model, rows, labels, device)
            total = int(len(labels))
            results.append(dict(fold=fold, correct=correct, total=total,
                                accuracy_pct=float(100 * correct / total), unit="row"))
    return results
