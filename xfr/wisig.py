"""WiSig multi-frame XFR-CD at -12 dB (Table VII).

All physical identities and observations are reconstructed from ManySig.pkl.
No experiment cache, private module, saved prediction, or local absolute path
is needed. The bundled fold manifest contains block IDs only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .representation import cross_frame_rows
from .schedules import cd_blocks
from .wisig_data import data_file, load_dataset, verify_source


TX_NAMES = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
RX_NAMES = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1", "2-19",
            "20-1", "3-19", "7-14", "7-7", "8-8")
M, L, N_ROLE_REFS, N_ROLE_BLOCKS = 240, 256, 72000, 300
_FOLD_PATH = Path(__file__).resolve().parents[1] / "configs" / "wisig" / "folds.json"
_FIXED = {"dataset": "wisig", "m": M, "L": L, "seed": 42, "chunk": 5,
          "snr_db": -12, "source_date": "2021_03_15", "target_date": "2021_03_01"}


@dataclass
class Blocks:
    """Clean physical blocks; reference IDs index date/TX/RX/packet order."""

    observations: np.ndarray  # float32 [600,240,256,2]
    labels: np.ndarray  # int64 [600]
    reference_ids: np.ndarray  # int32 [600,240]
    block_ids: np.ndarray  # int64 [600]
    folds: list[dict]


@dataclass
class Prepared:
    """Reusable normalized rows, shared by all five source-fold models."""

    inputs: np.ndarray  # float32 [600,256,2,240]
    labels: np.ndarray  # int64 [600], one label per physical block
    reference_ids: np.ndarray  # int32 [600,240]
    block_ids: np.ndarray  # int64 [600]
    folds: list[dict]


def _validate_config(config: Mapping) -> None:
    for key, value in _FIXED.items():
        if config.get(key) != value:
            raise ValueError(f"WiSig Table VII requires {key}={value!r}")
    if config.get("n_classes", 6) != 6:
        raise ValueError("WiSig Table VII requires six transmitters")


def _data_file(data_root: str | Path) -> Path:
    return data_file(data_root)


def _verify_source(path: Path, config: Mapping) -> None:
    verify_source(path, config.get("data", {}).get("source_sha256"))


def _folds() -> list[dict]:
    # Freeze the original StratifiedKFold(5,shuffle=True,random_state=42)
    # output, so sklearn and future splitter implementations are unnecessary.
    folds = json.loads(_FOLD_PATH.read_text(encoding="utf-8"))
    all_validation = []
    for i, fold in enumerate(folds, 1):
        if fold["fold"] != i or len(fold["train"]) != 240 or len(fold["val"]) != 60:
            raise ValueError("Invalid WiSig fold manifest")
        if sorted(fold["train"] + fold["val"]) != list(range(300)):
            raise ValueError("WiSig source folds must partition whole source blocks")
        if fold["test"] != list(range(300, 600)):
            raise ValueError("WiSig folds must share the ordered target blocks 300..599")
        all_validation.extend(fold["val"])
    if len(folds) != 5 or sorted(all_validation) != list(range(300)):
        raise ValueError("WiSig validation folds must cover each source block once")
    return folds


def _identities() -> tuple[np.ndarray, np.ndarray]:
    """Continue one RandomState stream across source and target RX shuffles."""
    rng = np.random.RandomState(42)
    refs = []
    labels = []
    for role in range(2):
        for label in range(6):
            pools = np.stack([
                role * N_ROLE_REFS + label * 12000 + rx * 1000
                + rng.permutation(1000).astype(np.int32)
                for rx in range(12)
            ])
            # RX0 five packets, RX1 five, ..., RX11 five; repeat four
            # cycles per block and retain the continuing receiver pools.
            refs.append(cd_blocks(pools, M, "short_runs", chunk_size=5))
            labels.extend([label] * 50)
    return np.concatenate(refs).astype(np.int32), np.asarray(labels, dtype=np.int64)


def load_blocks(data_path: str | Path, config: Mapping) -> Blocks:
    """Load raw, non-equalized I/Q and form canonical [B,m,L,IQ] blocks.

    Global packet order is source date, then target date; within each date,
    TX-list order, RX-list order, and the original 1000-packet record order.
    Global block IDs 0..299 are source, 300..599 are target. This function
    performs no amplitude normalization, added noise, or XFR transformation.
    """
    _validate_config(config)
    book = load_dataset(data_path, config["data"]["source_sha256"])
    if tuple(map(str, book["tx_list"])) != TX_NAMES:
        raise ValueError("ManySig transmitter list differs from the released six-class profile")
    if tuple(map(str, book["rx_list"])) != RX_NAMES:
        raise ValueError("ManySig receiver list differs from the released twelve-receiver profile")
    equalized = book["equalized_list"].index(0)
    raw = np.empty((2 * N_ROLE_REFS, L, 2), dtype=np.float32)
    offset = 0
    for date in (config["source_date"], config["target_date"]):
        date_index = book["capture_date_list"].index(date)
        for tx in range(6):
            for rx in range(12):
                record = np.asarray(book["data"][tx][rx][date_index][equalized], dtype=np.float32)
                if record.shape != (1000, L, 2) or not np.isfinite(record).all():
                    raise ValueError("Expected finite ManySig records with shape [1000,256,2]")
                raw[offset:offset + 1000] = record
                offset += 1000
    del book, record
    refs, labels = _identities()
    return Blocks(raw[refs], labels, refs, np.arange(600, dtype=np.int64), _folds())


def _noise_store() -> np.ndarray:
    """Draw float64 Gaussians in original global-packet/IQ/position order.

    Source draws must be consumed before target draws. Batching by 1024
    packets preserves the original randn(144000,2,256) MT19937 stream.
    """
    rng = np.random.RandomState(42)
    noise = np.empty((2 * N_ROLE_REFS, L, 2), dtype=np.float64)
    for start in range(0, len(noise), 1024):
        size = min(1024, len(noise) - start)
        noise[start:start + size] = rng.randn(size, 2, L).transpose(0, 2, 1)
    return noise


def _condition(blocks: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Original XFR row-RMS/AWGN arithmetic in physical block coordinates.

    RMS and normalized-row power are computed in complex64. Gaussian noise
    stays float64 until the final physical observation is cast to float32.
    The RMS factor restores physical scale before post-XFR I/Q z-scoring.
    """
    z = (blocks[..., 0] + 1j * blocks[..., 1]).astype(np.complex64)
    rms = np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True)) + 1e-12
    normalized = z / rms
    power = np.mean(np.abs(normalized) ** 2, axis=1, keepdims=True)
    noise_scale = rms * np.sqrt(power / (10 ** (snr_db / 10)) / 2)
    z = z + noise_scale * (noise[..., 0] + 1j * noise[..., 1])
    return np.stack((z.real, z.imag), axis=-1).astype(np.float32)


def prepare(data_root: str | Path, config: Mapping) -> Prepared:
    """Construct observations and normalized XFR rows once for all folds.

    Returns approximately 295 MB of float32 input data. Intermediate raw I/Q
    and Gaussian arrays are released when preparation returns; nothing is
    written to disk. See :func:`get_split` for the flattened model contract.
    """
    blocks = load_blocks(data_root, config)
    noise = _noise_store()
    inputs = np.empty((600, L, 2, M), dtype=np.float32)
    for start in range(0, 600, 5):
        end = start + 5
        physical = _condition(blocks.observations[start:end], noise[blocks.reference_ids[start:end]],
                              config["snr_db"])
        rows = torch.from_numpy(cross_frame_rows(physical))
        mean = rows.mean(-2, keepdim=True)
        std = rows.std(-2, keepdim=True, unbiased=False)
        rows = (rows - mean) / torch.where(std < 1e-8, torch.ones_like(std), std)
        inputs[start:end] = rows.permute(0, 1, 3, 2).numpy()
    return Prepared(inputs, blocks.labels, blocks.reference_ids, blocks.block_ids, blocks.folds)


def get_split(prepared: Prepared, config: Mapping, fold: int, split: str
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return float32 [N,2,240] rows, int64 labels, and global block IDs.

    Rows preserve the frozen split's block order and positions 0..255 inside
    every block. Each label and actual global block ID is repeated 256 times.
    ``test`` is the same target date for all folds; ``train`` and ``val`` are
    disjoint sets of complete source blocks. Fold numbers are one-based.
    """
    _validate_config(config)
    if isinstance(fold, bool) or fold not in range(1, 6) or split not in {"train", "val", "test"}:
        raise ValueError("Use fold 1..5 and split train/val/test")
    ids = np.asarray(prepared.folds[fold - 1][split], dtype=np.int64)
    inputs = np.ascontiguousarray(prepared.inputs[ids].reshape(-1, 2, M))
    return inputs, np.repeat(prepared.labels[ids], L), np.repeat(prepared.block_ids[ids], L)


def build_dataset(data_root: str | Path, config: Mapping, fold: int, split: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper; use prepare/get_split to reuse I/Q across folds."""
    return get_split(prepare(data_root, config), config, fold, split)


__all__ = ["Blocks", "Prepared", "load_blocks", "prepare", "get_split", "build_dataset"]
