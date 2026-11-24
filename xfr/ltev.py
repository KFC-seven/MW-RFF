"""LTE-V Table VIII observations, rebuilt from the 72 original DMRS MAT files.

The only numerical transform after RF conditioning is cross-frame axis exchange.
No old experiment module, derived cache, or machine-specific data path is used.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import h5py
import numpy as np

from .representation import cross_frame_rows
from .schedules import cd_blocks


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "ltev"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config):
    expected = {"dataset": "ltev", "m": 256, "L": 256, "snr_db": -6.0}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"This frozen LTE-V profile requires {key}={value!r}")
    condition = config.get("condition", {})
    expected_condition = {"snr_db": -6.0, "doppler_hz": 655.5555555555557,
                          "fs_hz": 5000000.0, "seed": 42}
    if condition != expected_condition:
        raise ValueError("LTE-V RMS/Doppler/AWGN condition differs from the frozen profile")


def _mat_text(dataset):
    value = dataset[()]
    if np.asarray(value).dtype.kind in "SUO":
        return "".join(x.decode("utf-8") if isinstance(x, bytes) else str(x)
                       for x in np.asarray(value).ravel()).strip()
    values = np.asarray(value).ravel()
    if len(values) > 1:
        return "".join(chr(int(c)) for c in values if int(c)).strip()
    return str(int(values[0]))


def _identifier(text, prefix):
    text = re.sub(rf"^{prefix}", "", text, flags=re.IGNORECASE)
    if not re.fullmatch(r"\d{1,3}", text):
        raise ValueError(f"Invalid LTE-V {prefix} metadata: {text!r}")
    return text.zfill(3)


def _read_mat(path):
    """Read v7.3 DMRS and identity, independent of the container filename."""
    with h5py.File(path, "r") as source:
        group = source["rfDataset"]
        value = group["dmrs"][:]
        if np.iscomplexobj(value):
            z = value.astype(np.complex64)
        elif value.dtype.names and {"real", "imag"} <= set(value.dtype.names):
            z = (value["real"] + 1j * value["imag"]).astype(np.complex64)
        else:
            raise ValueError(f"Expected complex DMRS in {path.name}")
        if z.shape[0] <= 2048 and z.shape[1] > z.shape[0]:
            z = z.T
        tx = _identifier(_mat_text(group["txID"]), "TX")
        receiver = _identifier(_mat_text(group["rxID"]), "RX")
        speed = int(_mat_text(group["velocity"]))
    if z.shape != (2999, 256):
        raise ValueError(f"Unexpected DMRS dimensions in {path.name}")
    if not np.isfinite(z).all():
        raise ValueError(f"Nonfinite DMRS in {path.name}")
    # Fix traversal/reduction order as well as values. A transposed HDF dataset
    # otherwise retains Fortran strides and changes float32 RMS summation.
    z = np.ascontiguousarray(z, dtype=np.complex64)
    digest = hashlib.sha256(np.ascontiguousarray(z, dtype="<c8").tobytes()).hexdigest()
    raw = np.stack([z.real, z.imag], axis=-1).astype(np.float32)
    return raw, (tx, receiver, speed), digest


def _record_identity(record):
    return record["tx"], record["receiver"], record["speed_kmh"]


def _read_record(path, record):
    raw, identity, digest = _read_mat(path)
    if identity != _record_identity(record) or digest != record["reference_complex64_sha256"]:
        raise ValueError(f"LTE-V reference content or TX/RX/speed differs: {path.name}")
    return raw


def _match_record_paths(root, records):
    candidates = sorted(root.rglob("*.mat"))
    if len(candidates) != len(records):
        raise ValueError(f"Expected exactly {len(records)} LTE-V MAT files, found {len(candidates)}")
    by_name = {}
    for path in candidates:
        by_name.setdefault(path.name, []).append(path)
    # Fast original-container path. A renamed/repacked dataset falls back to
    # reference-content identity; container bytes are never the sole identity.
    if all(len(by_name.get(r["filename"], [])) == 1 for r in records):
        ordered = [by_name[r["filename"]][0] for r in records]
        if all(p.stat().st_size == r["bytes"] and _sha256(p) == r["sha256"]
               for p, r in zip(ordered, records)):
            return ordered
    expected = {_record_identity(r): r for r in records}
    matched = {}
    for path in candidates:
        _, identity, digest = _read_mat(path)
        if identity not in expected:
            raise ValueError(f"Unknown LTE-V TX/RX/speed in {path.name}: {identity}")
        if identity in matched:
            raise ValueError(f"Ambiguous duplicate LTE-V TX/RX/speed: {identity}")
        if digest != expected[identity]["reference_complex64_sha256"]:
            raise ValueError(f"LTE-V reference content differs: {path.name}")
        matched[identity] = path
    if set(matched) != set(expected):
        raise ValueError("Incomplete LTE-V TX/RX/speed identity set")
    return [matched[_record_identity(r)] for r in records]


def condition_references(raw_iq, condition, rng):
    """Frozen arithmetic, including float32 RMS and float64 phase/noise.

    The caller must share one RandomState over files in registry order. Drawing
    per-file arrays consumes exactly the original full-reference RNG sequence.
    """
    x = np.asarray(raw_iq, dtype=np.float32)
    z = (x[:, :, 0] + 1j * x[:, :, 1]).astype(np.complex64)
    z = z / (np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True)) + 1e-12)
    phase = np.exp(1j * 2 * np.pi * condition["doppler_hz"]
                   * np.arange(z.shape[1], dtype=np.float64) / condition["fs_hz"])
    z = z * phase
    power = np.mean(np.abs(z) ** 2, axis=1, keepdims=True)
    noise = rng.randn(len(z), 2, z.shape[1])
    z = z + np.sqrt(power / (10 ** (condition["snr_db"] / 10)) / 2) * (
        noise[:, 0] + 1j * noise[:, 1])
    return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def build_blocks(records, n_classes=9):
    """Eight sorted RX/speed files contribute consecutive slices of 32 each."""
    references, labels, receivers = [], [], []
    for label in range(n_classes):
        group = sorted((r for r in records if r["label"] == label), key=lambda r: r["filename"])
        if len(group) != 8 or any(r["n"] != 2999 for r in group):
            raise ValueError("Each LTE-V transmitter requires eight 2999-reference files")
        pools = [r["offset"] + np.arange(r["n"], dtype=np.int32) for r in group]
        for members in cd_blocks(pools, 256, "domain_groups"):
            references.append(members)
            receivers.append(np.repeat([r["rx"] for r in group], 32))
            labels.append(label)
    return (np.asarray(references, dtype=np.int32), np.asarray(labels, dtype=np.int64),
            np.asarray(receivers, dtype=np.int16))


@dataclass
class PreparedLTEV:
    """Reusable conditioned reference store and physical identity metadata."""
    references: np.ndarray
    block_refs: np.ndarray
    block_labels: np.ndarray
    block_receivers: np.ndarray
    folds: list
    records: list
    condition: dict


def prepare(data_root, config):
    """Rebuild in memory using read-only MAT access; never writes to data_root.

    data_root can contain the 72 MAT files directly or in subdirectories. Every
    original-container identity or exact canonical complex64 content is checked
    against the portable registry. Preparation needs about 0.5 GB for the
    returned store plus a small per-file workspace.
    """
    _validate_config(config)
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LTE-V data directory does not exist: {root}")
    registry = _json(_CONFIG_DIR / "records.json")
    records = registry["records"]
    paths = _match_record_paths(root, records)
    references = np.empty((215928, 256, 2), dtype=np.float32)
    rng = np.random.RandomState(config["condition"]["seed"])
    for path, record in zip(paths, records):
        raw = _read_record(path, record)
        start, count = record["offset"], record["n"]
        references[start:start + count] = condition_references(raw, config["condition"], rng)
    refs, labels, rx = build_blocks(records)
    folds_path = _CONFIG_DIR / "folds.json"
    if _sha256(folds_path) != registry["folds_sha256"]:
        raise ValueError("Frozen LTE-V fold registry differs")
    folds = _json(folds_path)
    if len(folds) != 5 or refs.shape != (837, 256) or np.unique(refs).size != refs.size:
        raise ValueError("LTE-V physical block dimensions or uniqueness differ")
    for index, fold in enumerate(folds, 1):
        ids = [np.asarray(fold[s], dtype=np.int64) for s in ("train", "val", "test")]
        if (fold["fold"] != index or len(np.unique(np.concatenate(ids))) != 837
                or not np.array_equal(np.sort(np.concatenate(ids)), np.arange(837))):
            raise ValueError("Invalid LTE-V train/validation/test partition")
    for value in (references, refs, labels, rx):
        value.setflags(write=False)
    return PreparedLTEV(references, refs, labels, rx, folds, records, dict(config["condition"]))


def get_split(prepared, config, fold, split):
    """Return XFR rows [N,2,256], row TX labels, and repeated global block IDs.

    Row order is registry split order, then reference position k=0,...,255.
    Thus input row b*256+k contains IQ[k] from each of that block's 256 frames.
    """
    _validate_config(config)
    if prepared.condition != config["condition"]:
        raise ValueError("Prepared reference condition differs from config")
    if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)) or not 1 <= fold <= 5:
        raise ValueError("fold must be an integer from 1 through 5")
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    block_ids = np.asarray(prepared.folds[fold - 1][split], dtype=np.int64)
    inputs = np.empty((len(block_ids) * 256, 2, 256), dtype=np.float32)
    # Bounded temporary indexing avoids keeping a second full split IQ store.
    for start in range(0, len(block_ids), 16):
        ids = block_ids[start:start + 16]
        blocks = prepared.references[prepared.block_refs[ids]]
        rows = cross_frame_rows(blocks)
        inputs[start * 256:(start + len(ids)) * 256] = rows.transpose(0, 1, 3, 2).reshape(-1, 2, 256)
    return (inputs, np.repeat(prepared.block_labels[block_ids], 256),
            np.repeat(block_ids, 256))


def load_test_bundle(path, config):
    """Build the same test rows from a portable four-array raw-reference NPZ."""
    _validate_config(config)
    expected = config.get("test_bundle_sha256")
    if expected is None or _sha256(Path(path)) != expected:
        raise ValueError("The test bundle checksum differs from the released file")
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"references", "reference_ids", "labels", "block_ids"}:
            raise ValueError("Expected references, reference_ids, labels, and block_ids")
        signals, reference_ids = bundle["references"], bundle["reference_ids"]
        labels, block_ids = bundle["labels"], bundle["block_ids"]
    if (signals.dtype != np.complex128 or signals.shape != (210, 256, 256)
            or reference_ids.dtype != np.int32 or reference_ids.shape != (210, 256)
            or labels.dtype != np.int64 or labels.shape != (210,)
            or block_ids.dtype != np.int64 or block_ids.shape != (210,)):
        raise ValueError("Unexpected raw test-bundle array shapes or dtypes")
    registry = _json(_CONFIG_DIR / "records.json")
    records = registry["records"]
    if _sha256(_CONFIG_DIR / "folds.json") != registry["folds_sha256"]:
        raise ValueError("Frozen LTE-V fold registry differs")
    folds = _json(_CONFIG_DIR / "folds.json")
    refs, block_labels, _ = build_blocks(records)
    if (not all(fold["test"] == block_ids.tolist() for fold in folds)
            or not np.array_equal(reference_ids, refs[block_ids])
            or not np.array_equal(labels, block_labels[block_ids])):
        raise ValueError("The test bundle differs from the registered physical test blocks")
    indices = reference_ids.ravel()
    signals = signals.reshape(-1, 256)
    observations = np.empty((len(signals), 256, 2), dtype=np.float32)
    rng = np.random.RandomState(config["condition"]["seed"])
    for record in records:
        first, count = record["offset"], record["n"]
        selected = (indices >= first) & (indices < first + count)
        positions = indices[selected] - first
        # Omitted frames consume their original Gaussian draws. The source
        # float32 cast also matches the MAT reader's complex64 arithmetic.
        raw = np.zeros((count, 256, 2), dtype=np.float32)
        raw[positions, :, 0] = signals[selected].real
        raw[positions, :, 1] = signals[selected].imag
        observations[selected] = condition_references(raw, config["condition"], rng)[positions]
    physical = observations.reshape(len(labels), 256, 256, 2)
    rows = cross_frame_rows(physical).transpose(0, 1, 3, 2)
    return (np.ascontiguousarray(rows.reshape(-1, 2, 256)),
            np.repeat(labels, 256), np.repeat(block_ids, 256))
