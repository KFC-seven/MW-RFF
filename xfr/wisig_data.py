"""Read the shared WiSig reference bundle or the original ManySig pickle."""
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np

CONFIG = Path(__file__).resolve().parents[1] / "configs/wisig_data.json"


def _registry():
    return json.loads(CONFIG.read_text(encoding="utf8"))


def data_file(path):
    path = Path(path).expanduser()
    if path.is_file():
        return path
    for name in ("wisig-test.npz", "ManySig.pkl", "ManySig.pkl/ManySig.pkl"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"WiSig data was not found under {path}")


def verify_source(path, expected_source_sha256=None):
    path = Path(path)
    registry = _registry()
    if expected_source_sha256 and expected_source_sha256 != registry["source_sha256"]:
        raise ValueError("Configured ManySig source checksum differs from the shared profile")
    expected = registry["sha256"] if path.suffix.lower() == ".npz" else registry["source_sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("WiSig data checksum differs from the released data profile")


class _DateStore:
    def __init__(self, path, registry):
        self.archive = np.load(path, allow_pickle=False)
        self.registry = registry
        self.cache = {}
        metadata = {"date": registry["dates"], "transmitters": registry["transmitters"],
                    "receivers": registry["receivers"]}
        expected_keys = set(metadata) | {"equalized"} | set(registry["arrays"])
        if set(self.archive.files) != expected_keys or len(self.archive.files) != len(expected_keys):
            self.archive.close()
            raise ValueError("Unexpected WiSig reference bundle members")
        for key, expected in metadata.items():
            actual = self.archive[key]
            if actual.dtype.kind != "U" or actual.tolist() != expected:
                self.archive.close()
                raise ValueError(f"WiSig bundle {key} ordering differs")
        equalized = self.archive["equalized"]
        if equalized.shape != () or equalized.dtype != np.int8 or int(equalized) != 0:
            self.archive.close()
            raise ValueError("Expected non-equalized WiSig references")

    def date(self, index):
        key = "iq_" + self.registry["dates"][index]
        if key not in self.cache:
            values = self.archive[key]
            expected = self.registry["arrays"][key]
            if (list(values.shape) != expected["shape"] or values.dtype.str != expected["dtype"]
                    or not values.flags.c_contiguous or not np.isfinite(values).all()
                    or hashlib.sha256(memoryview(values).cast("B")).hexdigest() != expected["sha256"]):
                raise ValueError(f"WiSig reference array differs: {key}")
            values.setflags(write=False)
            self.cache[key] = values
        return self.cache[key]

    def __del__(self):
        archive = getattr(self, "archive", None)
        if archive is not None:
            archive.close()


class _DataView:
    """Expose data[TX][RX][date][equalized] without copying date arrays."""
    def __init__(self, store, indices=()):
        self.store = store
        self.indices = indices

    def __len__(self):
        return (6, 12, 4, 1)[len(self.indices)]

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not -len(self) <= index < len(self):
            raise IndexError("WiSig reference index is outside its domain")
        indices = self.indices + (int(index) % len(self),)
        if len(indices) < 4:
            return _DataView(self.store, indices)
        tx, rx, date, _ = indices
        return self.store.date(date)[tx, rx]


def load_dataset(path, expected_source_sha256=None):
    """Verify the input before decoding and return the common ManySig interface."""
    path = data_file(path)
    verify_source(path, expected_source_sha256)
    registry = _registry()
    if path.suffix.lower() == ".npz":
        store = _DateStore(path, registry)
        return dict(tx_list=registry["transmitters"], rx_list=registry["receivers"],
                    capture_date_list=registry["dates"], equalized_list=[0], data=_DataView(store))
    with path.open("rb") as stream:
        book = pickle.load(stream)
    if (list(map(str, book["tx_list"])) != registry["transmitters"]
            or list(map(str, book["rx_list"])) != registry["receivers"]
            or not set(registry["dates"]).issubset(book["capture_date_list"])
            or 0 not in book["equalized_list"]):
        raise ValueError("ManySig identities differ from the released data profile")
    return book
