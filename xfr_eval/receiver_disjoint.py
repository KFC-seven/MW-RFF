"""G2-only XFR-CD evaluation for the five reported WiSig receiver assignments."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .receiver_features import spectral_bank, project, batches
from .receiver_model import ReceiverClassifier
from xfr.wisig_data import load_dataset as load_wisig_dataset

CONFIG = Path(__file__).resolve().parents[1] / "configs/receiver_disjoint.json"


def load_dataset(path, config):
    dataset = load_wisig_dataset(path, config["data_sha256"])
    if (list(map(str, dataset["tx_list"])) != config["transmitters"]
            or list(map(str, dataset["rx_list"])) != config["receivers"]):
        raise ValueError("Unexpected ManySig transmitter or receiver order")
    return dataset


def test_blocks(dataset, case, config):
    date = dataset["capture_date_list"].index(config["date"])
    equalized = dataset["equalized_list"].index(0)
    receivers = case["receiver_indices"]["test"]
    physical = np.empty((150, 160, 160, 2), np.float32)
    crop = slice(*config["crop"])
    for tx in range(6):
        records = {}
        for rx in receivers:
            record = np.asarray(dataset["data"][tx][rx][date][equalized])
            if record.shape != (1000, 256, 2) or not np.isfinite(record).all():
                raise ValueError("Expected finite ManySig records with shape [1000,256,2]")
            records[rx] = record[:, crop, :].astype(np.float32)
        for round_id in range(25):
            members = [(rx, round_id * 40 + start + step)
                       for start in range(0, 40, 5) for rx in receivers for step in range(5)]
            physical[tx * 25 + round_id] = np.stack([records[rx][packet] for rx, packet in members])
    return physical


def load_assets(bundle, case):
    payloads = {}
    for name, checksum in (("model.pt", case["model_sha256"]), ("pca.npz", case["pca_sha256"])):
        member = f"receiver_disjoint/{case['case']}/{name}"
        if bundle.namelist().count(member) != 1:
            raise ValueError(f"Expected exactly one ZIP member: {member}")
        payload = bundle.read(member)
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError(f"Receiver-disjoint asset signature mismatch: {member}")
        payloads[name] = payload
    state = torch.load(BytesIO(payloads["model.pt"]), map_location="cpu", weights_only=True)
    with np.load(BytesIO(payloads["pca.npz"]), allow_pickle=False) as archive:
        pca = {key: archive[key] for key in ("mean", "components")}
    if (pca["mean"].shape != (160,) or pca["components"].shape != (160, 32)
            or any(value.dtype != np.float64 or not np.isfinite(value).all() for value in pca.values())):
        raise ValueError("Invalid saved G0 PCA32")
    return state, pca


def evaluate(args):
    """Return one {case, correct, total, accuracy_pct, unit} record per G2 split.

    The caller owns summary output via args.output. This function performs no
    training, source validation, fitted-statistic updates, or temporary extraction.
    """
    if args.experiment != "wisig-unseen":
        raise ValueError("Receiver-disjoint evaluation requires experiment='wisig-unseen'")
    config = json.loads(CONFIG.read_text(encoding="utf8"))
    selected = [case for case in config["cases"] if args.case == "all" or args.case == case["case"]]
    if not selected:
        raise ValueError("Unknown receiver assignment; use all or one of the five released case IDs")
    if args.device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = load_dataset(args.data, config)
    results = []
    with ZipFile(args.weights) as bundle, torch.inference_mode():
        for case in selected:
            state, pca = load_assets(bundle, case)
            model = ReceiverClassifier()
            model.load_state_dict(state, strict=True)
            model.eval().to(args.device)
            physical = test_blocks(dataset, case, config)
            bank = spectral_bank(physical, case["train_complex_rms"], case["variant"] in ("refZ", "refZ_mixRX"))
            del physical
            projection = project(bank, pca)
            labels = np.repeat(np.arange(6, dtype=np.int64), 4000)
            correct, total = 0, 0
            for values in batches(bank, projection):
                logits = model(torch.from_numpy(values).to(args.device))
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite receiver-disjoint model outputs")
                decisions = logits.argmax(1).cpu().numpy()
                correct += int((decisions == labels[total:total + len(decisions)]).sum())
                total += len(decisions)
            results.append(dict(case=case["case"], correct=correct, total=total,
                                accuracy_pct=100 * correct / total, unit="row"))
    return results
