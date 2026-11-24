"""Matched-budget multi-frame evaluation on WiSig and LTE-V."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from xfr import ltev, wisig
from xfr.evaluation import evaluate_inputs
from xfr.models import create_model


def evaluate(args):
    ds = "wisig" if args.experiment == "wisig-multiframe" else "ltev"
    cfg = json.loads((Path(__file__).resolve().parents[1]/"configs"/ds/"multiframe.json").read_text())
    loader = wisig if ds == "wisig" else ltev
    folds = list(range(1,6)) if args.fold == "all" else [int(args.fold)]
    path = Path(args.data)
    if ds == "ltev" and path.is_file() and path.suffix.lower() == ".npz":
        inputs, labels, block_ids = ltev.load_test_bundle(path, cfg)
    else:
        data = loader.prepare(args.data, cfg)
        test_ids = data.folds[folds[0]-1]["test"]
        if any(data.folds[f-1]["test"] != test_ids for f in folds):
            raise ValueError("The multi-frame profile requires one shared test pool.")
        inputs, labels, block_ids = loader.get_split(data, cfg, folds[0], "test")
        del data
    results = []
    with ZipFile(args.weights) as archive:
        for fold in folds:
            model = create_model(cfg)
            checkpoint = cfg["checkpoints"][str(fold)]
            raw = archive.read(checkpoint["file"])
            if hashlib.sha256(raw).hexdigest() != checkpoint["sha256"]:
                raise ValueError(f"Unexpected weights for {ds} fold {fold}")
            state = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            metrics, _ = evaluate_inputs(model.to(args.device), inputs, labels, block_ids, args.device)
            results.append(dict(fold=fold, correct=metrics["block_correct"], total=metrics["blocks"],
                                accuracy_pct=metrics["block_accuracy_pct"], unit="block",
                                row_correct=metrics["row_correct"], row_total=metrics["rows"],
                                row_accuracy_pct=metrics["row_accuracy_pct"]))
    return results
