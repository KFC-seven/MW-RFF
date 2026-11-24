"""Evaluate pretrained XFR models on WiSig ManySig or LTE-V DMRS references."""
import argparse
from collections import defaultdict
from importlib import import_module
import json
from pathlib import Path
import statistics

import torch

ROOT = Path(__file__).resolve().parent
PROFILES = {
    "wisig-date": ("wisig_history", "wisig-cross-date.zip"),
    "wisig-20db": ("wisig_history", "wisig-20db.zip"),
    "wisig-unseen": ("receiver_disjoint", "wisig-unseen-receivers.zip"),
    "ltev-sd-20db": ("ltev20", "ltev-20db.zip"),
    "ltev-cd-20db": ("ltev20", "ltev-20db.zip"),
    "wisig-multiframe": ("multiframe", "wisig-multiframe.zip"),
    "ltev-multiframe": ("multiframe", "ltev-multiframe.zip"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="experiment", required=True)
    for name in PROFILES:
        cmd = commands.add_parser(name)
        cmd.add_argument("--data", type=Path, required=True,
                         help="A released test NPZ, the original ManySig.pkl, or a directory of LTE-V MAT files")
        cmd.add_argument("--weights", type=Path, help="Model ZIP (defaults to weights/ in this repository)")
        cmd.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
        cmd.add_argument("--output", type=Path, help="Save scores and correct/total counts to a JSON file")
        if name == "wisig-unseen":
            cmd.add_argument("--case", choices=["all", "N09", "N10", "O05", "N05", "N08"], default="all")
        else:
            cmd.add_argument("--fold", choices=["all", "1", "2", "3", "4", "5"], default="all")
        if name == "wisig-date":
            dates = ["2021-03-01", "2021-03-08", "2021-03-15", "2021-03-23"]
            cmd.add_argument("--source-date", choices=dates, help="Training date (default 2021-03-15)")
            cmd.add_argument("--target-date", choices=dates, help="Test date (default 2021-03-01)")
            cmd.add_argument("--all-pairs", action="store_true", help="Evaluate the 16 train/test date pairs in Table III")
        if name in ("ltev-sd-20db", "ltev-cd-20db"):
            cmd.add_argument("--profile", choices=["legacy", "paper256"], default="legacy",
                             help="DMRS extraction profile described in docs/data.md")
    args = parser.parse_args()
    if args.experiment == "wisig-date":
        if args.all_pairs and (args.source_date or args.target_date):
            parser.error("Choose --all-pairs or specify a single date pair.")
        if not args.all_pairs:
            args.source_date = args.source_date or "2021-03-15"
            args.target_date = args.target_date or "2021-03-01"
    archive = PROFILES[args.experiment][1]
    if args.weights is None:
        args.weights = ROOT / "weights" / archive
    if not args.weights.is_file():
        parser.error(f"Place {archive} in weights/ or pass --weights PATH.")
    if not args.data.exists():
        parser.error(f"Data path does not exist: {args.data}")
    if args.output and (args.output.suffix.lower() != ".json" or
                        args.output.resolve() in (args.data.resolve(), args.weights.resolve())):
        parser.error("--output must name a JSON results file separate from the data and model.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch.")
    return args


def summarize(args, results):
    report = {"experiment": args.experiment, "unit": results[0]["unit"], "runs": results}
    scores = [row["accuracy_pct"] for row in results]
    if getattr(args, "all_pairs", False):
        by_pair = defaultdict(list)
        for row in results:
            by_pair[row["source_date"], row["target_date"]].append(row["accuracy_pct"])
        report["date_pairs"] = [
            {"source_date": source, "target_date": target, "mean_pct": statistics.mean(values)}
            for (source, target), values in by_pair.items()
        ]
        scores = [pair["mean_pct"] for pair in report["date_pairs"]]
    for name in ("case", "fold", "profile", "source_date", "target_date"):
        value = getattr(args, name, None)
        if value is not None:
            report[name] = value
    report["mean_pct"] = statistics.mean(scores)
    if len(scores) > 1:
        report["sample_sd_pct"] = statistics.stdev(scores)
    return report


def main():
    args = parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = args.experiment == "wisig-unseen"
    torch.backends.cuda.matmul.allow_tf32 = False
    module = import_module("xfr_eval." + PROFILES[args.experiment][0])
    results = module.evaluate(args)
    report = summarize(args, results)
    for row in results:
        print(json.dumps(row, allow_nan=False))
    print(f"Mean {report['unit']} accuracy: {report['mean_pct']:.2f}%")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
