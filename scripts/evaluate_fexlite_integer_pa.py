#!/usr/bin/env python3
"""Evaluate an exported FExLite integer DPD through a frozen OpenDPD PA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.rtl_evaluate import evaluate_fexlite_integer_pa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pa-checkpoint", required=True, type=Path)
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--dataset-name")
    dataset.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--protocol", choices=("segmented",), default="segmented")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument(
        "--qat-checkpoint",
        type=Path,
        help="Optional fake-QAT comparison checkpoint; SHA must match the manifest",
    )
    parser.add_argument("--nperseg", type=int, help="Override dataset spec segment length")
    parser.add_argument("--sample-rate", type=float, help="Override dataset spec sample rate")
    parser.add_argument(
        "--bw-main-ch", type=float, help="Override dataset spec main-channel bandwidth"
    )
    parser.add_argument(
        "--n-sub-ch", type=int, help="Override dataset spec subchannel count"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_fexlite_integer_pa(
        manifest_path=args.manifest,
        pa_checkpoint=args.pa_checkpoint,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_path=args.output,
        split=args.split,
        protocol=args.protocol,
        device=args.device,
        qat_checkpoint=args.qat_checkpoint,
        nperseg=args.nperseg,
        sample_rate=args.sample_rate,
        bw_main_ch=args.bw_main_ch,
        n_sub_ch=args.n_sub_ch,
    )
    print(json.dumps({
        "status": result["status"],
        "format": result["format"],
        "output": args.output.name,
        "protocol_sha256": result["protocol_sha256"],
        "provenance_sha256": result["provenance_sha256"],
        "integer_dpd_frozen_pa": result["metrics"]["integer_dpd_frozen_pa"],
    }, indent=2))


if __name__ == "__main__":
    main()
