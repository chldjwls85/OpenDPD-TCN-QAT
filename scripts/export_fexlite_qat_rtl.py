#!/usr/bin/env python3
"""Export an OpenDPD full-I/O QAT FExLite causal TCN for RTL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.rtl_export import export_fexlite_qat_rtl


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New package directory; an existing path is never overwritten",
    )
    parser.add_argument("--pa-checkpoint", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--input", type=Path, help="Optional .csv/.npy/.pt I/Q golden input")
    parser.add_argument("--golden-start", type=int, default=0)
    parser.add_argument("--golden-length", type=int, default=256)
    parser.add_argument("--golden-seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = export_fexlite_qat_rtl(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        pa_checkpoint=args.pa_checkpoint,
        input_path=args.input,
        golden_start=args.golden_start,
        golden_length=args.golden_length,
        golden_seed=args.golden_seed,
        dataset_name=args.dataset_name,
    )
    comparison = manifest["golden_vectors"]["integer_vs_fake_qat"]
    print(json.dumps({
        "status": "pass",
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "output_dir": str(args.output_dir.resolve()),
        "manifest": str((args.output_dir / "manifest.json").resolve()),
        "weight_count": manifest["model"]["weight_count"],
        "hidden_channels": manifest["model"]["hidden_channels"],
        "temporal_layers": manifest["model"]["temporal_layers"],
        "kernel_size": manifest["model"]["kernel_size"],
        "dilations": manifest["model"]["dilations"],
        "receptive_field_samples": manifest["model"]["receptive_field_samples"],
        "comparison": comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
