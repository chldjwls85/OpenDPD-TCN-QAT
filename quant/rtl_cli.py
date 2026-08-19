"""Installed command-line entry points for the RTL export contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rtl_export import export_fexlite_qat_rtl, verify_exported_golden


def export_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opendpd-rtl-export",
        description="Export a full-I/O QAT FExLite causal TCN for TCN-Compiler.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pa-checkpoint", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--golden-start", type=int, default=0)
    parser.add_argument("--golden-length", type=int, default=256)
    parser.add_argument("--golden-seed", type=int, default=2026)
    parser.add_argument(
        "--rounding-policy-mode",
        choices=[
            "baseline_rne", "prehs_floor", "global_floor",
            "research_prehs_input_floor",
            "research_prehs_input_rne",
            "research_posths_activation_floor",
            "research_global_floor_no_prehs",
        ],
        help="Verify that the checkpoint declares this rounding policy.",
    )
    args = parser.parse_args(argv)
    manifest = export_fexlite_qat_rtl(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        pa_checkpoint=args.pa_checkpoint,
        input_path=args.input,
        golden_start=args.golden_start,
        golden_length=args.golden_length,
        golden_seed=args.golden_seed,
        dataset_name=args.dataset_name,
        rounding_policy_mode=args.rounding_policy_mode,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest": str(args.output_dir / "manifest.json"),
                "hidden_channels": manifest["model"]["hidden_channels"],
                "temporal_layers": manifest["model"]["temporal_layers"],
                "mac_per_sample": manifest["model"]["mac_per_sample"],
            },
            indent=2,
        )
    )
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opendpd-rtl-verify",
        description="Recompute every exported integer golden trace at 0 LSB.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    traces = verify_exported_golden(args.manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "trace_count": len(traces),
                "traces": traces,
            },
            indent=2,
        )
    )
    return 0
