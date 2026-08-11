#!/usr/bin/env python3
"""Verify an exported FExLite QAT package at exactly 0 LSB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.rtl_export import verify_exported_golden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to an opendpd_fexlite_qat_rtl_export manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_exported_golden(args.manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "format": "opendpd_fexlite_qat_rtl_export_verification",
                "format_version": 1,
                "trace_count": len(result),
                "traces": result,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
