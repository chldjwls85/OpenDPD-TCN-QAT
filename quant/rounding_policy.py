"""Explicit, digest-bound rounding contracts for RTL-oriented TCN QAT."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


ROUND_TO_NEAREST_TIES_TO_EVEN = "round_to_nearest_ties_to_even"
DISCARD_LSB_SIGNED_FLOOR = "discard_lsb_signed_floor"
SUPPORTED_ROUNDING_MODES = {
    ROUND_TO_NEAREST_TIES_TO_EVEN,
    DISCARD_LSB_SIGNED_FLOOR,
}

BASELINE_RNE = "baseline_rne"
PREHS_FLOOR = "prehs_floor"
GLOBAL_FLOOR = "global_floor"
RESEARCH_PREHS_INPUT_FLOOR = "research_prehs_input_floor"
RESEARCH_PREHS_INPUT_RNE = "research_prehs_input_rne"
RESEARCH_POSTHS_ACTIVATION_FLOOR = "research_posths_activation_floor"
RESEARCH_GLOBAL_FLOOR_NO_PREHS = "research_global_floor_no_prehs"
OFFICIAL_POLICY_MODES = {BASELINE_RNE, PREHS_FLOOR, GLOBAL_FLOOR}
RESEARCH_POLICY_MODES = {
    RESEARCH_PREHS_INPUT_FLOOR,
    RESEARCH_PREHS_INPUT_RNE,
    RESEARCH_POSTHS_ACTIVATION_FLOOR,
    RESEARCH_GLOBAL_FLOOR_NO_PREHS,
}
SUPPORTED_POLICY_MODES = OFFICIAL_POLICY_MODES | RESEARCH_POLICY_MODES
POLICY_MODE_CODES = {
    BASELINE_RNE: 0,
    PREHS_FLOOR: 1,
    GLOBAL_FLOOR: 2,
    RESEARCH_PREHS_INPUT_FLOOR: 3,
    RESEARCH_POSTHS_ACTIVATION_FLOOR: 4,
    RESEARCH_PREHS_INPUT_RNE: 5,
    RESEARCH_GLOBAL_FLOOR_NO_PREHS: 6,
}
POLICY_MODES_BY_CODE = {
    code: mode for mode, code in POLICY_MODE_CODES.items()
}

POLICY_FORMAT = "opendpd_fexlite_rounding_policy"
POLICY_VERSION = 1
BOUNDARY_NAMES = (
    "stored_raw_input",
    "stored_weight",
    "stored_bias",
    "fex_feature_requantization",
    "pre_hardswish_requantization",
    "post_hardswish_activation_requantization",
    "residual_output_requantization",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def policy_boundaries(mode: str) -> dict[str, str]:
    if mode not in SUPPORTED_POLICY_MODES:
        raise ValueError(f"unsupported rounding policy mode: {mode!r}")
    runtime_floor = mode in {GLOBAL_FLOOR, RESEARCH_GLOBAL_FLOOR_NO_PREHS}
    pre_hardswish_floor = mode in {
        PREHS_FLOOR,
        GLOBAL_FLOOR,
        RESEARCH_PREHS_INPUT_FLOOR,
        RESEARCH_GLOBAL_FLOOR_NO_PREHS,
    }
    post_hardswish_floor = mode in {
        PREHS_FLOOR,
        GLOBAL_FLOOR,
        RESEARCH_POSTHS_ACTIVATION_FLOOR,
        RESEARCH_GLOBAL_FLOOR_NO_PREHS,
    }
    return {
        "stored_raw_input": ROUND_TO_NEAREST_TIES_TO_EVEN,
        "stored_weight": ROUND_TO_NEAREST_TIES_TO_EVEN,
        "stored_bias": ROUND_TO_NEAREST_TIES_TO_EVEN,
        "fex_feature_requantization": (
            DISCARD_LSB_SIGNED_FLOOR
            if runtime_floor else ROUND_TO_NEAREST_TIES_TO_EVEN
        ),
        "pre_hardswish_requantization": (
            DISCARD_LSB_SIGNED_FLOOR
            if pre_hardswish_floor else ROUND_TO_NEAREST_TIES_TO_EVEN
        ),
        "post_hardswish_activation_requantization": (
            DISCARD_LSB_SIGNED_FLOOR
            if post_hardswish_floor else ROUND_TO_NEAREST_TIES_TO_EVEN
        ),
        "residual_output_requantization": (
            DISCARD_LSB_SIGNED_FLOOR
            if runtime_floor else ROUND_TO_NEAREST_TIES_TO_EVEN
        ),
    }


def quantizes_pre_hardswish_input(mode: str) -> bool:
    """Return whether the policy inserts an explicit narrow Pre-HS boundary.

    Rounding and boundary presence are intentionally separate: the RNE
    research control narrows to the activation grid without changing the
    rounding rule.
    """
    if mode not in SUPPORTED_POLICY_MODES:
        raise ValueError(f"unsupported rounding policy mode: {mode!r}")
    return mode in {
        PREHS_FLOOR,
        GLOBAL_FLOOR,
        RESEARCH_PREHS_INPUT_FLOOR,
        RESEARCH_PREHS_INPUT_RNE,
    }


def rounding_policy_record(mode: str) -> dict[str, Any]:
    unsigned = {
        "format": POLICY_FORMAT,
        "format_version": POLICY_VERSION,
        "mode": mode,
        "boundaries": policy_boundaries(mode),
    }
    return {
        **unsigned,
        "capability_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def validate_rounding_policy(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("rounding_policy must be an object")
    mode = record.get("mode")
    expected = rounding_policy_record(mode)
    if dict(record) != expected:
        raise ValueError(
            "rounding_policy fields or capability_sha256 do not match the "
            f"canonical {mode!r} contract"
        )
    return expected


def mode_from_legacy_activation_spec(
    quantize_hardswish_input: bool,
    activation_rounding: str,
) -> str:
    if not quantize_hardswish_input and (
        activation_rounding == ROUND_TO_NEAREST_TIES_TO_EVEN
    ):
        return BASELINE_RNE
    if quantize_hardswish_input and (
        activation_rounding == DISCARD_LSB_SIGNED_FLOOR
    ):
        return PREHS_FLOOR
    raise ValueError("legacy activation spec does not map to a supported policy mode")


__all__ = [
    "BASELINE_RNE",
    "BOUNDARY_NAMES",
    "DISCARD_LSB_SIGNED_FLOOR",
    "GLOBAL_FLOOR",
    "OFFICIAL_POLICY_MODES",
    "POLICY_MODE_CODES",
    "POLICY_MODES_BY_CODE",
    "POLICY_FORMAT",
    "POLICY_VERSION",
    "PREHS_FLOOR",
    "RESEARCH_POLICY_MODES",
    "RESEARCH_GLOBAL_FLOOR_NO_PREHS",
    "RESEARCH_POSTHS_ACTIVATION_FLOOR",
    "RESEARCH_PREHS_INPUT_FLOOR",
    "RESEARCH_PREHS_INPUT_RNE",
    "ROUND_TO_NEAREST_TIES_TO_EVEN",
    "SUPPORTED_POLICY_MODES",
    "SUPPORTED_ROUNDING_MODES",
    "mode_from_legacy_activation_spec",
    "policy_boundaries",
    "quantizes_pre_hardswish_input",
    "rounding_policy_record",
    "validate_rounding_policy",
]
