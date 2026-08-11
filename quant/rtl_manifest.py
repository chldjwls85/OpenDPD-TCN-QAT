"""Strict validation for the TCN-Compiler/OpenDPD manifest-v1 boundary."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


MANIFEST_FORMAT = "opendpd_fexlite_qat_rtl_export"
MANIFEST_VERSION = 1
ROUNDING_MODE = "round_to_nearest_ties_to_even"
DISCARD_LSB_SIGNED_FLOOR = "discard_lsb_signed_floor"
SUPPORTED_ROUNDING_MODES = {ROUNDING_MODE, DISCARD_LSB_SIGNED_FLOOR}
QUANTIZER_OVERFLOW = "saturate_at_explicit_quantizer_output"
MEM_ENCODING = "two_complement_hex_one_word_per_line"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _power_of_two(scale: Any, exponent: Any, label: str) -> None:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"{label}.scale must be numeric")
    exponent = _integer(exponent, f"{label}.scale_exponent")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError(f"{label}.scale must be finite and positive")
    if float(scale) != 2.0**exponent:
        raise ValueError(f"{label}.scale is inconsistent with its power-of-two exponent")


def _quantizer(
    record: Mapping[str, Any],
    label: str,
    expected_bits: int | None = None,
    expected_rounding: str | None = None,
) -> int:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be an object")
    bits = _integer(record.get("bits"), f"{label}.bits", 2)
    if expected_bits is not None and bits != expected_bits:
        raise ValueError(f"{label}.bits must be {expected_bits}")
    if record.get("signed") is not True:
        raise ValueError(f"{label} must be signed")
    if record.get("qmin") != -(2 ** (bits - 1)):
        raise ValueError(f"{label}.qmin is not the signed {bits}-bit minimum")
    if record.get("qmax") != 2 ** (bits - 1) - 1:
        raise ValueError(f"{label}.qmax is not the signed {bits}-bit maximum")
    if record.get("zero_point") != 0:
        raise ValueError(f"{label}.zero_point must be zero")
    _power_of_two(record.get("scale"), record.get("scale_exponent"), label)
    rounding = record.get("rounding")
    if expected_rounding is not None and rounding != expected_rounding:
        description = (
            "signed RNE ties-to-even"
            if expected_rounding == ROUNDING_MODE else expected_rounding
        )
        raise ValueError(f"{label}.rounding must be {description}")
    if rounding not in SUPPORTED_ROUNDING_MODES:
        raise ValueError(f"{label}.rounding is unsupported: {rounding!r}")
    if record.get("overflow") != QUANTIZER_OVERFLOW:
        raise ValueError(f"{label}.overflow must describe explicit saturation")
    return bits


def _resolve_relative(root: Path, relative: Any, prefix: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("memory path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe memory path: {relative!r}")
    if prefix == "weights" and len(candidate.parts) != 1:
        raise ValueError(f"weight memory must be a basename: {relative!r}")
    if prefix == "golden_vectors" and (
        len(candidate.parts) != 2 or candidate.parts[0] != prefix
    ):
        raise ValueError(f"golden memory must be under golden_vectors/: {relative!r}")
    path = root / prefix / candidate if prefix == "weights" else root / candidate
    if not path.is_file():
        raise FileNotFoundError(f"declared memory does not exist: {path}")
    return path


def _memory(
    root: Path,
    info: Mapping[str, Any],
    *,
    label: str,
    prefix: str,
    expected_count: int,
    expected_bits: int,
) -> tuple[dict[str, Any], list[int]]:
    if not isinstance(info, Mapping):
        raise ValueError(f"{label} memory descriptor must be an object")
    if info.get("encoding") != MEM_ENCODING:
        raise ValueError(f"{label} memory encoding is unsupported")
    if info.get("count") != expected_count:
        raise ValueError(f"{label} memory count mismatch")
    if info.get("bits") != expected_bits:
        raise ValueError(f"{label} memory bit width mismatch")
    expected_sha = info.get("sha256")
    if not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None:
        raise ValueError(f"{label} memory SHA-256 is missing or malformed")
    path = _resolve_relative(root, info.get("path"), prefix)
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{label} memory SHA mismatch: expected {expected_sha}, got {actual_sha}"
        )
    words = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(words) != expected_count:
        raise ValueError(f"{label} memory contains {len(words)} words, expected {expected_count}")
    mask = (1 << expected_bits) - 1
    encoded = []
    for word in words:
        try:
            value = int(word, 16)
        except ValueError as exc:
            raise ValueError(f"{label} memory contains non-hex data") from exc
        if value > mask:
            raise ValueError(f"{label} memory word exceeds its declared bit width")
        encoded.append(value)
    sign = 1 << (expected_bits - 1)
    modulus = 1 << expected_bits
    signed = [value - modulus if value & sign else value for value in encoded]
    return {
        "role": label,
        "path_hint": "/".join(path.relative_to(root).parts),
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha,
    }, signed


def validate_manifest_v1(manifest_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and strictly validate a complete format-v1 export package."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"unsupported manifest format: {manifest.get('format')!r}")
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version: {manifest.get('format_version')!r}")
    root = manifest_path.parent

    quantization = manifest.get("quantization")
    if not isinstance(quantization, Mapping):
        raise ValueError("manifest.quantization must be an object")
    activation_bits = _quantizer(
        quantization.get("raw_input"), "raw_input",
        expected_rounding=ROUNDING_MODE,
    )
    _quantizer(
        quantization.get("dpd_output"), "dpd_output", activation_bits,
        expected_rounding=ROUNDING_MODE,
    )
    expected_boundary_scale = 2.0 ** (1 - activation_bits)
    for name in ("raw_input", "dpd_output"):
        if float(quantization[name]["scale"]) != expected_boundary_scale:
            raise ValueError(f"{name} physical scale must be exactly 2^(1-A)")
    numeric_contract = quantization.get("numeric_contract")
    if not isinstance(numeric_contract, Mapping):
        raise ValueError("numeric_contract must be an object")
    if numeric_contract.get("rounding") != ROUNDING_MODE:
        raise ValueError("numeric_contract rounding must be signed RNE ties-to-even")
    activation_rounding = numeric_contract.get(
        "activation_boundary_rounding", ROUNDING_MODE
    )
    if activation_rounding not in SUPPORTED_ROUNDING_MODES:
        raise ValueError("numeric_contract activation boundary rounding is unsupported")
    if not str(numeric_contract.get("saturation", "")).startswith("only at explicit "):
        raise ValueError("numeric_contract must declare explicit saturation")

    model = manifest.get("model")
    if not isinstance(model, Mapping) or model.get("backbone") != "fexlite_causal_tcn":
        raise ValueError("manifest model must be fexlite_causal_tcn")
    hidden = _integer(model.get("hidden_channels"), "hidden_channels", 1)
    temporal_layers = _integer(model.get("temporal_layers"), "temporal_layers", 1)
    kernel_size = _integer(model.get("kernel_size"), "kernel_size", 1)
    dilation_base = _integer(model.get("dilation_base"), "dilation_base", 1)
    dilations = [dilation_base**index for index in range(temporal_layers)]
    if model.get("dilations") != dilations:
        raise ValueError("model dilations do not match dilation_base")
    if model.get("input_channels") != 2 or model.get("feature_channels") != 6:
        raise ValueError("manifest requires 2 raw-I/Q and 6 FEx channels")
    if model.get("output_channels") != 2:
        raise ValueError("manifest requires two DPD output channels")
    receptive_field = 1 + sum((kernel_size - 1) * dilation for dilation in dilations)
    if model.get("receptive_field_samples") != receptive_field:
        raise ValueError("receptive field is inconsistent with kernel/dilations")

    layers = model.get("layers")
    if not isinstance(layers, list) or len(layers) != temporal_layers + 2:
        raise ValueError("manifest layer count must be temporal_layers + 2")
    artifacts: list[dict[str, Any]] = []
    total_weights = 0
    total_biases = 0
    weight_bits: int | None = None
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or layer.get("index") != index:
            raise ValueError("manifest layer indices must be contiguous")
        if index == 0:
            expected = ("input_projection", "input_projection", 6, hidden, 1, 1, 1)
        elif index == len(layers) - 1:
            expected = ("output_projection", "output_projection", hidden, 2, 1, 1, 1)
        else:
            expected = (
                f"dw{index}", "causal_depthwise", hidden, hidden, hidden,
                kernel_size, dilations[index - 1],
            )
        name, kind, in_channels, out_channels, groups, kernel, dilation = expected
        fields = (
            ("name", name), ("kind", kind), ("in_channels", in_channels),
            ("out_channels", out_channels), ("groups", groups),
            ("kernel_size", kernel), ("dilation", dilation), ("stride", 1),
        )
        for field, value in fields:
            if layer.get(field) != value:
                raise ValueError(f"layer {index} {field} must be {value!r}")
        delays = [(kernel - 1 - tap) * dilation for tap in range(kernel)]
        if layer.get("causal_delay_by_pytorch_kernel_index") != delays:
            raise ValueError(f"layer {index} causal tap order mismatch")
        shape = [out_channels, in_channels // groups, kernel]
        if layer.get("weight_shape_oik") != shape:
            raise ValueError(f"layer {index} weight shape mismatch")
        if layer.get("weight_flatten_order") != (
            "out_channel,input_channel_per_group,kernel_index"
        ):
            raise ValueError(f"layer {index} weight flatten order mismatch")
        layer_weight_bits = _quantizer(layer.get("weight"), f"layer {index} weight")
        if weight_bits is None:
            weight_bits = layer_weight_bits
        elif layer_weight_bits != weight_bits:
            raise ValueError("all v1 layer weights must use one precision")
        expected_activation_rounding = (
            ROUNDING_MODE if index == 0 else activation_rounding
        )
        _quantizer(
            layer.get("input_activation"),
            f"layer {index} activation",
            activation_bits,
            expected_rounding=expected_activation_rounding,
        )
        weight_count = math.prod(shape)
        artifact, _ = _memory(
            root, layer["weight"].get("mem"), label=f"{name} weight",
            prefix="weights", expected_count=weight_count,
            expected_bits=layer_weight_bits,
        )
        artifacts.append(artifact)
        total_weights += weight_count

        accumulator = layer.get("accumulator")
        if not isinstance(accumulator, Mapping):
            raise ValueError(f"layer {index} accumulator must be an object")
        expected_acc_exp = (
            layer["input_activation"]["scale_exponent"]
            + layer["weight"]["scale_exponent"]
        )
        _power_of_two(
            accumulator.get("scale"), accumulator.get("scale_exponent"),
            f"layer {index} accumulator",
        )
        if accumulator.get("scale_exponent") != expected_acc_exp:
            raise ValueError(f"layer {index} accumulator scale mismatch")
        if accumulator.get("saturation") != "none_before_following_requantizer":
            raise ValueError(f"layer {index} accumulator saturation contract mismatch")
        _integer(
            accumulator.get("minimum_signed_bits_from_full_code_range"),
            f"layer {index} accumulator bits", 2,
        )
        if layer.get("followed_by_hardswish") is not (index < len(layers) - 1):
            raise ValueError(f"layer {index} activation placement mismatch")
        hardswish_input = layer.get("hardswish_input")
        if hardswish_input is not None:
            if index == len(layers) - 1:
                raise ValueError("output projection cannot have a HardSwish input quantizer")
            _quantizer(
                hardswish_input,
                f"layer {index} HardSwish input",
                activation_bits,
                expected_rounding=activation_rounding,
            )

        bias = layer.get("bias")
        if bias is not None:
            if not isinstance(bias, Mapping) or bias.get("bits") != 32:
                raise ValueError(
                    f"layer {index} bias must use signed 32-bit TCN-Compiler v1"
                )
            if bias.get("scale_exponent") != expected_acc_exp:
                raise ValueError(f"layer {index} bias scale mismatch")
            _power_of_two(bias.get("scale"), bias.get("scale_exponent"), f"layer {index} bias")
            values = bias.get("values")
            if not isinstance(values, list) or len(values) != out_channels:
                raise ValueError(f"layer {index} bias value count mismatch")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                or value < -(2**31) or value > 2**31 - 1
                for value in values
            ):
                raise ValueError(f"layer {index} bias exceeds signed 32-bit range")
            artifact, stored_values = _memory(
                root, bias.get("mem"), label=f"{name} bias", prefix="weights",
                expected_count=out_channels, expected_bits=32,
            )
            if stored_values != values:
                raise ValueError(f"layer {index} bias values disagree with memory order")
            artifacts.append(artifact)
            total_biases += out_channels

    if model.get("weight_count") != total_weights or model.get("mac_per_sample") != total_weights:
        raise ValueError("model weight/MAC count mismatch")
    if model.get("bias_count") != total_biases:
        raise ValueError("model bias count mismatch")

    golden = manifest.get("golden_vectors")
    if not isinstance(golden, Mapping):
        raise ValueError("manifest must include golden_vectors")
    length = _integer(golden.get("length"), "golden length", 1)
    files = golden.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("golden vector files must be an object")
    expected_golden: dict[str, tuple[int, int]] = {
        "raw_input": (2, activation_bits),
        "dpd_output": (2, activation_bits),
    }
    for index, layer in enumerate(layers):
        expected_golden[f"conv{index}_input"] = (layer["in_channels"], activation_bits)
        expected_golden[f"conv{index}_accumulator"] = (
            layer["out_channels"],
            layer["accumulator"]["minimum_signed_bits_from_full_code_range"],
        )
        if layer.get("hardswish_input") is not None:
            expected_golden[f"conv{index}_hardswish_input"] = (
                layer["out_channels"], activation_bits
            )
    if set(files) != set(expected_golden):
        raise ValueError("golden trace set is incomplete or contains unknown traces")
    for name, (channels, bits) in expected_golden.items():
        info = files[name]
        if info.get("shape") != [length, channels]:
            raise ValueError(f"golden {name} shape mismatch")
        if info.get("layout") != "time_major_channel_minor":
            raise ValueError(f"golden {name} layout mismatch")
        artifact, _ = _memory(
            root, info, label=f"golden {name}", prefix="golden_vectors",
            expected_count=length * channels, expected_bits=bits,
        )
        artifacts.append(artifact)
    return manifest, artifacts


__all__ = [
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "ROUNDING_MODE",
    "DISCARD_LSB_SIGNED_FLOOR",
    "validate_manifest_v1",
]
