"""RTL export utilities for full-I/O QAT FExLite causal TCN models.

The exporter deliberately records both the trained fake-quantization model and
an explicit integer numeric contract.  The latter closes the implementation
choices that PyTorch leaves in floating point (bias, FEx, Hardswish, and the
residual add) and is the contract an RTL implementation should reproduce.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from models import CoreModel
from quant import get_quant_model
from quant.qmodules.quant_layers import INT_Conv1D
from quant.rtl_manifest import validate_manifest_v1


EXPORT_FORMAT_VERSION = 1
ROUNDING_MODE = "round_to_nearest_ties_to_even"
RTL_SPEC_VERSION = 1
LEGACY_DILATION_BASE = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path_hint(path: str | Path, root: Path) -> str:
    """Return a stable, non-absolute provenance hint.

    Hashes, rather than host paths, identify inputs.  Paths below the checkout
    remain checkout-relative; external artifacts use only their basename.
    """

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _git_metadata(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = command("git", "rev-parse", "HEAD")
    status = command("git", "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "note": (
            "A dirty tree is not identified by commit alone; source_file_sha256 "
            "below is authoritative for the export."
        ),
    }


def _effective_scale(quantizer) -> tuple[float, int]:
    with torch.no_grad():
        scale, _ = quantizer.round_scale2pow2(quantizer.scale.detach())
    value = float(scale.item())
    exponent = int(round(math.log2(value)))
    if value != 2.0**exponent:
        raise ValueError(f"quantizer scale is not a power of two: {value}")
    return value, exponent


def _quantizer_record(quantizer) -> dict[str, Any]:
    scale, exponent = _effective_scale(quantizer)
    return {
        "bits": int(quantizer.bits),
        "signed": not bool(quantizer.all_positive),
        "qmin": int(quantizer.Qn),
        "qmax": int(quantizer.Qp),
        "scale": scale,
        "scale_exponent": exponent,
        "zero_point": 0,
        "rounding": ROUNDING_MODE,
        "overflow": "saturate_at_explicit_quantizer_output",
    }


def quantize_codes(value: torch.Tensor, scale: float, qmin: int, qmax: int) -> torch.Tensor:
    return torch.round(value.detach().cpu() / scale).clamp(qmin, qmax).to(torch.int64)


def round_divide_even(numerator: torch.Tensor, denominator: int) -> torch.Tensor:
    """Signed integer division with round-to-nearest, ties-to-even."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = torch.where(numerator < 0, -1, 1)
    magnitude = numerator.abs()
    quotient = torch.div(magnitude, denominator, rounding_mode="floor")
    remainder = magnitude.remainder(denominator)
    twice = remainder * 2
    increment = (twice > denominator) | ((twice == denominator) & (quotient.remainder(2) == 1))
    return sign * (quotient + increment.to(torch.int64))


def requantize_pow2(
    numerator: torch.Tensor,
    source_exponent: int,
    target_exponent: int,
    divisor: int = 1,
) -> torch.Tensor:
    shift = int(source_exponent) - int(target_exponent)
    if shift >= 0:
        return round_divide_even(numerator << shift, divisor)
    return round_divide_even(numerator, divisor << (-shift))


def saturate(value: torch.Tensor, bits: int, signed: bool = True) -> torch.Tensor:
    qmin = -(2 ** (bits - 1)) if signed else 0
    qmax = 2 ** (bits - int(signed)) - 1
    return value.clamp(qmin, qmax).to(torch.int64)


def _write_mem(path: Path, values: torch.Tensor, bits: int) -> dict[str, Any]:
    values = values.detach().cpu().to(torch.int64).reshape(-1)
    hex_digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    with path.open("w") as handle:
        for value in values.tolist():
            handle.write(f"{value & mask:0{hex_digits}X}\n")
    return {
        "path": str(path.name),
        "count": int(values.numel()),
        "bits": bits,
        "encoding": "two_complement_hex_one_word_per_line",
        "sha256": sha256_file(path),
    }


def _write_decimal_csv(path: Path, values: torch.Tensor) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flat_index", "signed_integer"])
        for index, value in enumerate(values.detach().cpu().to(torch.int64).reshape(-1).tolist()):
            writer.writerow([index, value])


def _read_mem(path: Path, bits: int, shape: list[int]) -> torch.Tensor:
    words = [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]
    sign_bit = 1 << (bits - 1)
    modulus = 1 << bits
    signed = [word - modulus if word & sign_bit else word for word in words]
    expected = math.prod(shape)
    if len(signed) != expected:
        raise ValueError(f"{path}: expected {expected} words, found {len(signed)}")
    return torch.tensor(signed, dtype=torch.int64).reshape(shape)


def _infer_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Recover the QAT topology without relying on a filename convention.

    New checkpoints carry ``backbone._rtl_spec = [version, L, K, base]``.
    Legacy FExLite checkpoints predate that buffer.  Their convolution shapes
    still recover H/L/K and the legacy model family used dilation base two.
    """

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("QAT checkpoint must contain a state_dict mapping")

    weight_pattern = re.compile(r"backbone\.network\.(\d+)\.weight$")
    convolutions: list[tuple[int, str, torch.Tensor]] = []
    for key, value in state.items():
        match = weight_pattern.search(key)
        if match is not None and isinstance(value, torch.Tensor):
            convolutions.append((int(match.group(1)), key, value))
    convolutions.sort(key=lambda item: item[0])
    if len(convolutions) < 3:
        raise ValueError(
            "FExLite causal TCN checkpoint must contain input, temporal, and output Conv1d weights"
        )

    first_index, first_key, first_weight = convolutions[0]
    _, _, output_weight = convolutions[-1]
    temporal_weights = [value for _, _, value in convolutions[1:-1]]
    if first_index != 0 or list(first_weight.shape[1:]) != [6, 1]:
        raise ValueError(
            f"invalid FExLite input projection shape: {tuple(first_weight.shape)}"
        )
    hidden = int(first_weight.shape[0])
    if list(output_weight.shape) != [2, hidden, 1]:
        raise ValueError(
            f"invalid FExLite output projection shape: {tuple(output_weight.shape)}"
        )
    if not temporal_weights:
        raise ValueError("FExLite causal TCN requires at least one temporal layer")
    kernel_size = int(temporal_weights[0].shape[-1])
    expected_temporal_shape = [hidden, 1, kernel_size]
    if any(list(weight.shape) != expected_temporal_shape for weight in temporal_weights):
        shapes = [list(weight.shape) for weight in temporal_weights]
        raise ValueError(f"inconsistent FExLite temporal weight shapes: {shapes}")
    num_layers = len(temporal_weights)

    first_prefix = first_key[: -len(".weight")]
    try:
        weight_bits = int(state[f"{first_prefix}.n_bits_w"].item())
        activation_bits = int(state[f"{first_prefix}.n_bits_a"].item())
    except (KeyError, RuntimeError) as exc:
        raise ValueError("checkpoint is missing QAT precision buffers") from exc

    spec_items = [
        value for key, value in state.items()
        if key.endswith("backbone._rtl_spec")
    ]
    if len(spec_items) > 1:
        raise ValueError("checkpoint contains multiple FExLite _rtl_spec buffers")
    if spec_items:
        spec_values = [int(value) for value in spec_items[0].reshape(-1).tolist()]
        if len(spec_values) != 4:
            raise ValueError("FExLite _rtl_spec must contain [version, L, K, dilation_base]")
        version, spec_layers, spec_kernel, dilation_base = spec_values
        if version != RTL_SPEC_VERSION:
            raise ValueError(f"unsupported FExLite _rtl_spec version: {version}")
        if (spec_layers, spec_kernel) != (num_layers, kernel_size):
            raise ValueError(
                "FExLite _rtl_spec disagrees with Conv1d shapes: "
                f"spec L/K={spec_layers}/{spec_kernel}, weights L/K={num_layers}/{kernel_size}"
            )
        topology_source = "checkpoint_rtl_spec"
    else:
        # Every pre-_rtl_spec FExLite causal TCN used [1, 2, 4, 8]
        # dilations.  This explicitly covers the canonical H10/L4/K5 model
        # while also allowing legacy test-width checkpoints from that family.
        dilation_base = LEGACY_DILATION_BASE
        topology_source = "legacy_conv_shapes_with_base2"

    if dilation_base < 1:
        raise ValueError("dilation_base must be positive")
    return {
        "state": state,
        "hidden_channels": hidden,
        "num_layers": num_layers,
        "kernel_size": kernel_size,
        "dilation_base": dilation_base,
        "dilations": [dilation_base**index for index in range(num_layers)],
        "activation_bits": activation_bits,
        "weight_bits": weight_bits,
        "topology_source": topology_source,
        "legacy_canonical_h10_l4_k5": (
            not spec_items and hidden == 10 and num_layers == 4 and kernel_size == 5
        ),
    }


def load_qat_model(checkpoint: str | Path):
    checkpoint = Path(checkpoint)
    topology = _infer_checkpoint(checkpoint)
    hidden = topology["hidden_channels"]
    activation_bits = topology["activation_bits"]
    weight_bits = topology["weight_bits"]
    project = SimpleNamespace(
        quant=True,
        n_bits_w=weight_bits,
        n_bits_a=activation_bits,
        pretrained_model="",
        quant_dir_label="rtl_export",
        DPD_backbone="fexlite_causal_tcn",
        quant_calibration_batches=0,
        quant_calibration_quantile=0.9999,
    )
    model = get_quant_model(
        project,
        CoreModel(
            2,
            hidden,
            topology["num_layers"],
            "fexlite_causal_tcn",
            tcn_kernel_size=topology["kernel_size"],
            tcn_dilation_base=topology["dilation_base"],
        ),
    )
    incompatible = model.load_state_dict(topology["state"], strict=False)
    allowed_missing = {
        key for key in model.state_dict()
        if key.endswith("backbone._rtl_spec")
        and topology["topology_source"] == "legacy_conv_shapes_with_base2"
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "QAT checkpoint is incompatible with the recovered topology: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    expected_boundary_scale = 2.0 ** (1 - activation_bits)
    legacy_boundary_normalized = False
    for name, quantizer in (
        ("raw_input", model.input_quantizer),
        ("dpd_output", model.output_quantizer),
    ):
        stored_scale = float(quantizer.scale.detach().item())
        stored_effective = 2.0 ** int(round(math.log2(abs(stored_scale))))
        if stored_scale != expected_boundary_scale:
            if (
                topology["topology_source"] == "legacy_conv_shapes_with_base2"
                and topology["legacy_canonical_h10_l4_k5"]
                and stored_effective == expected_boundary_scale
            ):
                # Historical QAT exposed boundary scales to the optimizer even
                # though their rounded physical grid stayed at 2^(1-A).  Restore
                # that exact buffer value for legacy H10 compatibility only.
                quantizer.scale.fill_(expected_boundary_scale)
                legacy_boundary_normalized = True
            else:
                raise ValueError(
                    f"{name} checkpoint scale must be exactly 2^(1-A)="
                    f"{expected_boundary_scale}, got {stored_scale}"
                )
    model.assert_physical_io_scales()
    model.eval()
    topology["legacy_boundary_scale_normalized"] = legacy_boundary_normalized
    topology = {key: value for key, value in topology.items() if key != "state"}
    return model, topology


def _layer_kind(index: int, layer_count: int) -> str:
    if index == 0:
        return "input_projection"
    if index == layer_count - 1:
        return "output_projection"
    return "causal_depthwise"


def _layer_name(index: int, layer_count: int) -> str:
    if index == 0:
        return "input_projection"
    if index == layer_count - 1:
        return "output_projection"
    return f"dw{index}"


def _minimum_signed_bits(maximum_absolute: int) -> int:
    return max(2, int(maximum_absolute).bit_length() + 1)


def _quantize_bias_v1(module, name: str, accumulator_exponent: int):
    if module.bias is None:
        return None
    if not torch.isfinite(module.bias.detach()).all():
        raise ValueError(f"{name} bias contains a non-finite value")
    bias_scale = 2.0**accumulator_exponent
    codes = torch.round(module.bias.detach().cpu() / bias_scale).to(torch.int64)
    minimum = int(codes.min().item())
    maximum = int(codes.max().item())
    if minimum < -(2**31) or maximum > 2**31 - 1:
        raise OverflowError(
            f"{name} bias cannot fit the TCN-Compiler v1 signed 32-bit contract: "
            f"range=[{minimum}, {maximum}]"
        )
    return codes


def _extract_layers(
    model,
    weights_dir: Path,
    topology: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modules = [module for module in model.modules() if isinstance(module, INT_Conv1D)]
    expected_count = int(topology["num_layers"]) + 2
    if len(modules) != expected_count:
        raise ValueError(
            f"FExLite causal TCN must contain {expected_count} Conv1d layers, got {len(modules)}"
        )

    # Reject an incompatible checkpoint before creating any weight/bias memory.
    for index, module in enumerate(modules):
        activation_quant = _quantizer_record(module.act_quantizer)
        weight_quant = _quantizer_record(module.weight_quantizer)
        accumulator_exponent = (
            activation_quant["scale_exponent"] + weight_quant["scale_exponent"]
        )
        _quantize_bias_v1(
            module, _layer_name(index, len(modules)), accumulator_exponent
        )

    records = []
    runtime = []
    for index, module in enumerate(modules):
        name = _layer_name(index, len(modules))
        weight_quant = _quantizer_record(module.weight_quantizer)
        activation_quant = _quantizer_record(module.act_quantizer)
        weight_codes = quantize_codes(
            module.weight, weight_quant["scale"], weight_quant["qmin"], weight_quant["qmax"]
        )
        mem_path = weights_dir / f"{name}_weights.mem"
        mem = _write_mem(mem_path, weight_codes, weight_quant["bits"])
        _write_decimal_csv(weights_dir / f"{name}_weights.csv", weight_codes)

        kernel = int(module.kernel_size[0])
        dilation = int(module.dilation[0])
        delay_by_kernel_index = [(kernel - 1 - k) * dilation for k in range(kernel)]
        fan_in = int(module.in_channels // module.groups) * kernel
        product_bound = fan_in * max(abs(weight_quant["qmin"]), weight_quant["qmax"]) * max(
            abs(activation_quant["qmin"]), activation_quant["qmax"]
        )
        accumulator_exponent = activation_quant["scale_exponent"] + weight_quant["scale_exponent"]

        bias_codes = None
        bias_record = None
        if module.bias is not None:
            bias_scale = 2.0**accumulator_exponent
            bias_codes = _quantize_bias_v1(module, name, accumulator_exponent)
            bias_width = 32
            bias_path = weights_dir / f"{name}_bias.mem"
            bias_mem = _write_mem(bias_path, bias_codes, bias_width)
            _write_decimal_csv(weights_dir / f"{name}_bias.csv", bias_codes)
            reconstructed = bias_codes.to(torch.float64) * bias_scale
            error = reconstructed - module.bias.detach().cpu().to(torch.float64)
            bias_record = {
                "source": "FP32 QAT bias quantized to accumulator LSB",
                "scale": bias_scale,
                "scale_exponent": accumulator_exponent,
                "bits": bias_width,
                "values": bias_codes.tolist(),
                "max_abs_quantization_error": float(error.abs().max().item()),
                "mem": bias_mem,
            }

        max_accumulator = product_bound + (
            int(bias_codes.abs().max().item()) if bias_codes is not None else 0
        )
        record = {
            "index": index,
            "name": name,
            "kind": _layer_kind(index, len(modules)),
            "in_channels": int(module.in_channels),
            "out_channels": int(module.out_channels),
            "groups": int(module.groups),
            "kernel_size": kernel,
            "dilation": dilation,
            "stride": int(module.stride[0]),
            "causal_delay_by_pytorch_kernel_index": delay_by_kernel_index,
            "weight_shape_oik": list(module.weight.shape),
            "weight_flatten_order": "out_channel,input_channel_per_group,kernel_index",
            "weight": {**weight_quant, "mem": mem},
            "input_activation": activation_quant,
            "accumulator": {
                "scale": 2.0**accumulator_exponent,
                "scale_exponent": accumulator_exponent,
                "saturation": "none_before_following_requantizer",
                "minimum_signed_bits_from_full_code_range": _minimum_signed_bits(max_accumulator),
                "maximum_absolute_bound_in_accumulator_codes": max_accumulator,
            },
            "bias": bias_record,
            "followed_by_hardswish": index < len(modules) - 1,
        }
        records.append(record)
        runtime.append({
            "module": module,
            "weight_codes": weight_codes,
            "bias_codes": bias_codes,
            "record": record,
        })

    actual_dilations = [
        int(module.dilation[0]) for module in modules[1:-1]
    ]
    if actual_dilations != list(topology["dilations"]):
        raise ValueError(
            "instantiated temporal dilations disagree with checkpoint topology: "
            f"actual={actual_dilations}, expected={topology['dilations']}"
        )
    return records, runtime


def _causal_conv_integer(x: torch.Tensor, layer: dict[str, Any]) -> torch.Tensor:
    """Reference causal cross-correlation for time-major integer activations."""
    module = layer["module"]
    weights = layer["weight_codes"]
    bias = layer["bias_codes"]
    time_steps = int(x.shape[0])
    output = torch.zeros((time_steps, module.out_channels), dtype=torch.int64)
    kernel = int(module.kernel_size[0])
    dilation = int(module.dilation[0])
    depthwise = module.groups == module.in_channels == module.out_channels
    for t in range(time_steps):
        for out_channel in range(module.out_channels):
            value = int(bias[out_channel].item()) if bias is not None else 0
            input_channels = [out_channel] if depthwise else range(module.in_channels)
            for local_input, input_channel in enumerate(input_channels):
                weight_input = 0 if depthwise else local_input
                for kernel_index in range(kernel):
                    delay = (kernel - 1 - kernel_index) * dilation
                    source_time = t - delay
                    if source_time >= 0:
                        value += int(x[source_time, input_channel]) * int(
                            weights[out_channel, weight_input, kernel_index]
                        )
            output[t, out_channel] = value
    return output


def _hardswish_requantize(
    accumulator: torch.Tensor,
    accumulator_exponent: int,
    target: dict[str, Any],
) -> torch.Tensor:
    if accumulator_exponent > 0:
        raise ValueError("positive accumulator exponent is not supported by the integer contract")
    threshold = 3 << (-accumulator_exponent)
    result = torch.empty_like(accumulator)
    low = accumulator <= -threshold
    high = accumulator >= threshold
    middle = ~(low | high)
    result[low] = 0
    result[high] = requantize_pow2(
        accumulator[high], accumulator_exponent, target["scale_exponent"]
    )
    middle_acc = accumulator[middle]
    numerator = middle_acc * (middle_acc + threshold)
    result[middle] = requantize_pow2(
        numerator,
        2 * accumulator_exponent,
        target["scale_exponent"],
        divisor=6,
    )
    return result.clamp(target["qmin"], target["qmax"]).to(torch.int64)


def _fex_integer(input_codes: torch.Tensor, input_exponent: int, target: dict[str, Any]) -> torch.Tensor:
    i_codes = input_codes[:, 0]
    q_codes = input_codes[:, 1]
    power = i_codes.square() + q_codes.square()
    numerators = [
        (i_codes, input_exponent),
        (q_codes, input_exponent),
        (power, 2 * input_exponent),
        (power.square(), 4 * input_exponent),
        (i_codes * power, 3 * input_exponent),
        (q_codes * power, 3 * input_exponent),
    ]
    features = [
        requantize_pow2(value, exponent, target["scale_exponent"])
        for value, exponent in numerators
    ]
    return torch.stack(features, dim=1).clamp(target["qmin"], target["qmax"]).to(torch.int64)


def _residual_requantize(
    correction: torch.Tensor,
    correction_exponent: int,
    residual: torch.Tensor,
    residual_exponent: int,
    target: dict[str, Any],
) -> torch.Tensor:
    common_exponent = min(correction_exponent, residual_exponent)
    combined = (correction << (correction_exponent - common_exponent)) + (
        residual << (residual_exponent - common_exponent)
    )
    result = requantize_pow2(combined, common_exponent, target["scale_exponent"])
    return result.clamp(target["qmin"], target["qmax"]).to(torch.int64)


def run_integer_reference(
    input_codes: torch.Tensor,
    layers: list[dict[str, Any]],
    input_quantizer: dict[str, Any],
    output_quantizer: dict[str, Any],
) -> dict[str, torch.Tensor]:
    input_codes = input_codes.detach().cpu().to(torch.int64)
    traces: dict[str, torch.Tensor] = {"raw_input": input_codes}
    activation = _fex_integer(
        input_codes,
        input_quantizer["scale_exponent"],
        layers[0]["record"]["input_activation"],
    )
    traces["conv0_input"] = activation
    for index, layer in enumerate(layers):
        accumulator = _causal_conv_integer(activation, layer)
        traces[f"conv{index}_accumulator"] = accumulator
        record = layer["record"]
        if index < len(layers) - 1:
            target = layers[index + 1]["record"]["input_activation"]
            activation = _hardswish_requantize(
                accumulator, record["accumulator"]["scale_exponent"], target
            )
            traces[f"conv{index + 1}_input"] = activation
        else:
            traces["dpd_output"] = _residual_requantize(
                accumulator,
                record["accumulator"]["scale_exponent"],
                input_codes,
                input_quantizer["scale_exponent"],
                output_quantizer,
            )
    return traces


def load_exported_integer_runtime(manifest_path: str | Path):
    """Load the integer model from `.mem`; the QAT checkpoint is not required."""
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest, _ = validate_manifest_v1(manifest_path)
    runtime_layers = []
    for record in manifest["model"]["layers"]:
        module = SimpleNamespace(
            in_channels=record["in_channels"],
            out_channels=record["out_channels"],
            groups=record["groups"],
            kernel_size=(record["kernel_size"],),
            dilation=(record["dilation"],),
        )
        weight_info = record["weight"]["mem"]
        weights = _read_mem(
            root / "weights" / weight_info["path"],
            weight_info["bits"],
            record["weight_shape_oik"],
        )
        bias = None
        if record["bias"] is not None:
            bias_info = record["bias"]["mem"]
            bias = _read_mem(
                root / "weights" / bias_info["path"],
                bias_info["bits"],
                [record["out_channels"]],
            )
        runtime_layers.append({
            "module": module,
            "weight_codes": weights,
            "bias_codes": bias,
            "record": record,
        })
    return manifest, runtime_layers


def verify_exported_golden(manifest_path: str | Path) -> dict[str, Any]:
    """Recompute all exported traces from `.mem` files and require 0 LSB."""
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest, layers = load_exported_integer_runtime(manifest_path)
    files = manifest["golden_vectors"]["files"]
    raw_info = files["raw_input"]
    raw_input = _read_mem(
        root / raw_info["path"], raw_info["bits"], raw_info["shape"]
    )
    traces = run_integer_reference(
        raw_input,
        layers,
        manifest["quantization"]["raw_input"],
        manifest["quantization"]["dpd_output"],
    )
    result = {}
    for name, expected in traces.items():
        info = files[name]
        stored = _read_mem(root / info["path"], info["bits"], info["shape"])
        maximum = int((expected - stored).abs().max().item())
        result[name] = {"maximum_absolute_lsb_error": maximum, "match": maximum == 0}
    if not all(item["match"] for item in result.values()):
        raise AssertionError(f"exported golden verification failed: {result}")
    return result


def _fake_quant_traces(model, input_float: torch.Tensor) -> dict[str, torch.Tensor]:
    traces: dict[str, torch.Tensor] = {}
    handles = []
    input_record = _quantizer_record(model.input_quantizer)
    output_record = _quantizer_record(model.output_quantizer)

    def capture(name: str, scale: float, channel_first: bool):
        def hook(_module, _inputs, output):
            tensor = output.detach().cpu()
            if channel_first:
                tensor = tensor.transpose(1, 2)
            traces[name] = torch.round(tensor[0] / scale).to(torch.int64)
        return hook

    handles.append(model.input_quantizer.register_forward_hook(
        capture("raw_input", input_record["scale"], channel_first=False)
    ))
    conv_index = 0
    for module in model.modules():
        if isinstance(module, INT_Conv1D):
            activation_record = _quantizer_record(module.act_quantizer)
            handles.append(module.act_quantizer.register_forward_hook(
                capture(f"conv{conv_index}_input", activation_record["scale"], channel_first=True)
            ))
            conv_index += 1
    handles.append(model.output_quantizer.register_forward_hook(
        capture("dpd_output", output_record["scale"], channel_first=False)
    ))
    try:
        with torch.no_grad():
            model(input_float.unsqueeze(0))
    finally:
        for handle in handles:
            handle.remove()
    return traces


def _comparison(integer: dict[str, torch.Tensor], fake: dict[str, torch.Tensor]) -> dict[str, Any]:
    result = {}
    for name in sorted(set(integer) & set(fake)):
        difference = integer[name].to(torch.int64) - fake[name].to(torch.int64)
        result[name] = {
            "shape": list(integer[name].shape),
            "exact_match_fraction": float((difference == 0).to(torch.float64).mean().item()),
            "maximum_absolute_lsb_error": int(difference.abs().max().item()),
            "mean_absolute_lsb_error": float(difference.abs().to(torch.float64).mean().item()),
        }
    return result


def _default_input(length: int, input_quantizer: dict[str, Any], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    limit = min(1024, input_quantizer["qmax"])
    codes = torch.randint(-limit, limit + 1, (length, 2), generator=generator)
    if length >= 4:
        codes[0] = 0
        codes[1] = torch.tensor([limit, 0])
        codes[2] = torch.tensor([0, -limit])
        codes[3] = torch.tensor([limit, -limit])
    return codes.to(torch.float32) * input_quantizer["scale"]


def _load_input(path: str | Path, start: int, length: int) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
        value = torch.as_tensor(value)
    elif path.suffix == ".npy":
        import numpy as np
        value = torch.from_numpy(np.load(path))
    elif path.suffix == ".csv":
        import pandas as pd
        frame = pd.read_csv(path)
        value = torch.tensor(frame.iloc[:, :2].to_numpy())
    else:
        raise ValueError("golden input must be .csv, .npy, or .pt")
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"golden input must have shape [time, 2], got {tuple(value.shape)}")
    return value[start:start + length].to(torch.float32)


def _source_hashes(root: Path) -> dict[str, str]:
    relative_paths = [
        "backbones/fexlite_causal_tcn.py",
        "models.py",
        "quant/quant_envs.py",
        "quant/qmodules/quant_layers.py",
        "quant/qmodules/quantizers.py",
        "quant/rtl_export.py",
        "quant/rtl_manifest.py",
    ]
    return {
        path: sha256_file(root / path)
        for path in relative_paths
        if (root / path).exists()
    }


def _model_effective_quantizer_scales(model) -> dict[str, dict[str, float | int]]:
    result = {}
    for name, value in sorted(model.state_dict().items()):
        if not name.endswith("quantizer.scale"):
            continue
        raw_scale = float(torch.as_tensor(value).reshape(-1)[0].item())
        if not math.isfinite(raw_scale) or raw_scale == 0.0:
            raise ValueError(f"invalid quantizer scale {name}: {raw_scale}")
        exponent = int(round(math.log2(abs(raw_scale))))
        result[name] = {
            "effective_scale": 2.0**exponent,
            "scale_exponent": exponent,
        }
    return result


def _read_and_validate_qat_sidecars(
    checkpoint: Path,
    model,
    topology: dict[str, Any],
) -> dict[str, Any]:
    """Validate optional training metadata and bind it to this checkpoint."""
    calibration_path = checkpoint.with_suffix(".calibration.json")
    model_spec_path = checkpoint.with_suffix(".model_spec.json")
    checkpoint_sha = sha256_file(checkpoint)
    records: dict[str, Any] = {}

    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text())
        if calibration.get("format") != "opendpd_tcn_qat_calibration":
            raise ValueError("unsupported QAT calibration sidecar format")
        if calibration.get("format_version") != 1:
            raise ValueError("unsupported QAT calibration sidecar version")
        if calibration.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("calibration sidecar checkpoint SHA mismatch")
        if int(calibration.get("activation_bits", -1)) != int(topology["activation_bits"]):
            raise ValueError("calibration sidecar activation precision mismatch")
        if int(calibration.get("weight_bits", -1)) != int(topology["weight_bits"]):
            raise ValueError("calibration sidecar weight precision mismatch")
        quantile = calibration.get("quantile")
        maximum_batches = calibration.get("maximum_batches")
        if not isinstance(quantile, (int, float)) or not 0.0 < float(quantile) <= 1.0:
            raise ValueError("calibration sidecar quantile is invalid")
        if (
            isinstance(maximum_batches, bool)
            or not isinstance(maximum_batches, int)
            or maximum_batches < 1
        ):
            raise ValueError("calibration sidecar maximum_batches is invalid")
        calibrated = calibration.get("calibration_quantizers")
        expected_calibration_names = {
            "raw_input", "dpd_output",
            *{
                f"conv{index}_input"
                for index in range(int(topology["num_layers"]) + 2)
            },
        }
        if not isinstance(calibrated, dict) or set(calibrated) != expected_calibration_names:
            raise ValueError("calibration sidecar quantizer set is incomplete")
        expected_boundary = 2.0 ** (1 - int(topology["activation_bits"]))
        for name in ("raw_input", "dpd_output"):
            if (
                calibrated[name].get("bits") != int(topology["activation_bits"])
                or calibrated[name].get("scale") != expected_boundary
                or calibrated[name].get("policy") != "fixed_signed_unit_interface"
            ):
                raise ValueError(f"calibration sidecar {name} contract mismatch")
        for index in range(int(topology["num_layers"]) + 2):
            name = f"conv{index}_input"
            record = calibrated[name]
            scale = record.get("scale")
            if (
                record.get("bits") != int(topology["activation_bits"])
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0.0
                or float(scale) != 2.0 ** round(math.log2(float(scale)))
                or record.get("quantile") != quantile
                or not 1 <= int(record.get("batches", 0)) <= maximum_batches
            ):
                raise ValueError(f"calibration sidecar {name} contract mismatch")
        declared_scales = calibration.get("final_effective_quantizers")
        actual_scales = _model_effective_quantizer_scales(model)
        if declared_scales != actual_scales:
            raise ValueError("calibration sidecar final quantizer scales mismatch")
        records["calibration"] = {
            "status": "validated",
            "path_hint": calibration_path.name,
            "sha256": sha256_file(calibration_path),
            "format": calibration["format"],
            "format_version": calibration["format_version"],
            "checkpoint_sha256": calibration["checkpoint_sha256"],
        }
    else:
        records["calibration"] = {"status": "unavailable"}

    if model_spec_path.is_file():
        model_spec = json.loads(model_spec_path.read_text())
        if model_spec.get("format") != "opendpd_fexlite_causal_tcn_spec":
            raise ValueError("unsupported TCN model-spec sidecar format")
        if model_spec.get("format_version") != 1:
            raise ValueError("unsupported TCN model-spec sidecar version")
        expected = {
            "hidden_channels": int(topology["hidden_channels"]),
            "temporal_layers": int(topology["num_layers"]),
            "kernel_size": int(topology["kernel_size"]),
            "dilation_base": int(topology["dilation_base"]),
            "dilations": list(topology["dilations"]),
            "activation": "hardswish",
        }
        for name, value in expected.items():
            if model_spec.get(name) != value:
                raise ValueError(f"model-spec sidecar {name} mismatch")
        records["model_spec"] = {
            "status": "validated",
            "path_hint": model_spec_path.name,
            "sha256": sha256_file(model_spec_path),
            "format": model_spec["format"],
            "format_version": model_spec["format_version"],
        }
    else:
        records["model_spec"] = {"status": "unavailable"}

    if not calibration_path.is_file() and not model_spec_path.is_file():
        if topology["legacy_canonical_h10_l4_k5"]:
            records["status"] = "unavailable_legacy"
            records["reason"] = "checkpoint predates calibration/model-spec sidecars"
            records["calibration"]["status"] = "unavailable_legacy"
            records["model_spec"]["status"] = "unavailable_legacy"
        else:
            records["status"] = "unavailable"
    elif calibration_path.is_file() and model_spec_path.is_file():
        records["status"] = "validated"
    else:
        records["status"] = "validated_partial"
    return records


def _export_fexlite_qat_rtl_into(
    checkpoint: str | Path,
    output_dir: str | Path,
    pa_checkpoint: str | Path | None = None,
    input_path: str | Path | None = None,
    golden_start: int = 0,
    golden_length: int = 256,
    golden_seed: int = 2026,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    golden_dir = output_dir / "golden_vectors"
    weights_dir.mkdir(exist_ok=True)
    golden_dir.mkdir(exist_ok=True)

    root = Path(__file__).resolve().parent.parent
    model, topology = load_qat_model(checkpoint)
    hidden = int(topology["hidden_channels"])
    activation_bits = int(topology["activation_bits"])
    weight_bits = int(topology["weight_bits"])
    input_quantizer = _quantizer_record(model.input_quantizer)
    output_quantizer = _quantizer_record(model.output_quantizer)
    sidecar_record = _read_and_validate_qat_sidecars(
        checkpoint, model, topology
    )
    layer_records, runtime_layers = _extract_layers(
        model, weights_dir, topology
    )

    if input_path:
        input_float = _load_input(input_path, golden_start, golden_length)
        input_source = {
            "kind": "file",
            "path_hint": _portable_path_hint(input_path, root),
            "sha256": sha256_file(input_path),
            "start": golden_start,
        }
    else:
        input_float = _default_input(golden_length, input_quantizer, golden_seed)
        input_source = {"kind": "deterministic_synthetic", "seed": golden_seed}
    if int(input_float.shape[0]) < 1:
        raise ValueError("golden input selection is empty")

    input_codes = quantize_codes(
        input_float, input_quantizer["scale"], input_quantizer["qmin"], input_quantizer["qmax"]
    )
    integer_traces = run_integer_reference(
        input_codes, runtime_layers, input_quantizer, output_quantizer
    )
    fake_traces = _fake_quant_traces(model, input_float)
    comparison = _comparison(integer_traces, fake_traces)

    golden_files = {}
    for name, values in integer_traces.items():
        bits = activation_bits
        if name.endswith("accumulator"):
            layer_index = int(name[len("conv"):name.index("_")])
            bits = layer_records[layer_index]["accumulator"]["minimum_signed_bits_from_full_code_range"]
        path = golden_dir / f"{name}.mem"
        golden_files[name] = {
            **_write_mem(path, values, bits),
            "path": str(path.relative_to(output_dir)),
            "shape": list(values.shape),
            "layout": "time_major_channel_minor",
        }
        _write_decimal_csv(golden_dir / f"{name}.csv", values)

    total_weights = sum(record["weight"]["mem"]["count"] for record in layer_records)
    historical_values = sum(
        record["out_channels"] * max(record["causal_delay_by_pytorch_kernel_index"])
        for record in layer_records if record["kind"] == "causal_depthwise"
    )
    window_values = sum(
        record["out_channels"] * (max(record["causal_delay_by_pytorch_kernel_index"]) + 1)
        for record in layer_records if record["kind"] == "causal_depthwise"
    )

    pa_record = None
    if pa_checkpoint is not None:
        pa_path = Path(pa_checkpoint).resolve()
        pa_record = {
            "path_hint": _portable_path_hint(pa_path, root),
            "sha256": sha256_file(pa_path),
        }

    receptive_field = 1 + sum(
        (int(record["kernel_size"]) - 1) * int(record["dilation"])
        for record in layer_records
        if record["kind"] == "causal_depthwise"
    )

    manifest = {
        "format": "opendpd_fexlite_qat_rtl_export",
        "format_version": EXPORT_FORMAT_VERSION,
        "status": {
            "structure_export_complete": True,
            "integer_numeric_contract_complete": True,
            "fake_qat_compatibility_measured": True,
            "full_dataset_pa_metric_revalidation_complete": False,
            "rtl_synthesis_constraints_complete": False,
        },
        "provenance": {
            "dpd_checkpoint": {
                "path_hint": _portable_path_hint(checkpoint, root),
                "sha256": sha256_file(checkpoint),
            },
            "pa_checkpoint": pa_record,
            "git": _git_metadata(root),
            "source_file_sha256": _source_hashes(root),
            "training_sidecars": sidecar_record,
            "dataset_name": dataset_name,
            "topology_recovery": {
                "source": topology["topology_source"],
                "rtl_spec_version": (
                    RTL_SPEC_VERSION
                    if topology["topology_source"] == "checkpoint_rtl_spec"
                    else None
                ),
                "legacy_canonical_h10_l4_k5": bool(
                    topology["legacy_canonical_h10_l4_k5"]
                ),
            },
            "input_normalization": (
                "No normalization is applied by modules.data_collector.load_dataset; "
                "CSV I/Q values enter the signed raw-input quantizer directly."
            ),
        },
        "model": {
            "backbone": "fexlite_causal_tcn",
            "hidden_channels": hidden,
            "input_channels": 2,
            "feature_channels": 6,
            "output_channels": 2,
            "feature_order": ["I", "Q", "p=I^2+Q^2", "p^2", "I*p", "Q*p"],
            "layers": layer_records,
            "temporal_layers": int(topology["num_layers"]),
            "kernel_size": int(topology["kernel_size"]),
            "dilation_base": int(topology["dilation_base"]),
            "dilations": list(topology["dilations"]),
            "activation": "hardswish after input projection and each depthwise layer",
            "output_residual": "quantized correction + quantized raw I/Q, then output requantization",
            "receptive_field_samples": receptive_field,
            "mac_per_sample": total_weights,
            "weight_count": total_weights,
            "bias_count": sum(record["out_channels"] for record in layer_records if record["bias"]),
        },
        "quantization": {
            "raw_input": input_quantizer,
            "dpd_output": output_quantizer,
            "numeric_contract": {
                "integer_encoding": "signed two_complement unless signed=false",
                "rounding": ROUNDING_MODE,
                "fex": "exact integer powers/products followed by layer-0 activation requantization",
                "bias": "round FP32 QAT bias to input_scale*weight_scale",
                "mac": "full-precision signed integer accumulation without intermediate saturation",
                "hardswish": "exact x*clamp(x+3,0,6)/6 rational evaluation followed by next-layer requantization",
                "residual": "align correction and raw-I/Q power-of-two scales, add once, then output requantize",
                "saturation": (
                    f"only at explicit {activation_bits}-bit activation/output quantizers"
                ),
            },
        },
        "hardware_cost": {
            "raw_weight_bits": total_weights * weight_bits,
            "weight_buffer_bits": total_weights * weight_bits,
            "weight_storage_copies": 1,
            "physical_weight_state_bits": total_weights * weight_bits,
            "historical_activation_values": historical_values,
            "historical_activation_bits": historical_values * activation_bits,
            "activation_window_values_including_current": window_values,
            "activation_window_bits_including_current": window_values * activation_bits,
            "note": (
                "One master WeightBuffer directly feeds the MACs; there is no "
                "PE-local weight copy. Activation-window counts include each "
                "current sample wire, while historical counts include stored "
                "HiddenBuffer state only. Bias, FEx temporaries, accumulators, "
                "control, and interfaces are not included."
            ),
        },
        "golden_vectors": {
            "source": input_source,
            "length": int(input_float.shape[0]),
            "files": golden_files,
            "integer_vs_fake_qat": comparison,
            "acceptance_for_rtl": "all exported integer trace files must match RTL at 0 LSB",
        },
        "remaining_system_decisions": [
            "clock frequency and sample rate",
            "number of MAC units and weight-load schedule",
            "pipeline latency and valid/ready protocol",
            "reset warm-up validity policy",
            "full-dataset frozen-PA NMSE/EVM/ACLR revalidation of integer contract",
        ],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    reference_path = output_dir / "bitexact_reference.py"
    reference_path.write_text(
        "\"\"\"Generated 0-LSB verifier for the OpenDPD RTL integer contract.\"\"\"\n"
        "import json\n"
        "from pathlib import Path\n\n"
        "from quant.rtl_export import verify_exported_golden\n\n"
        "manifest = Path(__file__).parent / 'manifest.json'\n"
        "result = verify_exported_golden(manifest)\n"
        "print(json.dumps({'status': 'pass', 'traces': result}, indent=2))\n"
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# OpenDPD FExLite QAT RTL export\n\n"
        "`manifest.json` is the machine-readable implementation contract. Weight and "
        "golden `.mem` files are two's-complement hexadecimal, one word per line. CSV "
        "companions contain signed decimal values. PyTorch kernel index 0 is the oldest "
        "causal tap; use each layer's `causal_delay_by_pytorch_kernel_index`.\n\n"
        "After installing the OpenDPD-TCN-QAT fork, run `python3 bitexact_reference.py` "
        "to recompute every trace using only this package's manifest and memories.\n\n"
        "RTL must match every integer golden trace at 0 LSB. This package measures "
        "compatibility with the trained fake-QAT graph, but the integer contract must "
        "still be re-evaluated over the complete dataset through the frozen PA before "
        "claiming unchanged NMSE/EVM/ACLR.\n"
    )
    return manifest


def export_fexlite_qat_rtl(
    checkpoint: str | Path,
    output_dir: str | Path,
    pa_checkpoint: str | Path | None = None,
    input_path: str | Path | None = None,
    golden_start: int = 0,
    golden_length: int = 256,
    golden_seed: int = 2026,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Build an immutable export package and publish it with one rename.

    A partially written manifest must never look like a consumable TCN-Compiler
    package.  Generation therefore happens in a sibling temporary directory;
    the requested output must not already exist and is made visible only after
    every file has been written successfully.
    """

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"QAT checkpoint not found: {checkpoint_path}")
    if golden_start < 0:
        raise ValueError("golden_start must be non-negative")
    if golden_length < 1:
        raise ValueError("golden_length must be positive")
    if pa_checkpoint is not None and not Path(pa_checkpoint).expanduser().resolve().is_file():
        raise FileNotFoundError(f"PA checkpoint not found: {Path(pa_checkpoint).expanduser().resolve()}")
    if input_path is not None and not Path(input_path).expanduser().resolve().is_file():
        raise FileNotFoundError(f"golden input not found: {Path(input_path).expanduser().resolve()}")

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(
            f"refusing to replace an existing export package: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    ).resolve()
    published = False
    try:
        manifest = _export_fexlite_qat_rtl_into(
            checkpoint=checkpoint_path,
            output_dir=staging,
            pa_checkpoint=pa_checkpoint,
            input_path=input_path,
            golden_start=golden_start,
            golden_length=golden_length,
            golden_seed=golden_seed,
            dataset_name=dataset_name,
        )
        os.replace(staging, target)
        published = True
        return manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
