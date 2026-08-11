"""Frozen-PA evaluation for an exported FExLite integer RTL contract.

This module deliberately evaluates the exported ``manifest.json`` and ``.mem``
files instead of a PyTorch DPD checkpoint.  Test sequences are segmented in the
same way as :class:`modules.data_collector.IQSegmentDataset`: the final segment
is zero padded and every segment starts with zero TCN history and zero PA hidden
state.  Consequently, the reported metrics describe the numeric contract that
RTL must implement, not merely the fake-quantized training graph.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scipy
import torch

from models import CoreModel
from modules.data_collector import load_dataset
from quant.rtl_export import (
    load_exported_integer_runtime,
    load_qat_model,
    quantize_codes,
    run_integer_reference,
    sha256_file,
)
from quant.rtl_manifest import validate_manifest_v1
from utils.metrics import ACLR, EVM, NMSE
from utils.util import set_target_gain


EVALUATION_FORMAT = "opendpd_fexlite_integer_frozen_pa_evaluation"
EVALUATION_FORMAT_VERSION = 1
SUPPORTED_MANIFEST_FORMAT = "opendpd_fexlite_qat_rtl_export"
SUPPORTED_MANIFEST_VERSION = 1
DEFAULT_NPERSEG = 2560
DEFAULT_SAMPLE_RATE = 800e6
DEFAULT_MAIN_CHANNEL_BW = 200e6
DEFAULT_SUBCHANNELS = 10


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _path_hint(path: Path) -> str:
    """Return a useful, deliberately non-absolute artifact hint."""
    path = Path(path)
    parent = path.parent.name
    return f"{parent}/{path.name}" if parent else path.name


def _artifact_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "basename": path.name,
        "path_hint": _path_hint(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_write_json(path: str | Path, result: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _load_spec(dataset_path: Path) -> tuple[dict[str, Any], Path | None]:
    directory = dataset_path if dataset_path.is_dir() else dataset_path.parent
    spec_path = directory / "spec.json"
    if spec_path.exists():
        return json.loads(spec_path.read_text()), spec_path
    return {}, None


def _dataset_artifacts(
    dataset_path: Path,
    spec: Mapping[str, Any],
    spec_path: Path | None,
    split: str,
) -> list[dict[str, Any]]:
    paths: list[Path]
    if dataset_path.is_file():
        paths = [dataset_path]
    elif spec.get("dataset_format", "split_csv") == "single_csv":
        paths = [dataset_path / str(spec.get("csv_filename", "data.csv"))]
    else:
        paths = [
            dataset_path / "train_input.csv",
            dataset_path / "train_output.csv",
            dataset_path / f"{split}_input.csv",
            dataset_path / f"{split}_output.csv",
        ]
    if spec_path is not None:
        paths.append(spec_path)
    unique_paths = sorted(set(paths), key=lambda item: item.name)
    return [_artifact_record(path) for path in unique_paths]


def _resolve_dataset(
    dataset_name: str | None,
    dataset_path: str | Path | None,
) -> tuple[Path, str]:
    if bool(dataset_name) == bool(dataset_path):
        raise ValueError("provide exactly one of dataset_name and dataset_path")
    if dataset_name:
        root = Path(__file__).resolve().parent.parent
        resolved = root / "datasets" / dataset_name
        label = dataset_name
    else:
        resolved = Path(dataset_path).expanduser().resolve()  # type: ignore[arg-type]
        label = resolved.stem if resolved.is_file() else resolved.name
    if not resolved.exists():
        raise FileNotFoundError(f"dataset does not exist: {resolved}")
    return resolved, label


def _split_arrays(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_input, train_output, val_input, val_output, test_input, test_output = arrays
    if split == "test":
        return train_input, train_output, test_input, test_output
    if split == "val":
        return train_input, train_output, val_input, val_output
    raise ValueError("split must be 'test' or 'val'")


def _segment_zero_pad(value: np.ndarray, nperseg: int) -> tuple[np.ndarray, int]:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"I/Q data must have shape [samples, 2], got {value.shape}")
    if value.shape[0] == 0:
        raise ValueError("evaluation split is empty")
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")
    segment_count = math.ceil(value.shape[0] / nperseg)
    padded_samples = segment_count * nperseg - value.shape[0]
    if padded_samples:
        value = np.pad(value, ((0, padded_samples), (0, 0)), mode="constant")
    return value.reshape(segment_count, nperseg, 2), padded_samples


def _normalise_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("PA checkpoint does not contain a state dictionary")
    result = dict(state)
    if all(key.startswith("module.") for key in result):
        result = {key[len("module."):]: value for key, value in result.items()}
    return result


def _load_frozen_dgru_pa(
    checkpoint: Path, device: torch.device
) -> tuple[CoreModel, dict[str, Any]]:
    state = _normalise_state_dict(checkpoint)
    hidden_weight = next(
        (value for key, value in state.items() if key.endswith("backbone.fc_hid.weight")),
        None,
    )
    if hidden_weight is None or hidden_weight.ndim != 2:
        raise ValueError("checkpoint is not an OpenDPD DGRU PA checkpoint")
    hidden_size = int(hidden_weight.shape[0])
    layer_indices = []
    for key in state:
        match = re.search(r"backbone\.rnn\.weight_ih_l(\d+)$", key)
        if match:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        raise ValueError("DGRU checkpoint has no recurrent layers")
    num_layers = max(layer_indices) + 1
    expected_layers = set(range(num_layers))
    if set(layer_indices) != expected_layers:
        raise ValueError(f"non-contiguous DGRU layer indices: {sorted(layer_indices)}")

    model = CoreModel(2, hidden_size, num_layers, "dgru")
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return model, {
        "backbone": "dgru",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "frozen": all(not parameter.requires_grad for parameter in model.parameters()),
    }


def _verify_manifest_and_mem(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validated, artifacts = validate_manifest_v1(manifest_path)
    if validated != manifest:
        raise ValueError("manifest changed while it was being validated")
    return [
        artifact for artifact in artifacts
        if not artifact["role"].startswith("golden ")
    ]


def _run_integer_dpd(
    segments: np.ndarray,
    manifest: Mapping[str, Any],
    runtime_layers: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_quantizer = manifest["quantization"]["raw_input"]
    output_quantizer = manifest["quantization"]["dpd_output"]
    all_input_codes = []
    all_output_codes = []
    for segment in segments:
        input_float = torch.from_numpy(np.ascontiguousarray(segment))
        input_codes = quantize_codes(
            input_float,
            raw_quantizer["scale"],
            raw_quantizer["qmin"],
            raw_quantizer["qmax"],
        )
        output_codes = run_integer_reference(
            input_codes,
            runtime_layers,
            raw_quantizer,
            output_quantizer,
        )["dpd_output"]
        all_input_codes.append(input_codes)
        all_output_codes.append(output_codes)
    return torch.stack(all_input_codes), torch.stack(all_output_codes)


def _run_pa(
    pa_model: CoreModel,
    input_iq: torch.Tensor | np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if isinstance(input_iq, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(input_iq, dtype=np.float32))
    else:
        tensor = input_iq.detach().to(dtype=torch.float32)
    with torch.no_grad():
        result = pa_model(tensor.to(device))
    return result.detach().cpu().numpy()


def _metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    sample_rate: float,
    nperseg: int,
    bw_main_ch: float,
    n_sub_ch: int,
) -> dict[str, float]:
    aclr_left, aclr_right = ACLR(
        prediction,
        fs=sample_rate,
        nperseg=nperseg,
        bw_main_ch=bw_main_ch,
        n_sub_ch=n_sub_ch,
    )
    result = {
        "NMSE": float(NMSE(prediction, reference)),
        "EVM": float(EVM(
            prediction,
            reference,
            sample_rate=int(sample_rate),
            bw_main_ch=bw_main_ch,
            n_sub_ch=n_sub_ch,
            nperseg=nperseg,
        )),
        "ACLR_L": float(aclr_left),
        "ACLR_R": float(aclr_right),
        "ACLR_AVG": float((aclr_left + aclr_right) / 2),
    }
    non_finite = [name for name, value in result.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(f"non-finite PA metrics: {', '.join(non_finite)}")
    return result


def _tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    contiguous = np.ascontiguousarray(value)
    payload = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }
    return _sha256_json(payload)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    relative_paths = [
        "backbones/dgru.py",
        "backbones/fexlite_causal_tcn.py",
        "backbones/rvtdcnn.py",
        "models.py",
        "modules/data_collector.py",
        "quant/__init__.py",
        "quant/modules/gru.py",
        "quant/modules/ops.py",
        "quant/qmodules/__init__.py",
        "quant/qmodules/quant_layers.py",
        "quant/qmodules/quant_ops.py",
        "quant/qmodules/quantizers.py",
        "quant/quant_envs.py",
        "quant/rtl_evaluate.py",
        "quant/rtl_export.py",
        "quant/rtl_manifest.py",
        "utils/metrics.py",
        "utils/util.py",
    ]
    return {
        relative_path: sha256_file(root / relative_path)
        for relative_path in relative_paths
    }


def evaluate_fexlite_integer_pa(
    manifest_path: str | Path,
    pa_checkpoint: str | Path,
    *,
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    split: str = "test",
    protocol: str = "segmented",
    device: str | torch.device = "cpu",
    qat_checkpoint: str | Path | None = None,
    nperseg: int | None = None,
    sample_rate: float | None = None,
    bw_main_ch: float | None = None,
    n_sub_ch: int | None = None,
) -> dict[str, Any]:
    """Evaluate an exported integer DPD through a frozen DGRU PA.

    ``protocol='segmented'`` is the OpenDPD validation/test policy: zero-pad the
    final segment and independently reset both TCN history and DGRU hidden state
    at each segment boundary.
    """
    if protocol != "segmented":
        raise ValueError("only protocol='segmented' is supported")
    manifest_path = Path(manifest_path).expanduser().resolve()
    pa_checkpoint = Path(pa_checkpoint).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    if not pa_checkpoint.is_file():
        raise FileNotFoundError(f"PA checkpoint does not exist: {pa_checkpoint}")

    manifest, runtime_layers = load_exported_integer_runtime(manifest_path)
    weight_artifacts = _verify_manifest_and_mem(manifest_path, manifest)
    declared_pa = manifest.get("provenance", {}).get("pa_checkpoint")
    actual_pa_sha = sha256_file(pa_checkpoint)
    if declared_pa and declared_pa.get("sha256") != actual_pa_sha:
        raise ValueError("PA checkpoint SHA does not match the exported manifest")

    resolved_dataset, dataset_label = _resolve_dataset(dataset_name, dataset_path)
    declared_dataset = manifest.get("provenance", {}).get("dataset_name")
    if declared_dataset and dataset_name and declared_dataset != dataset_name:
        raise ValueError(
            f"dataset mismatch: manifest declares {declared_dataset!r}, "
            f"evaluation requested {dataset_name!r}"
        )
    spec, spec_path = _load_spec(resolved_dataset)
    nperseg = int(nperseg if nperseg is not None else spec.get("nperseg", DEFAULT_NPERSEG))
    sample_rate = float(
        sample_rate if sample_rate is not None
        else spec.get("input_signal_fs", DEFAULT_SAMPLE_RATE)
    )
    bw_main_ch = float(
        bw_main_ch if bw_main_ch is not None
        else spec.get("bw_main_ch", DEFAULT_MAIN_CHANNEL_BW)
    )
    n_sub_ch = int(
        n_sub_ch if n_sub_ch is not None
        else spec.get("n_sub_ch", DEFAULT_SUBCHANNELS)
    )
    if sample_rate <= 0 or bw_main_ch <= 0 or n_sub_ch <= 0:
        raise ValueError("sample rate, main-channel bandwidth, and subchannels must be positive")

    if dataset_name:
        arrays = load_dataset(dataset_name=dataset_name)
    else:
        arrays = load_dataset(dataset_path=resolved_dataset)
    train_input, train_output, eval_input, measured_output = _split_arrays(arrays, split)
    target_gain = float(set_target_gain(train_input, train_output))
    input_segments, padded_samples = _segment_zero_pad(eval_input, nperseg)
    measured_segments, measured_padding = _segment_zero_pad(measured_output, nperseg)
    if measured_padding != padded_samples:
        raise ValueError("input and output split lengths differ")
    reference_segments, reference_padding = _segment_zero_pad(
        target_gain * np.asarray(eval_input), nperseg
    )
    if reference_padding != padded_samples:
        raise AssertionError("reference segmentation is inconsistent")

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    pa_model, pa_model_record = _load_frozen_dgru_pa(pa_checkpoint, selected_device)

    input_codes, output_codes = _run_integer_dpd(
        input_segments, manifest, runtime_layers
    )
    output_scale = float(manifest["quantization"]["dpd_output"]["scale"])
    integer_dpd_float = output_codes.to(torch.float32) * output_scale
    integer_pa_output = _run_pa(pa_model, integer_dpd_float, selected_device)
    raw_pa_output = _run_pa(pa_model, input_segments, selected_device)

    metric_arguments = {
        "sample_rate": sample_rate,
        "nperseg": nperseg,
        "bw_main_ch": bw_main_ch,
        "n_sub_ch": n_sub_ch,
    }
    metrics = {
        "integer_dpd_frozen_pa": _metrics(
            integer_pa_output, reference_segments, **metric_arguments
        ),
        "frozen_pa_without_dpd": _metrics(
            raw_pa_output, reference_segments, **metric_arguments
        ),
        "measured_pa_without_dpd": _metrics(
            measured_segments, reference_segments, **metric_arguments
        ),
    }

    fake_qat_record = None
    if qat_checkpoint is not None:
        qat_checkpoint = Path(qat_checkpoint).expanduser().resolve()
        if not qat_checkpoint.is_file():
            raise FileNotFoundError(f"QAT checkpoint does not exist: {qat_checkpoint}")
        declared_dpd = manifest.get("provenance", {}).get("dpd_checkpoint")
        qat_sha = sha256_file(qat_checkpoint)
        if declared_dpd and declared_dpd.get("sha256") != qat_sha:
            raise ValueError("QAT checkpoint SHA does not match the exported manifest")
        loaded_qat = load_qat_model(qat_checkpoint)
        if len(loaded_qat) == 2 and isinstance(loaded_qat[1], Mapping):
            qat_model, qat_topology = loaded_qat
            qat_hidden = int(qat_topology["hidden_channels"])
            activation_bits = int(qat_topology["activation_bits"])
            weight_bits = int(qat_topology["weight_bits"])
        else:  # Compatibility with format-v1 exporter implementations.
            qat_model, qat_hidden, activation_bits, weight_bits = loaded_qat
        qat_model.eval().to(selected_device)
        with torch.no_grad():
            fake_dpd = qat_model(torch.from_numpy(input_segments).to(selected_device))
        fake_pa_output = _run_pa(pa_model, fake_dpd, selected_device)
        metrics["fake_qat_dpd_frozen_pa"] = _metrics(
            fake_pa_output, reference_segments, **metric_arguments
        )
        fake_codes = torch.round(fake_dpd.detach().cpu() / output_scale).to(torch.int64)
        code_delta = fake_codes - output_codes
        fake_qat_record = {
            "checkpoint": _artifact_record(qat_checkpoint),
            "hidden_channels": qat_hidden,
            "activation_bits": activation_bits,
            "weight_bits": weight_bits,
            "integer_output_code_comparison": {
                "maximum_absolute_lsb_error": int(code_delta.abs().max().item()),
                "mean_absolute_lsb_error": float(code_delta.abs().to(torch.float64).mean().item()),
                "exact_code_fraction": float((code_delta == 0).to(torch.float64).mean().item()),
            },
        }

    raw_quantizer = manifest["quantization"]["raw_input"]
    protocol_record = {
        "name": "opendpd_segmented_frozen_pa_v1",
        "split": split,
        "segment_length": nperseg,
        "segment_count": int(input_segments.shape[0]),
        "original_sample_count": int(eval_input.shape[0]),
        "evaluated_sample_count_including_padding": int(input_segments.shape[0] * nperseg),
        "zero_padding_samples": padded_samples,
        "integer_tcn_history_reset": "zero at every segment boundary",
        "dgru_hidden_state_reset": "zero at every segment boundary",
        "final_segment_policy": "right zero-pad to segment_length and include padding in metrics",
        "target_reference": "training-split peak-amplitude gain multiplied by evaluation input",
        "target_gain": target_gain,
        "metric_implementation": "OpenDPD utils.metrics NMSE/EVM/ACLR",
        "sample_rate_hz": sample_rate,
        "main_channel_bandwidth_hz": bw_main_ch,
        "subchannel_count": n_sub_ch,
        "units": "NMSE, EVM, and ACLR are reported in dB",
    }

    dataset_files = _dataset_artifacts(
        resolved_dataset, spec, spec_path, split
    )
    provenance = {
        "manifest": _artifact_record(manifest_path),
        "manifest_declared_provenance_sha256": _sha256_json(
            manifest.get("provenance", {})
        ),
        "frozen_pa_checkpoint": _artifact_record(pa_checkpoint),
        "dataset": {
            "name": dataset_label,
            "path_hint": resolved_dataset.name,
            "files": dataset_files,
            "files_sha256": _sha256_json(dataset_files),
        },
        "integer_weight_artifacts": weight_artifacts,
        "integer_weight_artifacts_sha256": _sha256_json(weight_artifacts),
        "evaluator_source_sha256": _source_hashes(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "device_type": selected_device.type,
        },
    }

    qmin = int(raw_quantizer["qmin"])
    qmax = int(raw_quantizer["qmax"])
    result: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "format_version": EVALUATION_FORMAT_VERSION,
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_record,
        "protocol_sha256": _sha256_json(protocol_record),
        "provenance": provenance,
        "provenance_sha256": _sha256_json(provenance),
        "model": {
            "integer_dpd": {
                "backbone": manifest["model"].get("backbone"),
                "hidden_channels": manifest["model"].get("hidden_channels"),
                "receptive_field_samples": manifest["model"].get("receptive_field_samples"),
                "manifest_format": manifest["format"],
                "manifest_format_version": manifest["format_version"],
            },
            "frozen_pa": pa_model_record,
        },
        "quantization": {
            "raw_input": manifest["quantization"]["raw_input"],
            "dpd_output": manifest["quantization"]["dpd_output"],
            "raw_input_saturation_count": int(
                ((input_codes == qmin) | (input_codes == qmax)).sum().item()
            ),
            "raw_input_code_count": int(input_codes.numel()),
        },
        "tensor_sha256": {
            "quantized_raw_input_codes": _tensor_sha256(input_codes),
            "integer_dpd_output_codes": _tensor_sha256(output_codes),
            "integer_dpd_frozen_pa_output": _tensor_sha256(integer_pa_output),
            "linear_reference": _tensor_sha256(reference_segments),
        },
        "metric_units": "dB",
        "metrics": metrics,
    }
    if fake_qat_record is not None:
        result["fake_qat_comparison"] = fake_qat_record
    if output_path is not None:
        _atomic_write_json(output_path, result)
    return result


__all__ = [
    "EVALUATION_FORMAT",
    "EVALUATION_FORMAT_VERSION",
    "evaluate_fexlite_integer_pa",
]
