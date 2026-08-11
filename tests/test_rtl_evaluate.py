"""Self-contained integer-contract/frozen-PA evaluation tests."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CoreModel
from quant import get_quant_model
from quant.qmodules.quant_layers import INT_Conv1D
from quant.rtl_export import export_fexlite_qat_rtl
from quant.rtl_evaluate import (
    EVALUATION_FORMAT,
    EVALUATION_FORMAT_VERSION,
    evaluate_fexlite_integer_pa,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zero_mem(path: Path, count: int, bits: int = 8) -> dict:
    digits = (bits + 3) // 4
    path.write_text("".join(f"{0:0{digits}X}\n" for _ in range(count)))
    return {
        "path": path.name,
        "count": count,
        "bits": bits,
        "encoding": "two_complement_hex_one_word_per_line",
        "sha256": _sha256(path),
    }


def _make_identity_integer_export(root: Path, pa_checkpoint: Path) -> Path:
    project = type("ProjectArgs", (), {
        "quant": True,
        "n_bits_w": 8,
        "n_bits_a": 8,
        "pretrained_model": "",
        "quant_dir_label": "evaluation_test",
        "DPD_backbone": "fexlite_causal_tcn",
        "quant_calibration_batches": 1,
        "quant_calibration_quantile": 1.0,
    })()
    model = get_quant_model(
        project,
        CoreModel(
            2, 1, 4, "fexlite_causal_tcn",
            tcn_kernel_size=1, tcn_dilation_base=2,
        ),
    )
    with torch.no_grad():
        for layer in (item for item in model.modules() if isinstance(item, INT_Conv1D)):
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
    checkpoint = root.parent / "identity_qat.pt"
    torch.save(model.state_dict(), checkpoint)
    export_fexlite_qat_rtl(
        checkpoint,
        root,
        pa_checkpoint=pa_checkpoint,
        golden_length=8,
    )
    return root / "manifest.json"


def _write_iq(path: Path, value: np.ndarray) -> None:
    pd.DataFrame(value, columns=["I", "Q"]).to_csv(path, index=False)


def _make_dataset(root: Path) -> None:
    root.mkdir()
    rng = np.random.default_rng(27)
    train_input = rng.normal(scale=0.18, size=(192, 2)).astype(np.float32)
    val_input = rng.normal(scale=0.18, size=(64, 2)).astype(np.float32)
    test_input = rng.normal(scale=0.18, size=(128, 2)).astype(np.float32)
    for split, input_iq in (
        ("train", train_input), ("val", val_input), ("test", test_input)
    ):
        output_iq = 0.8 * input_iq + 0.03 * input_iq**3
        _write_iq(root / f"{split}_input.csv", input_iq)
        _write_iq(root / f"{split}_output.csv", output_iq)
    (root / "spec.json").write_text(json.dumps({
        "dataset_format": "split_csv",
        "input_signal_fs": 800e6,
        "bw_main_ch": 200e6,
        "n_sub_ch": 2,
        "nperseg": 64,
    }, indent=2) + "\n")


def _contains_absolute_path(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and value.startswith("/")


def test_integer_contract_frozen_pa_evaluation_is_finite_and_self_contained(tmp_path):
    torch.manual_seed(19)
    pa = CoreModel(2, 2, 1, "dgru")
    pa_checkpoint = tmp_path / "toy_pa.pt"
    torch.save(pa.state_dict(), pa_checkpoint)
    dataset_path = tmp_path / "toy_dataset"
    _make_dataset(dataset_path)
    manifest_path = _make_identity_integer_export(
        tmp_path / "integer_export", pa_checkpoint
    )
    output_path = tmp_path / "evaluation.json"

    result = evaluate_fexlite_integer_pa(
        manifest_path,
        pa_checkpoint,
        dataset_path=dataset_path,
        output_path=output_path,
        split="test",
        protocol="segmented",
    )

    assert output_path.is_file()
    assert json.loads(output_path.read_text()) == result
    assert result["format"] == EVALUATION_FORMAT
    assert result["format_version"] == EVALUATION_FORMAT_VERSION
    assert result["status"] == "pass"
    assert result["protocol"]["segment_length"] == 64
    assert result["protocol"]["segment_count"] == 2
    assert result["protocol"]["zero_padding_samples"] == 0
    assert len(result["protocol_sha256"]) == 64
    assert len(result["provenance_sha256"]) == 64
    assert result["model"]["frozen_pa"]["frozen"] is True
    assert result["provenance"]["runtime"]["pandas"] == pd.__version__
    assert "quant/rtl_manifest.py" in result["provenance"]["evaluator_source_sha256"]
    assert "modules/data_collector.py" in result["provenance"]["evaluator_source_sha256"]
    assert result["quantization"]["raw_input_code_count"] == 128 * 2
    for metric_set in result["metrics"].values():
        assert set(metric_set) == {"NMSE", "EVM", "ACLR_L", "ACLR_R", "ACLR_AVG"}
        assert all(math.isfinite(value) for value in metric_set.values())
    assert not _contains_absolute_path(result)
    assert not list(tmp_path.glob(".evaluation.json.*.tmp"))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_integer_contract_frozen_pa_evaluation_is_finite_and_self_contained(
            Path(directory)
        )
    print("PASS test_integer_contract_frozen_pa_evaluation_is_finite_and_self_contained")
