"""Native FExLite causal-TCN QAT regression tests."""

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CoreModel
from quant import get_quant_model
from quant.qmodules.quant_layers import INT_Conv1D


def _project(bits=8, pretrained_model=""):
    return SimpleNamespace(
        quant=True,
        n_bits_w=bits,
        n_bits_a=bits,
        pretrained_model=pretrained_model,
        quant_dir_label="test",
        DPD_backbone="fexlite_causal_tcn",
        quant_calibration_batches=2,
        quant_calibration_quantile=1.0,
    )


def test_structure_is_parameterized_and_causal():
    torch.manual_seed(3)
    model = CoreModel(
        2,
        3,
        2,
        "fexlite_causal_tcn",
        tcn_kernel_size=3,
        tcn_dilation_base=3,
    ).eval()
    assert model.backbone.dilations == (1, 3)
    assert model.backbone.receptive_field_samples == 9
    assert model.backbone._rtl_spec.tolist() == [1, 2, 3, 3]
    source = torch.randn(1, 24, 2)
    changed = source.clone()
    changed[:, 15:] = torch.randn_like(changed[:, 15:])
    with torch.no_grad():
        before = model(source)
        after = model(changed)
    assert before.shape == source.shape
    assert torch.equal(before[:, :15], after[:, :15])


def test_native_qat_replaces_all_conv1d_and_preserves_bias():
    torch.manual_seed(5)
    float_model = CoreModel(2, 3, 2, "fexlite_causal_tcn")
    expected_bias = float_model.backbone.network[0].bias.detach().clone()
    quantized = get_quant_model(_project(), float_model)
    layers = [m for m in quantized.modules() if isinstance(m, INT_Conv1D)]
    assert len(layers) == 4
    assert torch.equal(layers[0].bias, expected_bias)
    assert all(int(layer.n_bits_a.item()) == 8 for layer in layers)
    assert all(int(layer.n_bits_w.item()) == 8 for layer in layers)
    for layer in layers:
        required = layer.weight.detach().abs().max() / layer.weight_quantizer.Qp
        assert layer.weight_quantizer.scale >= required


def test_calibration_uses_train_loader_and_keeps_physical_io_grid():
    torch.manual_seed(7)
    project = _project(bits=12)
    model = get_quant_model(
        project, CoreModel(2, 2, 1, "fexlite_causal_tcn", tcn_kernel_size=3)
    )
    features = torch.randn(6, 20, 2).clamp(-1.0, 1.0)
    targets = torch.zeros_like(features)
    loader = DataLoader(TensorDataset(features, targets), batch_size=2)
    calibration = project.quant_env.calibrate(loader, torch.device("cpu"))
    assert calibration["raw_input"]["scale"] == 2**-11
    assert calibration["dpd_output"]["scale"] == 2**-11
    assert {"conv0_input", "conv1_input", "conv2_input"} <= calibration.keys()
    parameter_names = dict(model.named_parameters())
    buffer_names = dict(model.named_buffers())
    assert "input_quantizer.scale" not in parameter_names
    assert "output_quantizer.scale" not in parameter_names
    assert "input_quantizer.scale" in buffer_names
    assert "output_quantizer.scale" in buffer_names
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = (
        model.input_quantizer.scale.clone(),
        model.output_quantizer.scale.clone(),
    )
    optimizer.zero_grad()
    model(features[:1]).square().mean().backward()
    optimizer.step()
    assert torch.equal(model.input_quantizer.scale, before[0])
    assert torch.equal(model.output_quantizer.scale, before[1])
    for layer in (m for m in model.modules() if isinstance(m, INT_Conv1D)):
        assert isinstance(layer.act_quantizer.scale, torch.nn.Parameter)
        assert layer.act_quantizer.scale.requires_grad
        exponent = torch.log2(layer.act_quantizer.scale.detach())
        assert torch.equal(exponent, exponent.round())
    with torch.no_grad():
        output = model(features[:1])
    assert torch.equal(output / 2**-11, (output / 2**-11).round())


def test_tcn_qat_fails_closed_for_missing_pretrained_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        project = _project(
            pretrained_model=str(Path(directory) / "missing.pt")
        )
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "native FExLite TCN QAT setup failed"
        ):
            get_quant_model(project, CoreModel(2, 2, 1, "fexlite_causal_tcn"))


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
