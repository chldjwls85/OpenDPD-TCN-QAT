"""Forward-pass tests for every implemented backbone.

Instantiates each backbone through models.CoreModel exactly as the training
pipeline does, then checks the (batch, time, 2) -> (batch, time, 2) forward
contract on CPU with finite outputs. Catches broken imports, constructor
signature drift, and shape regressions for all supported models at once.
"""

import pytest
import torch

from models import CascadedModel, CoreModel
from utils.util import count_net_params

# Every backbone_type implemented in models.CoreModel. 'neuraltx' is excluded:
# its models.py branch references a backbones/neuraltx.py module that does not
# exist in the repository.
IMPLEMENTED_BACKBONES = [
    "gmp",
    "gru",
    "dgru",
    "qgru",
    "qgru_amp1",
    "lstm",
    "vdlstm",
    "rvtdcnn",
    "apnrru",
    "bojanet",
    "deltagru",
    "deltajanet",
    "pgjanet",
    "dvrjanet",
    "tres_deltagru",
    "tcn",
    "mcldnn",
]

BATCH, TIME = 2, 32


def build_model(backbone_type, hidden_size=8):
    torch.manual_seed(0)
    return CoreModel(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=1,
        backbone_type=backbone_type,
        window_size=4,
        num_dvr_units=3,
        thx=0.0,
        thh=0.0,
    )


@pytest.mark.parametrize("backbone_type", IMPLEMENTED_BACKBONES)
def test_forward_pass_shape_and_finiteness(backbone_type):
    model = build_model(backbone_type)
    torch.manual_seed(1)
    # Nonzero random I/Q input: several backbones normalize by the signal
    # amplitude and would produce NaN on all-zero input.
    x = torch.randn(BATCH, TIME, 2) * 0.5 + 0.1

    with torch.no_grad():
        out = model(x)

    assert out.shape == (BATCH, TIME, 2), (
        f"{backbone_type}: expected output shape {(BATCH, TIME, 2)}, got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), f"{backbone_type}: output contains NaN/Inf"


@pytest.mark.parametrize("backbone_type", IMPLEMENTED_BACKBONES)
def test_has_trainable_parameters(backbone_type):
    model = build_model(backbone_type)
    assert count_net_params(model) > 0


def test_unknown_backbone_raises_value_error():
    with pytest.raises(ValueError):
        build_model("no_such_backbone")


def test_cascaded_model_freezes_pa():
    dpd_model = build_model("gru")
    pa_model = build_model("gru")
    cascaded = CascadedModel(dpd_model, pa_model)
    cascaded.freeze_pa_model()

    assert all(not p.requires_grad for p in cascaded.pa_model.parameters())
    assert all(p.requires_grad for p in cascaded.dpd_model.parameters())

    torch.manual_seed(2)
    x = torch.randn(BATCH, TIME, 2) * 0.5 + 0.1
    out = cascaded(x)
    assert out.shape == (BATCH, TIME, 2)
    assert torch.isfinite(out).all()
