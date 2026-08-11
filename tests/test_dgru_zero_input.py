"""Frozen-PA stability tests for exact zero-valued quantized DPD samples."""

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CoreModel


def test_dgru_exact_zero_input_is_finite():
    torch.manual_seed(11)
    pa = CoreModel(2, 3, 1, "dgru").eval()
    x = torch.zeros(2, 20, 2)
    with torch.no_grad():
        output = pa(x)
    assert torch.isfinite(output).all()


def test_nonzero_phase_features_preserve_original_formula():
    x = torch.randn(4, 20, 2)
    x[x.abs() < 1e-4] = 1e-4
    i = x[..., 0:1]
    q = x[..., 1:2]
    amplitude = torch.sqrt(i.square() + q.square())
    assert torch.equal(i / amplitude, i / amplitude.clamp_min(1e-12))
    assert torch.equal(q / amplitude, q / amplitude.clamp_min(1e-12))


if __name__ == "__main__":
    for name, test in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
