"""Regression tests for collision-resistant FExLite TCN model IDs."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project import Project


def _project(**overrides):
    values = {
        "seed": 4,
        "DPD_backbone": "fexlite_causal_tcn",
        "DPD_hidden_size": 10,
        "frame_length": 200,
        "DPD_num_layers": 4,
        "tcn_kernel_size": 5,
        "tcn_dilation_base": 2,
        "quant": True,
        "n_bits_a": 12,
        "n_bits_w": 12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _model_id(proj, params=304, legacy=False):
    return Project.gen_dpd_model_id(proj, params, legacy=legacy)


class ModelIdTests(unittest.TestCase):
    def test_tcn_topology_and_precision_are_part_of_id(self):
        baseline = _model_id(_project())
        self.assertIn("_L_4_K_5_DB_2_A_12_W_12_", baseline)
        self.assertNotEqual(baseline, _model_id(_project(tcn_dilation_base=3)))
        self.assertNotEqual(baseline, _model_id(_project(DPD_num_layers=3)))
        self.assertNotEqual(baseline, _model_id(_project(tcn_kernel_size=3)))
        self.assertNotEqual(baseline, _model_id(_project(n_bits_a=10)))
        self.assertNotEqual(baseline, _model_id(_project(n_bits_w=10)))

    def test_legacy_tcn_and_non_tcn_ids_remain_available(self):
        legacy = _model_id(_project(), legacy=True)
        self.assertEqual(
            legacy,
            "DPD_S_4_M_FEXLITE_CAUSAL_TCN_H_10_F_200_P_304",
        )
        gru = _project(DPD_backbone="gru", quant=False)
        self.assertEqual(
            _model_id(gru),
            "DPD_S_4_M_GRU_H_10_F_200_P_304",
        )


if __name__ == "__main__":
    unittest.main()
