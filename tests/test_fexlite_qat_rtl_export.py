"""Exporter-only regression tests for the TCN-Compiler frontend contract."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CoreModel
from quant import get_quant_model
from quant.rtl_export import export_fexlite_qat_rtl, verify_exported_golden


def _quantized_model(
    hidden: int,
    temporal_layers: int,
    kernel_size: int,
    dilation_base: int = 2,
    bits: int = 8,
):
    project = SimpleNamespace(
        quant=True,
        n_bits_w=bits,
        n_bits_a=bits,
        pretrained_model="",
        quant_dir_label="export_test",
        DPD_backbone="fexlite_causal_tcn",
        quant_calibration_batches=1,
        quant_calibration_quantile=0.9999,
    )
    float_model = CoreModel(
        2,
        hidden,
        temporal_layers,
        "fexlite_causal_tcn",
        tcn_kernel_size=kernel_size,
        tcn_dilation_base=dilation_base,
    )
    return get_quant_model(project, float_model).eval()


def _assert_zero_lsb(test: unittest.TestCase, manifest_path: Path) -> None:
    result = verify_exported_golden(manifest_path)
    test.assertTrue(result)
    test.assertTrue(all(item["match"] for item in result.values()))
    test.assertTrue(
        all(
            item["maximum_absolute_lsb_error"] == 0
            for item in result.values()
        )
    )


class FExLiteQATRTLExportTests(unittest.TestCase):
    def test_toy_h2_l1_k3_spec_export_is_atomic_and_bit_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            torch.manual_seed(7)
            checkpoint = tmp_path / "toy_h2_l1_k3.pt"
            torch.save(_quantized_model(2, 1, 3).state_dict(), checkpoint)
            output = tmp_path / "toy_export"

            manifest = export_fexlite_qat_rtl(
                checkpoint,
                output,
                golden_length=16,
                golden_seed=11,
                dataset_name="deterministic_toy",
            )
            loaded = json.loads((output / "manifest.json").read_text())

            self.assertEqual(manifest, loaded)
            self.assertEqual(
                loaded["format"], "opendpd_fexlite_qat_rtl_export"
            )
            self.assertEqual(loaded["format_version"], 1)
            self.assertEqual(loaded["model"]["hidden_channels"], 2)
            self.assertEqual(loaded["model"]["temporal_layers"], 1)
            self.assertEqual(loaded["model"]["kernel_size"], 3)
            self.assertEqual(loaded["model"]["dilations"], [1])
            self.assertEqual(loaded["model"]["receptive_field_samples"], 3)
            self.assertEqual(loaded["model"]["weight_count"], 22)
            hardware_cost = loaded["hardware_cost"]
            self.assertEqual(hardware_cost["raw_weight_bits"], 22 * 8)
            self.assertEqual(hardware_cost["weight_buffer_bits"], 22 * 8)
            self.assertEqual(hardware_cost["weight_storage_copies"], 1)
            self.assertEqual(hardware_cost["physical_weight_state_bits"], 22 * 8)
            self.assertNotIn("mac_local_weight_bits", hardware_cost)
            self.assertEqual(hardware_cost["historical_activation_values"], 4)
            self.assertEqual(
                hardware_cost["activation_window_values_including_current"], 6
            )
            self.assertEqual(
                [layer["kind"] for layer in loaded["model"]["layers"]],
                [
                    "input_projection",
                    "causal_depthwise",
                    "output_projection",
                ],
            )
            self.assertEqual(
                loaded["provenance"]["topology_recovery"]["source"],
                "checkpoint_rtl_spec",
            )
            checkpoint_record = loaded["provenance"]["dpd_checkpoint"]
            self.assertEqual(checkpoint_record["path_hint"], checkpoint.name)
            self.assertFalse(Path(checkpoint_record["path_hint"]).is_absolute())
            self.assertNotIn(
                str(Path(__file__).resolve().parent.parent),
                (output / "bitexact_reference.py").read_text(),
            )
            self.assertFalse(list(tmp_path.glob(".toy_export.tmp-*")))
            _assert_zero_lsb(self, output / "manifest.json")

            with self.assertRaises(FileExistsError):
                export_fexlite_qat_rtl(checkpoint, output, golden_length=4)

    def test_rtl_spec_restores_nonlegacy_dilation_base(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            torch.manual_seed(9)
            checkpoint = tmp_path / "spec_h2_l2_k3_base3.pt"
            torch.save(
                _quantized_model(
                    2, 2, 3, dilation_base=3, bits=8
                ).state_dict(),
                checkpoint,
            )
            output = tmp_path / "spec_export"

            manifest = export_fexlite_qat_rtl(
                checkpoint,
                output,
                golden_length=8,
                golden_seed=19,
            )

            self.assertEqual(manifest["model"]["temporal_layers"], 2)
            self.assertEqual(manifest["model"]["kernel_size"], 3)
            self.assertEqual(manifest["model"]["dilation_base"], 3)
            self.assertEqual(manifest["model"]["dilations"], [1, 3])
            self.assertEqual(manifest["model"]["receptive_field_samples"], 9)
            self.assertEqual(
                [
                    layer["dilation"]
                    for layer in manifest["model"]["layers"]
                    if layer["kind"] == "causal_depthwise"
                ],
                [1, 3],
            )
            _assert_zero_lsb(self, output / "manifest.json")

    def test_legacy_canonical_h10_l4_k5_without_spec_uses_base2(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            torch.manual_seed(13)
            model = _quantized_model(10, 4, 5, bits=12)
            state = model.state_dict()
            spec_keys = [
                key for key in state if key.endswith("backbone._rtl_spec")
            ]
            self.assertEqual(len(spec_keys), 1)
            state.pop(spec_keys[0])
            # Historical boundary Parameters could drift inside the same
            # effective 2^-11 physical grid.  Legacy loading normalizes them.
            state["input_quantizer.scale"].fill_(0.00048651493852958083)
            state["output_quantizer.scale"].fill_(0.00048651493852958083)
            checkpoint = tmp_path / "legacy_h10_l4_k5.pt"
            torch.save(state, checkpoint)
            output = tmp_path / "legacy_export"

            manifest = export_fexlite_qat_rtl(
                checkpoint,
                output,
                golden_length=8,
                golden_seed=17,
            )

            self.assertEqual(manifest["model"]["hidden_channels"], 10)
            self.assertEqual(manifest["model"]["temporal_layers"], 4)
            self.assertEqual(manifest["model"]["kernel_size"], 5)
            self.assertEqual(manifest["model"]["dilation_base"], 2)
            self.assertEqual(manifest["model"]["dilations"], [1, 2, 4, 8])
            self.assertEqual(manifest["model"]["receptive_field_samples"], 61)
            self.assertEqual(manifest["model"]["weight_count"], 280)
            hardware_cost = manifest["hardware_cost"]
            self.assertEqual(hardware_cost["weight_buffer_bits"], 3360)
            self.assertEqual(hardware_cost["weight_storage_copies"], 1)
            self.assertEqual(hardware_cost["physical_weight_state_bits"], 3360)
            self.assertNotIn("mac_local_weight_bits", hardware_cost)
            self.assertEqual(hardware_cost["historical_activation_values"], 600)
            self.assertEqual(
                hardware_cost["activation_window_values_including_current"], 640
            )
            recovery = manifest["provenance"]["topology_recovery"]
            self.assertEqual(
                recovery["source"], "legacy_conv_shapes_with_base2"
            )
            self.assertTrue(recovery["legacy_canonical_h10_l4_k5"])
            self.assertEqual(
                manifest["provenance"]["training_sidecars"]["status"],
                "unavailable_legacy",
            )
            self.assertEqual(len(manifest["model"]["layers"]), 6)
            self.assertEqual(manifest["model"]["layers"][-1]["index"], 5)
            self.assertEqual(
                manifest["model"]["layers"][-1]["kind"],
                "output_projection",
            )
            _assert_zero_lsb(self, output / "manifest.json")

    def test_strict_validator_rejects_rounding_and_golden_sha_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            torch.save(_quantized_model(2, 1, 3).state_dict(), checkpoint)
            output = root / "export"
            export_fexlite_qat_rtl(checkpoint, output, golden_length=4)
            manifest_path = output / "manifest.json"
            original = json.loads(manifest_path.read_text())

            altered = json.loads(json.dumps(original))
            altered["quantization"]["raw_input"]["rounding"] = "round_away_from_zero"
            manifest_path.write_text(json.dumps(altered, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "RNE ties-to-even"):
                verify_exported_golden(manifest_path)

            manifest_path.write_text(json.dumps(original, indent=2) + "\n")
            raw_path = output / original["golden_vectors"]["files"]["raw_input"]["path"]
            raw_path.write_text(raw_path.read_text() + "00\n")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                verify_exported_golden(manifest_path)

    def test_new_checkpoint_requires_exact_physical_grid_and_signed32_bias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = _quantized_model(2, 1, 3)
            state = model.state_dict()
            state["input_quantizer.scale"].fill_(2**-5)
            bad_grid = root / "bad_grid.pt"
            torch.save(state, bad_grid)
            with self.assertRaisesRegex(ValueError, "exactly 2\^\(1-A\)"):
                export_fexlite_qat_rtl(bad_grid, root / "bad_grid_export")

            state = model.state_dict()
            state["input_quantizer.scale"].fill_(2**-7)
            state["output_quantizer.scale"].fill_(2**-7)
            weight_scale = float(state[
                "model.backbone.network.0.weight_quantizer.scale"
            ].item())
            weight_exponent = round(torch.log2(torch.tensor(abs(weight_scale))).item())
            activation_scale = float(state[
                "model.backbone.network.0.act_quantizer.scale"
            ].item())
            activation_exponent = round(torch.log2(torch.tensor(abs(activation_scale))).item())
            accumulator_scale = 2.0 ** (weight_exponent + activation_exponent)
            state["model.backbone.network.0.bias"].fill_((2**31) * accumulator_scale)
            bad_bias = root / "bad_bias.pt"
            torch.save(state, bad_bias)
            with self.assertRaisesRegex(OverflowError, "signed 32-bit"):
                export_fexlite_qat_rtl(bad_bias, root / "bad_bias_export")


if __name__ == "__main__":
    unittest.main()
