"""Tests for explicit PA inputs and atomically published QAT artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import hashlib
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CoreModel
from quant import get_quant_model
from quant.rtl_export import export_fexlite_qat_rtl
from quant.rounding_policy import (
    GLOBAL_FLOOR,
    RESEARCH_PREHS_INPUT_RNE,
    ROUND_TO_NEAREST_TIES_TO_EVEN,
    rounding_policy_record,
)
from steps.train_dpd import (
    _publish_dpd_artifacts,
    _publish_qat_artifacts,
    _resolve_pa_checkpoint,
)
from steps.training_artifacts import publish_checkpoint


def _project(
    root: Path,
    save_count: int = 1,
    rounding_policy_mode: str = "baseline_rne",
) -> SimpleNamespace:
    source = root / "legacy" / "best.pt"
    source.parent.mkdir(parents=True)
    project = SimpleNamespace(
        path_save_file_best=str(source),
        qat_output_checkpoint=str(root / "owned" / "qat.pt"),
        logger=SimpleNamespace(checkpoint_save_count=save_count),
        quant_calibration={
            "raw_input": {
                "bits": 12, "scale": 2**-11,
                "policy": "fixed_signed_unit_interface",
            },
            "dpd_output": {
                "bits": 12, "scale": 2**-11,
                "policy": "fixed_signed_unit_interface",
            },
            **{
                f"conv{index}_input": {
                    "clip": 0.5,
                    "scale": 2**-11,
                    "bits": 12,
                    "quantile": 0.9999,
                    "batches": 2,
                }
                for index in range(4)
            },
        },
        dataset_name="unit",
        seed=4,
        n_bits_a=12,
        n_bits_w=12,
        quant_calibration_quantile=0.9999,
        quant_calibration_batches=2,
        DPD_hidden_size=3,
        DPD_num_layers=2,
        tcn_kernel_size=5,
        tcn_dilation_base=2,
        rounding_policy_mode=rounding_policy_mode,
    )
    quant_project = SimpleNamespace(
        quant=True,
        n_bits_w=12,
        n_bits_a=12,
        pretrained_model="",
        quant_dir_label="training_artifact_test",
        DPD_backbone="fexlite_causal_tcn",
        quant_calibration_batches=2,
        quant_calibration_quantile=0.9999,
        rounding_policy_mode=rounding_policy_mode,
    )
    qat_model = get_quant_model(
        quant_project,
        CoreModel(
            2, 3, 2, "fexlite_causal_tcn",
            tcn_kernel_size=5, tcn_dilation_base=2,
        ),
    )
    torch.save(qat_model.state_dict(), source)
    return project


class TrainingArtifactTests(unittest.TestCase):
    def test_pa_checkpoint_is_atomically_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "logger" / "pa.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"new-pa")
            target = root / "owned" / "pa.pt"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"stale-pa")
            proj = SimpleNamespace(
                path_save_file_best=str(source),
                logger=SimpleNamespace(checkpoint_save_count=1),
            )
            published = publish_checkpoint(proj, str(target), "PA")
            self.assertEqual(published, target.resolve())
            self.assertEqual(target.read_bytes(), b"new-pa")

    def test_fp32_dpd_publication_does_not_require_qat_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "logger" / "dpd.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fp32-dpd")
            target = root / "owned" / "dpd.pt"
            proj = SimpleNamespace(
                quant=False,
                path_save_file_best=str(source),
                dpd_output_checkpoint=str(target),
                qat_output_checkpoint="",
                logger=SimpleNamespace(checkpoint_save_count=1),
            )
            published = _publish_dpd_artifacts(proj)
            self.assertEqual(published, target.resolve())
            self.assertEqual(target.read_bytes(), b"fp32-dpd")
            self.assertFalse(target.with_suffix(".calibration.json").exists())

    def test_explicit_pa_checkpoint_overrides_legacy_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "pa.pt"
            explicit.write_bytes(b"pa")
            proj = SimpleNamespace(pa_checkpoint=str(explicit))
            resolved = _resolve_pa_checkpoint(proj, root / "missing-legacy.pt")
            self.assertEqual(resolved, explicit.resolve())

    def test_checkpoint_and_matching_sidecars_are_published(self):
        with tempfile.TemporaryDirectory() as directory:
            proj = _project(Path(directory))
            target = Path(proj.qat_output_checkpoint)
            target.parent.mkdir(parents=True)
            target.write_bytes(b"stale-external")

            published = _publish_qat_artifacts(proj)

            self.assertEqual(published, target.resolve())
            self.assertEqual(target.read_bytes(), Path(proj.path_save_file_best).read_bytes())
            source = Path(proj.path_save_file_best)
            for suffix in (".calibration.json", ".model_spec.json"):
                source_document = json.loads(source.with_suffix(suffix).read_text())
                target_document = json.loads(target.with_suffix(suffix).read_text())
                self.assertEqual(source_document, target_document)
            self.assertEqual(
                json.loads(target.with_suffix(".model_spec.json").read_text())["dilations"],
                [1, 2],
            )
            calibration = json.loads(target.with_suffix(".calibration.json").read_text())
            self.assertEqual(
                calibration["checkpoint_sha256"],
                hashlib.sha256(target.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                calibration["final_effective_quantizers"]["input_quantizer.scale"],
                {"effective_scale": 2**-11, "scale_exponent": -11},
            )
            expected_policy = rounding_policy_record("baseline_rne")
            self.assertEqual(calibration["rounding_policy"], expected_policy)
            self.assertEqual(
                json.loads(target.with_suffix(".model_spec.json").read_text())[
                    "rounding_policy"
                ],
                expected_policy,
            )

            export_dir = target.parent / "rtl_export"
            manifest = export_fexlite_qat_rtl(target, export_dir, golden_length=4)
            self.assertEqual(
                manifest["provenance"]["training_sidecars"]["status"],
                "validated",
            )

    def test_global_floor_sidecars_bind_canonical_policy_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proj = _project(root, rounding_policy_mode=GLOBAL_FLOOR)
            # This unit fixture does not run calibration, so add the explicit
            # pre-HardSwish records required by the global-floor sidecar.
            proj.quant_calibration.update({
                f"hardswish{index}_input": {
                    "clip": 0.5,
                    "design_clip": 3.0,
                    "scale": 2**-11,
                    "bits": 12,
                    "rounding": "discard_lsb_signed_floor",
                    "quantile": 0.9999,
                    "batches": 2,
                }
                for index in range(3)
            })
            for index in range(4):
                proj.quant_calibration[f"conv{index}_input"]["rounding"] = (
                    "discard_lsb_signed_floor"
                )
            checkpoint = _publish_qat_artifacts(proj)
            expected = rounding_policy_record(GLOBAL_FLOOR)
            for suffix in (".calibration.json", ".model_spec.json"):
                sidecar = json.loads(checkpoint.with_suffix(suffix).read_text())
                self.assertEqual(sidecar["rounding_policy_mode"], GLOBAL_FLOOR)
                self.assertEqual(sidecar["rounding_policy"], expected)
            manifest = export_fexlite_qat_rtl(
                checkpoint,
                root / "global_export",
                golden_length=4,
                rounding_policy_mode=GLOBAL_FLOOR,
            )
            self.assertEqual(
                manifest["quantization"]["rounding_policy"], expected
            )

    def test_narrow_prehs_rne_sidecar_is_exportable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proj = _project(
                root, rounding_policy_mode=RESEARCH_PREHS_INPUT_RNE
            )
            proj.quant_calibration.update({
                f"hardswish{index}_input": {
                    "clip": 0.5,
                    "design_clip": 3.0,
                    "scale": 2**-11,
                    "bits": 12,
                    "rounding": ROUND_TO_NEAREST_TIES_TO_EVEN,
                    "quantile": 0.9999,
                    "batches": 2,
                }
                for index in range(3)
            })
            checkpoint = _publish_qat_artifacts(proj)
            manifest = export_fexlite_qat_rtl(
                checkpoint,
                root / "prehs_rne_export",
                golden_length=4,
                rounding_policy_mode=RESEARCH_PREHS_INPUT_RNE,
            )
            self.assertEqual(
                manifest["quantization"]["rounding_policy"],
                rounding_policy_record(RESEARCH_PREHS_INPUT_RNE),
            )
            self.assertTrue(all(
                layer.get("hardswish_input") is not None
                for layer in manifest["model"]["layers"][:-1]
            ))

    def test_preexisting_checkpoint_is_not_published_without_new_save(self):
        with tempfile.TemporaryDirectory() as directory:
            proj = _project(Path(directory), save_count=0)
            target = Path(proj.qat_output_checkpoint)
            target.parent.mkdir(parents=True)
            target.write_bytes(b"preexisting")
            with self.assertRaisesRegex(RuntimeError, "did not save a new best"):
                _publish_qat_artifacts(proj)
            self.assertEqual(target.read_bytes(), b"preexisting")

    def test_export_rejects_sidecar_bound_to_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proj = _project(root)
            checkpoint = _publish_qat_artifacts(proj)
            sidecar_path = checkpoint.with_suffix(".calibration.json")
            sidecar = json.loads(sidecar_path.read_text())
            sidecar["checkpoint_sha256"] = "0" * 64
            sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA mismatch"):
                export_fexlite_qat_rtl(
                    checkpoint, root / "rejected_export", golden_length=4
                )


if __name__ == "__main__":
    unittest.main()
