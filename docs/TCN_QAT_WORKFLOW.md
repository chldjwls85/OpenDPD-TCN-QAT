[English](TCN_QAT_WORKFLOW.md) | [한국어](TCN_QAT_WORKFLOW_KO.md)

# OpenDPD-TCN-QAT

OpenDPD-TCN-QAT is the official GitHub fork of
[lab-emi/OpenDPD](https://github.com/lab-emi/OpenDPD) maintained for DPD-Flow.
DPD-Flow consumes a pinned commit of this repository as a Git submodule. The
fork preserves the measured-data, frozen-PA, and metric infrastructure of
OpenDPD while adding a native, fail-closed quantization-aware training (QAT)
path for the causal FExLite TCN used by the RTL backend.

The power-amplifier model remains a frozen software surrogate. Only the DPD
model is quantized, exported, and lowered to RTL.

## What this fork adds

- parameterized hidden width `H`, temporal depth `L`, kernel size `K`, and
  dilation base;
- signed raw-I/Q fake quantization before FEx and signed output quantization
  after the residual;
- independent Conv1d weight/activation quantizers and train-split-only
  power-of-two activation calibration;
- persistent checkpoint topology metadata for unambiguous reconstruction;
- a versioned integer export containing weights, scales, causal delays,
  numeric rules, hashes, and layerwise golden vectors;
- exact integer verification and full-test evaluation through the same frozen
  DGRU PA; and
- a manifest-only boundary to the frontend-neutral TCN-Compiler backend.

## Installation

For standalone frontend development, clone this repository and install it in
editable mode:

```bash
git clone https://github.com/chldjwls85/OpenDPD-TCN-QAT.git
cd OpenDPD-TCN-QAT
python3 -m pip install -e .
```

For an integrated checkout, clone DPD-Flow recursively and install its three
local packages from the DPD-Flow root:

```bash
git clone --recurse-submodules https://github.com/chldjwls85/DPD-Flow.git
cd DPD-Flow
python3 -m pip install -e .
python3 -m pip install -e ./flow
python3 -m pip install -e ./frontend/OpenDPD-TCN-QAT
```

The frontend requires Python 3.10 or newer and PyTorch 2.0 or newer. A
CUDA-enabled PyTorch installation is recommended for training; export and
integer verification also run on CPU.

## Canonical FExLite TCN workflow

Run the following frontend commands from this repository root, or from
`frontend/OpenDPD-TCN-QAT/` inside DPD-Flow.

The integrated runner can execute the complete OpenDPD learning order instead
of requiring prepared checkpoints: `train_pa` publishes an explicit PA output,
floating-point `train_dpd` publishes an explicit FP32 DPD output, and QAT uses
those exact runner-owned artifacts. Use
`flow/configs/h13_a14w14_seed4_fulltrain.json` from the monorepo root for that
resumable path. The commands below show the final QAT step directly.

### 1. Quantization-aware training

This H13/A14W14 example starts from an architecture-compatible FP32 DPD and an
explicit frozen-PA checkpoint:

```bash
python3 main.py --step train_dpd --dataset_name DPA_200MHz \
  --DPD_backbone fexlite_causal_tcn \
  --DPD_hidden_size 13 --DPD_num_layers 4 \
  --tcn_kernel_size 5 --tcn_dilation_base 2 \
  --quant --n_bits_a 14 --n_bits_w 14 \
  --pretrained_model /path/to/DPD_FP32.pt \
  --pa_checkpoint /path/to/PA.pt \
  --qat_output_checkpoint /path/to/DPD_QAT.pt \
  --n_epochs 200 --accelerator cuda
```

The dataset CSV values are loaded as FP32 measurements. The model fake-quantizes
raw I/Q immediately on entry, so A14 includes the external activation boundary;
the source CSV files are not rewritten as integer files. Calibration uses only
the training split. The published checkpoint is accompanied by calibration and
model-spec JSON sidecars.

### 2. Integer export

The export directory must be new; the exporter does not overwrite an existing
package.

```bash
python3 scripts/export_fexlite_qat_rtl.py \
  --checkpoint /path/to/DPD_QAT.pt \
  --pa-checkpoint /path/to/PA.pt \
  --dataset-name DPA_200MHz \
  --input datasets/DPA_200MHz/test_input.csv \
  --output-dir /path/to/rtl_export
```

The portable package contains `manifest.json`, `weights/*.mem`, and
`golden_vectors/*.mem`. TCN-Compiler consumes only this package; it does not import
PyTorch modules, checkpoints, or dataset loaders.

### 3. Integer verification

```bash
python3 scripts/verify_fexlite_qat_rtl.py \
  --manifest /path/to/rtl_export/manifest.json
```

The verifier reloads the exported memories and requires 0-LSB agreement with
every published integer golden trace.

### 4. Frozen-PA evaluation

```bash
python3 scripts/evaluate_fexlite_integer_pa.py \
  --manifest /path/to/rtl_export/manifest.json \
  --pa-checkpoint /path/to/PA.pt \
  --qat-checkpoint /path/to/DPD_QAT.pt \
  --dataset-name DPA_200MHz --split test --protocol segmented \
  --device cuda --output /path/to/integer_pa_metrics.json
```

The segmented protocol resets TCN history and frozen-PA state at each
`nperseg` boundary. It reports integer-DPD RF metrics and, when
`--qat-checkpoint` is supplied, the separate fake-QAT-versus-integer comparison.

For a complete resumable PA-modeling-to-synthesis run, use `dpdflow` from the
monorepo root. See the [DPD-Flow overview](https://github.com/chldjwls85/DPD-Flow)
and [integrated flow](https://github.com/chldjwls85/DPD-Flow/tree/main/flow).

## Bundled datasets

This fork bundles five upstream OpenDPD datasets: `APA_200MHz`,
`APA_200MHz_b`, `DPA_160MHz`, `DPA_200MHz`, and `MyCustomPA`. The four measured
PA datasets use split train/validation/test CSV files; `MyCustomPA` is the
single-CSV custom-dataset fixture. Regenerable diagnostic PNGs are intentionally
excluded from Git and can be recreated with each dataset's `plot_dataset.py`.
`DPA_200MHz` remains the qualified DPD-Flow baseline; bundling another
dataset does not implicitly qualify its checkpoint or evaluation protocol. See
the [dataset contract](../datasets/README_TCN_QAT.md) before changing data or protocol.

## Documentation

- [TCN-QAT architecture and export boundary](TCN_QAT_ARCHITECTURE.md)
- [TCN-Compiler numeric contract](https://github.com/chldjwls85/DPD-Flow/blob/main/docs/NUMERIC_CONTRACT.md)
- [Flow validation policy](https://github.com/chldjwls85/DPD-Flow/blob/main/flow/docs/VALIDATION.md)
- [Upstream provenance and license](UPSTREAM_AND_LICENSE.md)

`examples/api_usage_example.py` is a generic, state-changing upstream API demo:
it trains several models and creates a custom dataset. It is not the canonical
FExLite TCN QAT or DPD-Flow workflow.

## Upstream attribution

This frontend is a modified work derived from
[lab-emi/OpenDPD](https://github.com/lab-emi/OpenDPD). Upstream authorship,
and the Apache License 2.0 text are preserved; the publication citations remain
available in the pinned upstream README. The exact source commit and the
fork-specific change list are recorded in the
[provenance note](UPSTREAM_AND_LICENSE.md).
