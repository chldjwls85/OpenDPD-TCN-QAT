[English](TCN_QAT_ARCHITECTURE.md) | [한국어](TCN_QAT_ARCHITECTURE_KO.md)

# TCN-QAT Architecture

## Purpose

Upstream OpenDPD supports FP32 TCN training, but it does not provide native QAT
for a Conv1d-based TCN, raw/output I/Q boundary quantization, and RTL integer
export as one verified path. This fork adds that path while preserving the
bundled five-dataset collection, frozen PA, and training/metric infrastructure.

## Model

FExLite features appear in the following order.

```text
I, Q, p=I²+Q², p², I·p, Q·p
```

The network consists of a 6→H pointwise input projection, `L` H-channel
depthwise causal convolutions, an H→2 output projection, and a raw-I/Q residual.
Temporal layer `i` uses dilation `dilation_base**i`. H, L, K, and the dilation
base are configurable through constructor and OpenDPD CLI arguments.

Each temporal convolution creates left causal context with PyTorch padding and
then removes the right padding with `Chomp1d`. The test suite includes a
causality test that changes future samples and confirms that earlier outputs do
not change.

## Checkpoint Topology Record

The model contains a persistent `_rtl_spec=[version,L,K,dilation_base]` buffer.
Because the QAT checkpoint preserves this value, the exporter can reconstruct
the layer count and dilations without a filename convention. For canonical
legacy checkpoints without `_rtl_spec`, the exporter reads H/L/K from the
convolution weight shapes and explicitly applies the legacy dilation base of 2.

## Explicit Training I/O and Saved Names

A full reproducible run publishes three explicit artifacts in dependency
order. `train_pa --pa_output_checkpoint` publishes the PA surrogate. A
non-quantized `train_dpd` invocation receives that exact PA through
`--pa_checkpoint` and publishes its floating-point DPD through
`--dpd_output_checkpoint`. The QAT invocation receives both artifacts through
`--pa_checkpoint` and `--pretrained_model`, then publishes the new QAT result
through `--qat_output_checkpoint`.

If the PA input path is omitted, the previous `save/<dataset>/train_pa/`
discovery rule remains available for backward compatibility. The integrated
flow never relies on that implicit lookup: it either produces a runner-owned
checkpoint or passes an explicit immutable checkpoint.

PA, FP32 DPD, and QAT results are published only if the training logger saves a
best checkpoint at least once during the current invocation. The implementation
writes a temporary file completely in the output directory and then replaces
the target with `os.replace`. QAT calibration and model-spec JSON files are
written atomically with matching content. Publication fails if the current run
saves nothing, even if an old output file remains.

Internal logger filenames for the FExLite TCN include L, K, and the dilation
base as well as H. Under QAT, they also include the A/W bit widths, which
prevents experiments with different topologies or precisions from overwriting
the same internal path. Naming rules for non-TCN models are unchanged, and the
inference path makes one additional attempt with the legacy TCN name if the new
name is absent.

## QAT Environment

`FExLiteTCNQuantEnv` is separate from the GRU quantization environment. It
replaces every Conv1d with `INT_Conv1D`, and each layer owns independent weight
and activation quantizers. It copies pretrained biases and initializes each
weight scale to the smallest power of two that covers the signed code range
without clipping the maximum weight magnitude.

The raw-input quantizer precedes FEx, and the final-output quantizer follows the
residual. For A-bit activation codes, the physical I/Q boundary scale is
`2^(1-A)`, with a zero point of 0. The training dataset is therefore read as
FP32, but it is fake-quantized to the selected signed grid immediately after
entering the model graph.

Calibration caches the first N batches of the training loader once, then
processes layers sequentially so that every Conv1d observes the same samples.
The raw/output interface scales remain fixed; only internal activation scales
are set to powers of two that cover the absolute quantile.

## Export and Equivalence Boundary

The exporter publishes a self-contained `opendpd_fexlite_qat_rtl_export` v1
package containing `manifest.json`, `weights/*.mem`, and
`golden_vectors/*.mem`. Every package path is relative, and the manifest records
the SHA-256 identity of each memory. TCN-Compiler consumes only this package: it does
not import this frontend's Python modules, checkpoint, or dataset loader.

The exact integer evaluator defined by the export is the hardware reference.
Reloading the package must reproduce every integer golden trace at 0 LSB, and
RTL must independently match those traces at 0 LSB. Fake-QAT execution is a
different equivalence level: PyTorch operation and quantizer placement can
produce small code differences, so fake-QAT-versus-integer results must be
measured and reported separately from integer-versus-RTL agreement.

The canonical rounding, saturation, FEx, tap-order, HardSwish, and residual
arithmetic is maintained once in the monorepo
[numeric contract](https://github.com/chldjwls85/DPD-Flow/blob/main/docs/NUMERIC_CONTRACT.md).
See the
[validation policy](https://github.com/chldjwls85/DPD-Flow/blob/main/flow/docs/VALIDATION.md)
for the three distinct
agreement claims and the qualified baseline result.
