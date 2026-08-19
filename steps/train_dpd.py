__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"
# Modified in the OpenDPD-TCN-QAT fork to calibrate full-I/O TCN QAT.

import os
import json
import hashlib
import math
import tempfile
from pathlib import Path
import torch
import models as model
from project import Project
from utils.util import count_net_params
import sys
sys.path.append('../..')
from quant import get_quant_model
from quant.rounding_policy import rounding_policy_record
from steps.training_artifacts import atomic_copy as _atomic_copy
from steps.training_artifacts import publish_checkpoint


def _resolve_pa_checkpoint(proj: Project, legacy_path: str | Path) -> Path:
    """Resolve and validate the explicit PA input while preserving legacy use."""
    configured = getattr(proj, 'pa_checkpoint', '')
    checkpoint = Path(configured or legacy_path).expanduser().resolve()
    if not checkpoint.is_file():
        source = 'explicit --pa_checkpoint' if configured else 'legacy save/ path'
        raise FileNotFoundError(f"PA checkpoint from {source} does not exist: {checkpoint}")
    return checkpoint


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                'w', dir=path.parent, prefix=f'.{path.name}.', delete=False) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _final_quantizer_scales(checkpoint: Path) -> dict[str, dict]:
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('trained QAT checkpoint must contain a state_dict mapping')
    result = {}
    for name, value in sorted(state.items()):
        if not name.endswith('quantizer.scale'):
            continue
        raw_scale = float(torch.as_tensor(value).reshape(-1)[0].item())
        if not math.isfinite(raw_scale) or raw_scale == 0.0:
            raise ValueError(f'invalid final quantizer scale {name}: {raw_scale}')
        exponent = int(round(math.log2(abs(raw_scale))))
        result[name] = {
            'effective_scale': 2.0 ** exponent,
            'scale_exponent': exponent,
        }
    if not result:
        raise ValueError('trained QAT checkpoint contains no quantizer scales')
    precision_keys = [
        key for key in state if key.endswith('backbone.network.0.n_bits_a')
    ]
    if len(precision_keys) != 1:
        raise ValueError('trained QAT checkpoint has no unambiguous activation precision')
    expected_boundary = 2.0 ** (1 - int(state[precision_keys[0]].item()))
    for name in ('input_quantizer.scale', 'output_quantizer.scale'):
        stored = state.get(name)
        stored_scale = (
            float(torch.as_tensor(stored).reshape(-1)[0].item())
            if stored is not None else None
        )
        if (
            name not in result
            or result[name]['effective_scale'] != expected_boundary
            or stored_scale != expected_boundary
        ):
            raise ValueError(
                f'{name} is not the immutable fixed signed-unit interface scale '
                f'{expected_boundary}'
            )
    return result


def _qat_sidecars(proj: Project, checkpoint: Path) -> dict[str, dict]:
    calibration = getattr(proj, 'quant_calibration', None)
    if not calibration:
        return {}
    checkpoint_sha256 = _sha256_file(checkpoint)
    final_scales = _final_quantizer_scales(checkpoint)
    rounding_policy = rounding_policy_record(getattr(
        proj, 'rounding_policy_mode', 'baseline_rne'
    ))
    requested_pre_hs_bits = int(getattr(proj, 'pre_hardswish_bits', 0))
    pre_hardswish_bits = (
        int(proj.n_bits_a) if requested_pre_hs_bits == 0
        else requested_pre_hs_bits
    )
    return {
        '.calibration.json': {
            'format': 'opendpd_tcn_qat_calibration',
            'format_version': 1,
            'dataset_name': proj.dataset_name,
            'split': 'train',
            'seed': proj.seed,
            'activation_bits': proj.n_bits_a,
            'weight_bits': proj.n_bits_w,
            'pre_hardswish_bits': pre_hardswish_bits,
            'rounding_policy_mode': getattr(
                proj, 'rounding_policy_mode', 'baseline_rne'
            ),
            'rounding_policy': rounding_policy,
            'quantile': proj.quant_calibration_quantile,
            'maximum_batches': proj.quant_calibration_batches,
            'checkpoint_sha256': checkpoint_sha256,
            'calibration_quantizers': calibration,
            'final_effective_quantizers': final_scales,
        },
        '.model_spec.json': {
            'format': 'opendpd_fexlite_causal_tcn_spec',
            'format_version': 1,
            'hidden_channels': proj.DPD_hidden_size,
            'temporal_layers': proj.DPD_num_layers,
            'kernel_size': proj.tcn_kernel_size,
            'dilation_base': proj.tcn_dilation_base,
            'dilations': [
                proj.tcn_dilation_base ** index
                for index in range(proj.DPD_num_layers)
            ],
            'activation_bits': proj.n_bits_a,
            'weight_bits': proj.n_bits_w,
            'pre_hardswish_bits': pre_hardswish_bits,
            'rounding_policy_mode': getattr(
                proj, 'rounding_policy_mode', 'baseline_rne'
            ),
            'rounding_policy': rounding_policy,
            'activation': 'hardswish',
        },
    }


def _publish_qat_artifacts(proj: Project) -> Path:
    """Publish a checkpoint saved by this training invocation and its sidecars."""
    source = publish_checkpoint(proj, '', 'QAT')

    # The logger-owned best checkpoint must exist before its immutable content
    # hash can bind the calibration record.  Sidecars deliberately do not hash
    # themselves, avoiding a circular provenance dependency.
    documents = _qat_sidecars(proj, source)
    requested = getattr(proj, 'qat_output_checkpoint', '')
    if requested and len(documents) != 2:
        raise RuntimeError(
            '--qat_output_checkpoint requires calibrated FExLite TCN QAT sidecars'
        )

    # Keep the legacy logger-owned checkpoint self-describing as well.
    for suffix, document in documents.items():
        _atomic_json(source.with_suffix(suffix), document)

    published = source
    if requested:
        published = Path(requested).expanduser().resolve()
        _atomic_copy(source, published)
        for suffix, document in documents.items():
            _atomic_json(published.with_suffix(suffix), document)

    proj.published_qat_checkpoint = str(published)
    return published


def _publish_dpd_artifacts(proj: Project) -> Path:
    """Publish either an FP32 DPD checkpoint or a QAT checkpoint."""

    if getattr(proj, 'quant', False):
        if getattr(proj, 'dpd_output_checkpoint', ''):
            raise ValueError(
                '--dpd_output_checkpoint is for floating-point DPD training; '
                'use --qat_output_checkpoint with --quant'
            )
        return _publish_qat_artifacts(proj)
    if getattr(proj, 'qat_output_checkpoint', ''):
        raise ValueError('--qat_output_checkpoint requires --quant')
    published = publish_checkpoint(
        proj, getattr(proj, 'dpd_output_checkpoint', ''), 'FP32 DPD'
    )
    proj.published_dpd_checkpoint = str(published)
    return published


def main(proj: Project):
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    # Set Accelerator Device
    proj.set_device()

    # Build Dataloaders
    (train_loader, val_loader, test_loader), input_size = proj.build_dataloaders()

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate PA Model
    net_pa = model.CoreModel(input_size=input_size,
                             hidden_size=proj.PA_hidden_size,
                             num_layers=proj.PA_num_layers,
                             backbone_type=proj.PA_backbone,
                             window_size=proj.window_size,
                             num_dvr_units=proj.num_dvr_units,
                             thx=proj.thx,
                             thh=proj.thh)
    n_net_pa_params = count_net_params(net_pa)
    print("::: Number of PA Model Parameters: ", n_net_pa_params)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)

    # Load Pretrained PA Model
    path_pa_model = _resolve_pa_checkpoint(
        proj,
        os.path.join('save', proj.dataset_name, 'train_pa', pa_model_id + '.pt')
    )
    print("::: PA Model Checkpoint: ", path_pa_model)
    net_pa.load_state_dict(torch.load(path_pa_model, map_location='cpu'))

    # Instantiate DPD Model
    net_dpd = model.CoreModel(input_size=input_size,
                              hidden_size=proj.DPD_hidden_size,
                              num_layers=proj.DPD_num_layers,
                              backbone_type=proj.DPD_backbone,
                              window_size=proj.window_size,
                              num_dvr_units=proj.num_dvr_units,
                              thx=proj.thx,
                              thh=proj.thh,
                              tcn_kernel_size=proj.tcn_kernel_size,
                              tcn_dilation_base=proj.tcn_dilation_base)

    net_dpd = get_quant_model(proj, net_dpd)
    if proj.collect_delta_stats and hasattr(net_dpd.backbone, 'set_debug'):
        net_dpd.backbone.set_debug(1)

    print("::: DPD Model: ", net_dpd)
    n_net_dpd_params = count_net_params(net_dpd)
    print("::: Number of DPD Model Parameters: ", n_net_dpd_params)
    dpd_model_id = proj.gen_dpd_model_id(n_net_dpd_params)

    # Instantiate Cascaded Model
    net_cas = model.CascadedModel(dpd_model=net_dpd, pa_model=net_pa)

    # Freeze PA Model
    net_cas.freeze_pa_model()

    # Move the network to the proper device
    net_cas = net_cas.to(proj.device)

    if (proj.quant and hasattr(proj, 'quant_env')
            and hasattr(proj.quant_env, 'calibrate')):
        proj.quant_calibration = proj.quant_env.calibrate(
            train_loader, proj.device
        )

    ###########################################################################################################
    # Logger, Loss and Optimizer Settings
    ###########################################################################################################
    # Build Logger
    proj.build_logger(model_id=dpd_model_id)

    # Select Loss function
    criterion = proj.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = proj.build_optimizer(net=net_cas)

    ###########################################################################################################
    # Plotting Setup
    ###########################################################################################################
    plot_dir = None
    pa_only_data = None
    full_input_iq = None
    full_pa_only_c = None
    if proj.plot:
        from utils.plotting import get_plot_dir_train_dpd, needs_full_seq_constellation, load_full_dataset_iq
        import numpy as np
        plot_dir = get_plot_dir_train_dpd(proj.dataset_name, pa_model_id, dpd_model_id)

        # Load actual measured PA output from CSV for plotting
        # (not the PA model prediction, which smooths out spectral regrowth)
        from modules.data_collector import load_dataset as _load_raw
        _, _, _, y_val_raw, _, y_test_raw = _load_raw(dataset_name=proj.dataset_name)
        nperseg = proj.args.nperseg

        pa_only_data = {}
        if proj.eval_val:
            n_seg = len(y_val_raw) // nperseg
            pa_only_data['val'] = y_val_raw[:n_seg * nperseg].reshape(n_seg, nperseg, 2)
        if proj.eval_test:
            n_seg = len(y_test_raw) // nperseg
            pa_only_data['test'] = y_test_raw[:n_seg * nperseg].reshape(n_seg, nperseg, 2)

        # Load full dataset for full-sequence plotting (constellation, PSD, AM/AM, AM/PM)
        full_input_iq, full_output_iq = load_full_dataset_iq(proj.dataset_name)
        full_pa_only_c = full_output_iq[:, 0] + 1j * full_output_iq[:, 1]

    # Build metadata for dashboard
    metadata = {
        'dataset': proj.dataset_name,
        'step': 'train_dpd',
        'backbone': proj.DPD_backbone.upper(),
        'hidden_size': proj.DPD_hidden_size,
        'n_params': n_net_dpd_params,
        'model_id': dpd_model_id,
    }

    ###########################################################################################################
    # Training
    ###########################################################################################################
    proj.train(net=net_cas,
               criterion=criterion,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               train_loader=train_loader,
               val_loader=val_loader,
               test_loader=test_loader,
               best_model_metric='ACLR_AVG',
               plot_dir=plot_dir,
               pa_only_data=pa_only_data,
               metadata=metadata,
               full_input_iq=full_input_iq,
               full_pa_only_c=full_pa_only_c)

    published = _publish_dpd_artifacts(proj)
    print("::: Published DPD Checkpoint: ", published)
