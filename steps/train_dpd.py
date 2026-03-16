__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import os
import torch
import models as model
from project import Project
from utils.util import count_net_params
import sys
sys.path.append('../..')
from quant import get_quant_model

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
                             num_dvr_units=proj.num_dvr_units)
    n_net_pa_params = count_net_params(net_pa)
    print("::: Number of PA Model Parameters: ", n_net_pa_params)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)

    # Load Pretrained PA Model
    path_pa_model = os.path.join('save', proj.dataset_name, 'train_pa', pa_model_id + '.pt')
    net_pa.load_state_dict(torch.load(path_pa_model, map_location='cpu'))

    # Instantiate DPD Model
    net_dpd = model.CoreModel(input_size=input_size,
                              hidden_size=proj.DPD_hidden_size,
                              num_layers=proj.DPD_num_layers,
                              backbone_type=proj.DPD_backbone,
                              window_size=proj.window_size,
                              num_dvr_units=proj.num_dvr_units,
                              thx=proj.thx,
                              thh=proj.thh)
    
    net_dpd = get_quant_model(proj, net_dpd)
    
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

        # Compute PA-only response for plotting (PA is frozen, so this is constant)
        pa_only_data = {}
        net_pa_inference = net_cas.pa_model
        net_pa_inference.eval()
        with torch.no_grad():
            if proj.eval_val:
                input_data_val = np.concatenate([b[0].numpy() for b in val_loader], axis=0)
                val_input_t = torch.tensor(input_data_val, dtype=torch.float32).to(proj.device)
                pa_only_data['val'] = net_pa_inference(val_input_t).cpu().numpy()
            if proj.eval_test:
                input_data_test = np.concatenate([b[0].numpy() for b in test_loader], axis=0)
                test_input_t = torch.tensor(input_data_test, dtype=torch.float32).to(proj.device)
                pa_only_data['test'] = net_pa_inference(test_input_t).cpu().numpy()

            # Load full dataset for APA constellation plotting
            # Use measured PA output from CSV (not PA model prediction)
            if needs_full_seq_constellation(proj.dataset_name):
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
