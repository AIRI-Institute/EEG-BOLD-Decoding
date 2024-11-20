import os
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import RandomSampler

from bw_linker.brain_wave_pipeline.eegfmri_dataset import build_full_dataset
from bw_linker.brain_wave_pipeline.brain_wave_linker import BrainWaveLinkerSystem
from bw_linker.data_preprocessing.load_brain_data import load_multiple_files, split_datasets
from bw_linker.utils.constants import EEG_SAMPLING_RATE
from bw_linker.utils.helpers import get_run_name, get_window_sizes_sample, print_wrapped_text


def get_torch_datasets(project_root: str, config: dict):
    """
    Build PyTorch datasets

    Parameters
    ----------
    project_root : str
        A path to the root of the project
    config : dict
        A configuration file of the whole experiment

    Returns
    -------
    train_dataset : torch.utils.data.ConcatDataset
        A PyTorch ConcatDataset which unites all task-specific train data
    validation_dataset : torch.utils.data.ConcatDataset
        A PyTorch ConcatDataset which unites all task-specific validation data
    test_dataset : torch.utils.data.ConcatDataset
        A PyTorch ConcatDataset which unites all task-specific test data
    eeg_channels : list[str]
        A list of EEG channels loaded from files
    rois : list[str]
        A list of fMRI ROIs loaded from files
    """

    datasets, rois, eeg_channels, sampling_rates_ratio = load_multiple_files(
        root=os.path.join(project_root, config['root']),
        subjects=config['subjects'],
        desired_fmri_sampling_rate=config['fmri_sampling_rate'],
        fmri_interpolation_type=config['fmri_interpolation_type'], rois=config['rois'],
        eeg_channels=config['eeg_channels'], delay_sec=config['delay_sec'],
        separate_global_trend=config['separate_global_trend'],
        starting_point_sec=config['starting_point_sec'],
        rois_for_global_trend=config['rois_for_global_trend'],
        roi_folder=config['roi_folder']
    )

    if config['eeg_channels'] is None:
        config['eeg_channels'] = eeg_channels
    if config['rois'] is None:
        config['rois'] = rois

    print('EEG channels:', config['eeg_channels'])
    print('fMRI rois:', config['rois'])

    train_datasets, validation_datasets, test_datasets = split_datasets(
        datasets=datasets, sampling_rates_ratio=sampling_rates_ratio
    )
    all_datasets = {'train': train_datasets, 'validation': validation_datasets, 'test': test_datasets}

    print(f'Sampling rates ratio: {sampling_rates_ratio}')
    for name, dataset in all_datasets.items():
        print(f'Split: {name}')
        for (eeg_path, fmri_path), value in dataset.items():
            print(
                f'Path to EEG: {eeg_path}, total EEG size: {value["eeg"].shape}, total BOLD size: {value["fmri"].shape}'
            )
        print('\n')

    torch_datasets = {}

    for split_name in ['train', 'validation', 'test']:

        split_dataset, split_rois = build_full_dataset(
            datasets=all_datasets[split_name],
            fmri_window_size=config[f'{split_name}_dataset_params']['fmri_window_samples'],
            eeg_window_size=config[f'{split_name}_dataset_params']['eeg_window_samples'],
            sampling_rates_ratio=sampling_rates_ratio,
            eeg_channels=eeg_channels, fmri_rois=rois,
            stride=config[f'{split_name}_dataset_params']['stride_samples'],
            eeg_standardization_kwargs=config['eeg_standardization_kwargs'],
            fmri_standardization_kwargs=config['fmri_standardization_kwargs']
        )

        torch_datasets[split_name] = {'dataset': split_dataset, 'rois': split_rois}

    assert torch_datasets['train']['rois'] == torch_datasets['validation']['rois'] == torch_datasets['test']['rois'], (
        torch_datasets['train']['rois'], torch_datasets['validation']['rois'], torch_datasets['test']['rois']
    )
    rois = list(torch_datasets['train']['rois'])

    print('Train size:', len(torch_datasets['train']['dataset']))
    print('Validation size:', len(torch_datasets['validation']['dataset']))
    print('Test size:', len(torch_datasets['test']['dataset']))
    eeg_example, bold_example = torch_datasets['train']['dataset'][0]
    print(f'EEG size: {eeg_example.size()}, BOLD size: {bold_example.size()}')

    return (torch_datasets['train']['dataset'], torch_datasets['validation']['dataset'],
            torch_datasets['test']['dataset'], eeg_channels, rois)


def build_dataloader(dataset: torch.utils.data.Dataset, sampler_parameters: Optional[dict],
                     dataloader_parameters: dict):
    """
    Builds a dataloader

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A dataset for the loader
    sampler_parameters : dict or None
        A dict of the following structure:
            name: sampler name
            parameters: dict with sampler keyword arguments
        If None, no sampler is used
    dataloader_parameters : dict
        A dict with dataloader keyword arguments

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        A PyTorch dataloader
    """
    samplers = {
        'RandomSampler': RandomSampler
    }

    if sampler_parameters is not None:
        sampler = samplers[sampler_parameters['name']](dataset, **sampler_parameters['parameters'])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **dataloader_parameters)

    return dataloader


def extract_predicted_time_series(model: torch.nn.Module, ckpt_path: str, dataset: torch.utils.data.ConcatDataset,
                                  ds_name: str, device: str):
    """
    Does inference on a dataset and saves predicted and ground truth time series into a .csv file

    Parameters
    ----------
    model : torch.nn.Module
        A neural network
    ckpt_path : str
        A path to the checkpoint
    dataset : torch.utils.data.ConcatDataset
        A ConcatDataset instance, where every Dataset is from a different EEG-fMRI recording run
    ds_name : str
        A name of the dataset (e.g. train/validation/test)
    device : str
        Which device to use for inference (e.g. 'cpu', 'cuda')
    """
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location=device)
    state_dict = {}
    # Original checkpoint for a torch.nn.Module was inside a lightning module, therefore removing additional prefix
    for key, value in checkpoint['state_dict'].items():
        key = key.split('.')
        assert key[0] == 'model'
        new_key = '.'.join(key[1:])
        state_dict[new_key] = value
    model.load_state_dict(state_dict)
    model.eval()

    # Check ROI consistency
    roi_names = None
    for ds in dataset.datasets:
        if roi_names is None:
            roi_names = list(ds.get_rois())
        else:
            assert roi_names == list(ds.get_rois()), (roi_names, ds.get_rois())

    # Inference and save predictions
    corrs = {'task': []}
    for roi_name in roi_names:
        corrs[roi_name] = []

    save_root = os.path.dirname(os.path.dirname(ckpt_path))

    results_folder = os.path.join(save_root, 'time_series', ds_name)
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    for ds in dataset.datasets:

        eeg, gt_fmri, (eeg_data_path, fmri_data_path) = ds.get_all_data()
        task_info = get_run_name(fmri_data_path)
        eeg = eeg.unsqueeze(0)  # imitating batch dimension
        eeg.to(device)

        with torch.no_grad():
            pred_fmri = model(eeg)
            assert pred_fmri.size(0) == 1  # remove batch dimension
            pred_fmri = pred_fmri[0, ...]

        pred_fmri = pred_fmri.cpu().numpy()
        gt_fmri = gt_fmri.cpu().numpy()

        assert (pred_fmri.shape[-1] + 20) == gt_fmri.shape[-1], (
            f'{gt_fmri.shape[-1]} is expected to be 10 seconds longer as BOLD starts 5 seconds later '
            f'and other 5 seconds are cropped due to no padding in the model'
        )

        gt_fmri = gt_fmri[:, :-20]
        assert pred_fmri.shape == gt_fmri.shape

        ds_results = {}

        corrs['task'].append(task_info)
        for roi_idx, roi_name in enumerate(roi_names):
            roi_pred = pred_fmri[roi_idx, :]
            roi_gt = gt_fmri[roi_idx, :]
            ds_results[f'{roi_name}_pred'] = roi_pred
            ds_results[f'{roi_name}_gt'] = roi_gt
            corrs[roi_name].append(np.corrcoef(roi_pred, roi_gt)[0, 1])

        ds_results = pd.DataFrame(ds_results)
        ds_results.to_csv(os.path.join(results_folder, f'{task_info}.csv'), index=False)

    corrs = pd.DataFrame(corrs)
    corrs.to_csv(os.path.join(save_root, f'{ds_name}_full_correlations.csv'), index=False)


def run_full_pipeline(config, criterion_config, optimizer_config, scheduler_config, trainer_config, sampler_configs,
                      dataloader_configs, wandb_kwargs, model_config, project_root='.'):
    """
    Runs a full training pipeline for the neural network

    Parameters
    ----------
    config : dict
        A full config of the experiment
    criterion_config : dict
        A config of the following structure:
            name: A name of the required criterion
            kwargs: A dict with keyword arguments for a desired criterion
    optimizer_config : dict
        A config of the following structure:
            name: A name of the optimizer
            kwargs: A dictionary with keyword arguments for the Adam optimizer initialization
    scheduler_config : dict
        A dictionary with keyword arguments for the ReduceLROnPlateau learning rate scheduler initialization
    trainer_config : dict
        A config of the following structure:
        early_stopping: None (no early stopping) or dict with following parameters: monitor, mode, patience
        checkpointing: None (no checkpointing) or dict of keyword arguments for the ModelCheckpoint
        kwargs: dict of keyword arguments into the lightning.Trainer
    sampler_configs : dict
        A config of the following structure:
        train:
            name: sampler name
            parameters: dict with sampler keyword arguments. If None, no sampler is used
        validation:
            name: sampler name
            parameters: dict with sampler keyword arguments. If None, no sampler is used
        test:
            name: sampler name
            parameters: dict with sampler keyword arguments. If None, no sampler is used
    dataloader_configs : dict
        A config of the following structure:
            train: A dict with dataloader keyword arguments
            validation: A dict with dataloader keyword arguments
            test: A dict with dataloader keyword arguments
    wandb_kwargs : dict
        Keyword arguments for Weights and Biases WandbLogger
    model_config : dict
        A dictionary with additional model hyperparameters
    project_root : str
        A path to the root of the project
    """
    for data_mode in ['train', 'validation', 'test']:
        data_mode = f'{data_mode}_dataset_params'
        config[data_mode]['eeg_window_samples'] = get_window_sizes_sample(
            size_sec=config[data_mode]['eeg_window_sec'], sampling_rate=EEG_SAMPLING_RATE
        )

        config[data_mode]['fmri_window_samples'] = get_window_sizes_sample(
            size_sec=config[data_mode]['fmri_window_sec'], sampling_rate=config['fmri_sampling_rate']
        )

        config[data_mode]['stride_samples'] = get_window_sizes_sample(
            size_sec=config[data_mode]['stride_sec'], sampling_rate=config['fmri_sampling_rate']
        )

    # Dataset building

    train_dataset, validation_dataset, test_dataset, eeg_channels, rois = get_torch_datasets(
        project_root=project_root, config=config
    )

    # Building a model

    brain_wave_linker_net = BrainWaveLinkerSystem(
        nn_parameters=model_config,
        eeg_channel_names=eeg_channels,
        roi_names=rois,
        criterion_name=criterion_config['name'],
        criterion_kwargs=criterion_config['kwargs'],
        optimizer_name=optimizer_config['name'],
        optimizer_kwargs=optimizer_config['kwargs'],
        scheduler_kwargs=scheduler_config,
        dataframe_logging_modes=config['dataframe_logging_modes']
    )

    train_loader = build_dataloader(dataset=train_dataset, sampler_parameters=sampler_configs['train'],
                                    dataloader_parameters=dataloader_configs['train'])
    val_loader = build_dataloader(dataset=validation_dataset, sampler_parameters=sampler_configs['validation'],
                                  dataloader_parameters=dataloader_configs['validation'])
    test_loader = build_dataloader(dataset=test_dataset, sampler_parameters=None,
                                   dataloader_parameters=dataloader_configs['test'])
    print('train_loader', len(train_loader))
    print('val_loader', len(val_loader))
    print('test_loader', len(test_loader))

    wandb_logger = WandbLogger(name=config['wandb_run_name'], project=config['project_name'],
                               save_dir=os.path.join(project_root, 'wandb_logs', config['project_name']),
                               **wandb_kwargs)

    callbacks = [ModelCheckpoint(**trainer_config['checkpointing'])]

    if trainer_config['early_stopping'] is not None:
        callbacks.append(EarlyStopping(monitor=trainer_config['early_stopping']['monitor'],
                                       mode=trainer_config['early_stopping']['mode'],
                                       patience=trainer_config['early_stopping']['patience']))

    trainer = L.Trainer(callbacks=callbacks,
                        logger=wandb_logger,
                        **trainer_config['kwargs'])
    print(brain_wave_linker_net.model)
    trainer.fit(model=brain_wave_linker_net, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f'Training complete. Best model is available at: {best_model_path}')
    print_wrapped_text(header_text='MODEL', main_body=brain_wave_linker_net.model)
    print_wrapped_text(header_text='Experiment config', main_body=config)
    print_wrapped_text(header_text='Criterion config', main_body=criterion_config)
    print_wrapped_text(header_text='Optimizer config', main_body=optimizer_config)
    print_wrapped_text(header_text='Scheduler config', main_body=scheduler_config)
    print_wrapped_text(header_text='Trainer config', main_body=trainer_config)
    print_wrapped_text(header_text='wandb config', main_body=wandb_kwargs)
    print_wrapped_text(header_text='Model config', main_body=model_config)

    trainer.test(brain_wave_linker_net, dataloaders=test_loader, ckpt_path=best_model_path)

    for logging_mode in config['dataframe_logging_modes']:
        brain_wave_linker_net.log_metrics_tables(mode=logging_mode)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    extract_predicted_time_series(
        model=brain_wave_linker_net.model, ckpt_path=best_model_path, dataset=test_loader.dataset, ds_name='test',
        device=device
    )

    wandb.finish()
