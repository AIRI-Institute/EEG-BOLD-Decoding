from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from bw_linker.brain_wave_pipeline.criterions import build_criterion


class BrainWaveLinker(nn.Module):
    """
    A BrainWaveLinker neural network which predicts BOLD signal from EEG

    Parameters
    ----------
    in_channels : int
        A number of input EEG channels
    out_channels : int
        A number of output fMRI ROIs
    temporal_filter_kernel : int
        A size of the window for temporal filtering in seconds. If None, uses twice the delay
    dropout_prob : float
        Probability for the Dropout layer in the Pyramidal subsampling. Default: 0.25
    intermediate_channels : int or None
        Amount of hidden channels for intermediate layers. If None, uses out_channels. Default: None
    """
    def __init__(
            self, in_channels: int, out_channels: int, temporal_filter_kernel: int, dropout_prob: float = 0.25,
            intermediate_channels: Optional[int] = None
    ):
        super().__init__()

        if intermediate_channels is None:
            intermediate_channels = out_channels

        self.spatial_filer = nn.Conv1d(in_channels,  intermediate_channels, kernel_size=1)

        self.pyramidal_subsampling = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(in_channels=intermediate_channels, out_channels=intermediate_channels,
                      kernel_size=5, stride=5),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(in_channels=intermediate_channels, out_channels=intermediate_channels,
                      kernel_size=5, stride=5),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(in_channels=intermediate_channels, out_channels=intermediate_channels,
                      kernel_size=5, stride=5),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        self.temporal_filter = nn.Conv1d(
            intermediate_channels, out_channels, kernel_size=temporal_filter_kernel, padding=0
        )

    def forward(self, x: torch.Tensor):
        """
        Get predicted BOLD signal

        Parameters
        ----------
        x : torch.Tensor
            A tensor with EEG signal of size (batch, n_eeg_channels, n_eeg_times)

        Returns
        -------
        out : torch.Tensor
            A predicted BOLD tensor of size (batch, n_rois, n_fmri_times)
        """
        out_sf = self.spatial_filer(x)
        out_ps = self.pyramidal_subsampling(out_sf)
        out = self.temporal_filter(out_ps)
        return out


class BrainWaveLinkerSystem(L.LightningModule):
    """
    A Lightning module with training and inference code for the neural network

    Parameters
    ----------
    nn_parameters : dict
        A dictionary with parameters for neural network initialization
    roi_names : list[str]
        A list of fMRI ROIs in order in which they appear in dataset
    eeg_channel_names : list[str]
        A list of EEG channels in order in which they appear in dataset
    criterion_name : str
        A name of the required criterion
    criterion_kwargs : dict
        A dict with keyword arguments for a desired criterion
    optimizer_name : str
        A name of the optimizer
    optimizer_kwargs : dict
        A dictionary with keyword arguments for the Adam optimizer initialization
    scheduler_kwargs : dict
        A dictionary with keyword arguments for the ReduceLROnPlateau learning rate scheduler initialization
    dataframe_logging_modes : list[str]
        A list of modes during which a model will save all metrics instead of just averaged over batch. Options:
        'train', 'val', 'test'
    """
    def __init__(self,
                 nn_parameters: dict,
                 roi_names: list[str], eeg_channel_names: list[str],
                 criterion_name: str, criterion_kwargs: dict,
                 optimizer_name: str, optimizer_kwargs: dict,
                 scheduler_kwargs: dict,
                 dataframe_logging_modes: list[str]):
        super().__init__()

        self.model = BrainWaveLinker(
            in_channels=len(eeg_channel_names),
            out_channels=len(roi_names),
            **nn_parameters
        )
        self.criterion = build_criterion(criterion_name=criterion_name, criterion_kwargs=criterion_kwargs)
        self.eeg_channel_names = list(eeg_channel_names)
        self.roi_names = list(roi_names)
        self.criterion_name = criterion_name
        self.criterion_kwargs = criterion_kwargs
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_name = 'ReduceLROnPlateau'
        self.scheduler_kwargs = scheduler_kwargs
        self.dataframe_logging_modes = dataframe_logging_modes
        self.full_metrics = {}
        for dataframe_logging_mode in self.dataframe_logging_modes:
            assert dataframe_logging_mode in ['train', 'val', 'test'], dataframe_logging_mode
        for dataframe_logging_mode in ['train', 'val', 'test']:
            if dataframe_logging_mode in self.dataframe_logging_modes:
                self.full_metrics[dataframe_logging_mode] = {roi_name: [] for roi_name in self.roi_names}
            else:
                self.full_metrics[dataframe_logging_mode] = None

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

    def log_metrics_tables(self, mode: str):
        assert self.full_metrics[mode] is not None, (self.full_metrics.keys(), mode)
        self.logger.log_table(key=f'{mode}_full_correlations',
                              dataframe=pd.DataFrame(self.full_metrics[mode]))

    def common_step(self, batch: tuple[torch.Tensor], batch_idx: int, mode: str):
        """
        A common step for training, validation and testing iteration

        Parameters
        ----------
        batch : tuple[torch.Tensor]
            A tuple of EEG and fMRI tensors
        batch_idx : int
            An index of the current batch
        mode : str
            Current mode. Options: 'train', 'val', 'test'. Is used to correctly log metrics

        Returns
        -------
        loss : torch.Tensor
            Loss value of the criterion for the current batch
        batched_correlations : dict
            A dictionary with correlations for every batch element for every ROI
        average_correlations : dict
            A dictionary with correlations for every ROI averaged over a batch
        """
        inputs, target = batch
        output = self(inputs)
        loss = self.criterion(output, target)
        mean_corr, batched_correlations, average_correlations = self.calculate_metric_values(output=output,
                                                                                             target=target,
                                                                                             metrics_dict=
                                                                                             self.full_metrics[mode],
                                                                                             prefix=f'{mode}_')
        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{mode}_mean_corr', mean_corr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(average_correlations, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss, batched_correlations, average_correlations

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        loss, batched_correlations, average_correlations = self.common_step(batch=batch,
                                                                            batch_idx=batch_idx,
                                                                            mode='train')
        return loss

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        loss, batched_correlations, average_correlations = self.common_step(batch=batch,
                                                                            batch_idx=batch_idx,
                                                                            mode='val')
        return loss

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        loss, batched_correlations, average_correlations = self.common_step(batch=batch,
                                                                            batch_idx=batch_idx,
                                                                            mode='test')
        return loss

    def calculate_metric_values(self, output: torch.Tensor, target: torch.Tensor, metrics_dict: Optional[dict] = None,
                                prefix: str = ''):
        """
        Calculates and returns additional correlation metrics

        Parameters
        ----------
        output : torch.Tensor
            An output of the model, predicted BOLD
        target : torch.Tensor
            The ground truth BOLD sequence
        metrics_dict : dict or None
            A dictionary with lists of correlations per ROI for batch elements during previous iterations. New ones will
            be added to it inplace. Default: None
        prefix : str
            The prefix for the metrics naming. Note, add _ after the name into prefix for better readability.
            Recommended options: 'train_', 'val_', 'test_'. Default: ''

        Returns
        -------
        mean_corr : float
            Average over batch and ROIs correlation value
        batched_correlations : dict
            A dictionary with correlations for every batch element for every ROI
        average_correlations : dict
            A dictionary with correlations for every ROI averaged over a batch
        """
        assert output.size(1) == target.size(1) == len(self.roi_names), (output.size(), target.size(), self.roi_names)
        batched_correlations = {roi_name: [] for roi_name in self.roi_names}
        for b_idx in range(output.size(0)):
            for roi_idx, roi_name in enumerate(self.roi_names):
                corr = np.corrcoef(output[b_idx, roi_idx, :].clone().detach().cpu().numpy(),
                                   target[b_idx, roi_idx, :].clone().detach().cpu().numpy())[0, 1]
                batched_correlations[roi_name].append(corr)
                if metrics_dict is not None:
                    metrics_dict[roi_name].append(corr)
        average_correlations = {f'{prefix}mean_corr_{roi_name}': np.nanmean(roi_values) for roi_name, roi_values in
                                batched_correlations.items()}
        mean_corr = np.nanmean(list(average_correlations.values()))
        return mean_corr, batched_correlations, average_correlations

    def configure_optimizers(self):
        optimizers = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
            'SGD': torch.optim.SGD
        }
        optimizer = optimizers[self.optimizer_name](self.model.parameters(), **self.optimizer_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_kwargs),
                # ReduceLROnPlateau scheduler requires a monitor
                'monitor': 'val_loss',
                'frequency': 1
            },
        }
