import math
from typing import Optional

import numpy as np
import torch
from scipy.stats import iqr
from torch.utils.data import ConcatDataset, Dataset


class EEGfMRIDataset(Dataset):
    """
    Initializes a PyTorch EEG-to-fMRI dataset which splits eeg and fmri arrays into segments

    Parameters
    ----------
    eeg : np.array
        A numpy array with EEG signal. Shape: (n_channels, n_times)
    fmri : np.array
        A numpy array with fMRI (BOLD) signal. Shape: (n_rois, n_times)
    fmri_window_size : int
        A length of a single fMRI segment in samples. If both fmri_window_size and eeg_window_size are None,
        they are automatically chosen to include the whole EEG and fMRI as a single item
    eeg_window_size : int
        A length of a single EEG segment in samples. If both fmri_window_size and eeg_window_size are None,
        they are automatically chosen to include the whole EEG and fMRI as a single item
    sampling_rates_ratio : int
        A ratio of EEG sampling rate to the fMRI sampling rate
    eeg_channels : list[str]
        A list of EEG channels in order in which they appear in eeg array. Default: None
    fmri_rois : list[str]
        A list of fMRI ROIs in order in which they appear in fmri array. Must be provided if global trend in
        extracted. Default: None
    ds_path : str or None
        A path to the dataset file from which eeg and fmri were extracted. Default: None
    stride : int
        A stride to use while splitting eeg and fmri into segments. Default: 1
    eeg_standardization_kwargs : dict or None
        A dictionary with arguments for EEG standardization.
        Required keys:
        'subtract'. Options: 'mean', 'median', None. If None, subtracts 0
        'divide_by'. Options: 'std', 'iqr', None. If None, divides by 1
        'axis'. Options: 0, 1, None. Over which axis to subtract and divide.
        If None, no standardization is performed. Default: None
    fmri_standardization_kwargs : dict or None
        A dictionary with arguments for fMRI standardization.
        Required keys:
        'subtract'. Options: 'mean', 'median', None. If None, subtracts 0
        'divide_by'. Options: 'std', 'iqr', None. If None, divides by 1
        'axis'. Options: 0, 1, None. Over which axis to subtract and divide.
        If None, no standardization is performed. Default: None
    """
    def __init__(self, eeg: np.array, fmri: np.array, fmri_window_size: int, eeg_window_size: int,
                 sampling_rates_ratio: int,
                 eeg_channels: list[str], fmri_rois: list[str],
                 ds_path: Optional[str] = None, stride: int = 1,
                 eeg_standardization_kwargs: Optional[dict] = None,
                 fmri_standardization_kwargs: Optional[dict] = None):

        if eeg_standardization_kwargs is not None:
            eeg = self.standardize_data(data=eeg, **eeg_standardization_kwargs)
        if fmri_standardization_kwargs is not None:
            fmri = self.standardize_data(data=fmri, **fmri_standardization_kwargs)

        self.eeg = torch.tensor(eeg.astype(np.float32))
        self.fmri = torch.tensor(fmri.astype(np.float32))

        self.sampling_rates_ratio = sampling_rates_ratio
        self.eeg_channels = eeg_channels
        self.fmri_rois = list(fmri_rois)
        self.ds_path = ds_path
        self.eeg_window_size = eeg_window_size
        self.fmri_window_size = fmri_window_size
        self.stride = stride

    def __getitem__(self, item: int):
        item *= self.stride
        fmri_data = self.fmri[:, item:item + self.fmri_window_size]
        eeg_item = item * self.sampling_rates_ratio
        eeg_data = self.eeg[:, eeg_item:eeg_item + self.eeg_window_size]
        assert fmri_data.size(-1) == self.fmri_window_size
        assert eeg_data.size(-1) == self.eeg_window_size
        return eeg_data, fmri_data

    def __len__(self):
        return min(
            math.ceil((self.fmri.size(-1) - self.fmri_window_size + 1) / self.stride),
            math.ceil((self.eeg.size(-1) - self.eeg_window_size + 1) / (self.sampling_rates_ratio * self.stride))
        )

    def get_all_data(self):
        return self.eeg, self.fmri, self.ds_path

    def get_rois(self):
        return list(self.fmri_rois)

    @staticmethod
    def standardize_data(data: np.array, subtract: Optional[str] = 'mean', divide_by: Optional[str] = 'std',
                         axis: Optional[int] = 0):
        """
        Standardizes data with custom subtraction and division

        Parameters
        ----------
        data : np.array
            An array which should be standardized. Shape: (n_channels, n_times)
        subtract : str or None
            What to subtract from data. Options: 'mean', 'median', None. If None, subtracts 0. Default: 'mean'
        divide_by : str or None
            What to divide data by. Options: 'std', 'iqr', None. If None, divides by 1. Default: 'std'
        axis : int or None
            Over which axis to subtract and divide. axis=0 is channel-wise, axis=1 is time-wise,
            axis=None is global (over both axes). Default: 0

        Returns
        -------
        standardized_data : np.array
            A standardized data array
        """
        if (subtract is None) and (divide_by is None):
            return data
        functions = {'mean': np.mean, 'median': np.median, 'std': np.std, 'iqr': iqr}
        if subtract is None:
            subtraction_value = 0
        else:
            subtraction_value = functions[subtract](data, axis=axis, keepdims=True)
        if divide_by is None:
            division_value = 1
        else:
            division_value = functions[divide_by](data, axis=axis, keepdims=True)
        return (data - subtraction_value) / division_value


def build_full_dataset(datasets: dict, fmri_window_size: int, eeg_window_size: int, sampling_rates_ratio: int,
                       eeg_channels: list[str], fmri_rois: list[str], stride: int = 1,
                       eeg_standardization_kwargs: Optional[dict] = None,
                       fmri_standardization_kwargs: Optional[dict] = None):
    """
    Builds and returns a PyTorch dataset concatenated from datasets built from task-specific arrays

    Parameters
    ----------
    datasets : dict
        A dictionary with EEG and fMRI data per task. Structure: keys - paths to dataset file (data for a specific
        task), values - dictionaries with keys 'eeg' and 'fmri' which contain np.arrays with EEG and fMRI data for
        this task
    fmri_window_size : int
        A length of a single fMRI segment in samples. If both fmri_window_size and eeg_window_size are None,
        they are automatically chosen to include the whole EEG and fMRI as a single item
    eeg_window_size : int
        A length of a single EEG segment in samples. If both fmri_window_size and eeg_window_size are None,
        they are automatically chosen to include the whole EEG and fMRI as a single item
    sampling_rates_ratio : int
        A ratio of EEG sampling rate to the fMRI sampling rate
    eeg_channels : list[str]
        A list of EEG channels in order in which they appear in eeg array. Default: None
    fmri_rois : list[str]
        A list of fMRI ROIs in order in which they appear in fmri array. Must be provided if global trend in
        extracted. Default: None
    stride : int
        A stride to use while splitting eeg and fmri into segments. Default: 1
    eeg_standardization_kwargs : dict or None
        A dictionary with arguments for EEG standardization.
        Required keys:
        'subtract'. Options: 'mean', 'median', None. If None, subtracts 0
        'divide_by'. Options: 'std', 'iqr', None. If None, divides by 1
        'axis'. Options: 0, 1, None. Over which axis to subtract and divide.
        If None, no standardization is performed. Default: None
    fmri_standardization_kwargs : dict or None
        A dictionary with arguments for fMRI standardization.
        Required keys:
        'subtract'. Options: 'mean', 'median', None. If None, subtracts 0
        'divide_by'. Options: 'std', 'iqr', None. If None, divides by 1
        'axis'. Options: 0, 1, None. Over which axis to subtract and divide.
        If None, no standardization is performed. Default: None

    Returns
    -------
    full_dataset: torch.utils.data.ConcatDataset
        A PyTorch ConcatDataset which unites all task-specific data passed in datasets
    all_rois : list[str]
        A list of fMRI ROIs in order in which they appear in fmri array
    """
    torch_datasets = []
    all_rois = None
    for ds_path, dataset in datasets.items():
        torch_dataset = EEGfMRIDataset(
            eeg=dataset['eeg'], fmri=dataset['fmri'],
            fmri_window_size=fmri_window_size, eeg_window_size=eeg_window_size,
            sampling_rates_ratio=sampling_rates_ratio,
            eeg_channels=eeg_channels, fmri_rois=fmri_rois,
            ds_path=ds_path, stride=stride,
            eeg_standardization_kwargs=eeg_standardization_kwargs,
            fmri_standardization_kwargs=fmri_standardization_kwargs
        )
        torch_rois = torch_dataset.get_rois()
        if all_rois is None:
            all_rois = list(torch_rois)
        else:
            assert all_rois == torch_rois, (all_rois, torch_rois)
        torch_datasets.append(torch_dataset)
    full_dataset = ConcatDataset(torch_datasets)

    return full_dataset, all_rois
