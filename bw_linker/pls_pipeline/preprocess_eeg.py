from typing import Optional

import numpy as np
from tqdm import tqdm

from bw_linker.pls_pipeline.pls_utils import bandpower, flatten_eeg
from bw_linker.utils.constants import EEG_SAMPLING_RATE
from bw_linker.utils.helpers import normalize_time_series


def preprocess_eeg_for_pls(
        eeg: np.ndarray, fmri: np.ndarray, fmri_sampling_rate: int, eeg_ch_names: Optional[list[str]] = None,
        window_size_sec: float = 1, overlap_sec: float = 0.5, n_shifts: int = 60, shift_step_sec: float = 0.5,
        bands: tuple[tuple[int, int]] = (
                (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )
):
    """
    Does EEG preprocessing for multiple dataset parts. Preprocessing according to the algorithm described in
    Singer N, Poker G, Dunsky-Moran N, Nemni S, Reznik Balter S, Doron M, Baker T, Dagher A, Zatorre RJ, Hendler T.
    Development and validation of an fMRI-informed EEG model of reward-related ventral striatum activation.
    Neuroimage. 2023 Aug 1;276:120183. doi: 10.1016/j.neuroimage.2023.120183.

    Parameters
    ----------
    eeg : np.ndarray
        An array with EEG data, shape (n_channels, n_times)
    fmri : np.ndarray
        An array with fMRI data, shape (n_rois, n_times_fmri)
    fmri_sampling_rate : int
        fMRI sampling rate
    eeg_ch_names : list[str] or None
        A list of names of EEG channels in order of appearance in eeg array. If None, creates '0', '1', '2', ... as
        names. Default: None
    window_size_sec : float
        Window size for bandpower calculation in seconds. Default: 1
    overlap_sec : float
        Overlap in seconds between window. Default: 0.5
    n_shifts : int
        Amount of lag shifts to use for prediction. Default: 60
    shift_step_sec : float
        Time difference in seconds for every shift backwards. Default: 0.5
    bands : tuple[tuple[int, int]]
        A sequence of bands to use for bandpower calculations. Default: (
                (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )

    Returns
    -------
    eeg : np.ndarray
        An array of EEG features of shape (n_bands, n_shifts, n_channels, n_times)
    fmri : np.ndarray
        An array of respective fMRI
    flattened_eeg : np.ndarray
        An array of EEG features flattened over features axis; shape (n_bands * n_shifts * n_channels, n_times)
    orig_indices : list
        A list where 1d index of a current feature corresponds to the 3d index of an original feature
    indices_per_channel : dict
        A dictionary with channel names as keys and list of indices of EEG features which refer to this channel as
        values
    """
    assert eeg.ndim == 2, eeg.shape
    assert fmri.ndim == 2, fmri.shape
    assert fmri_sampling_rate == 2

    assert isinstance(EEG_SAMPLING_RATE, int), EEG_SAMPLING_RATE
    assert isinstance(fmri_sampling_rate, int), fmri_sampling_rate
    window_size = int(window_size_sec * EEG_SAMPLING_RATE)  # 1 sec
    overlap = int(overlap_sec * EEG_SAMPLING_RATE)  # 0.5 sec
    step = window_size - overlap
    new_sampling_rate = EEG_SAMPLING_RATE // step
    assert new_sampling_rate == 2, (new_sampling_rate, EEG_SAMPLING_RATE, step, overlap, window_size)
    shift_step = int(shift_step_sec * new_sampling_rate)
    fmri_shift_step = int(shift_step_sec * fmri_sampling_rate)
    max_shift = (n_shifts - 1) * shift_step
    fmri_shift = (n_shifts - 1) * fmri_shift_step
    assert max_shift == fmri_shift, (max_shift, fmri_shift)
    n_bands = len(bands)

    n_channels = eeg.shape[0]
    n_times = eeg.shape[1] // step
    shifted_n_times = n_times - max_shift

    preprocessed_eeg = np.full((n_bands, n_channels, n_times), fill_value=np.inf)

    for win_idx in range(n_times):
        start = win_idx * step
        window = eeg[:, start:start+window_size]

        for band_idx, band in enumerate(bands):
            preprocessed_eeg[band_idx, :, win_idx] = bandpower(signal=window, fs=EEG_SAMPLING_RATE, band=band)

    assert not np.isinf(preprocessed_eeg).any(), (preprocessed_eeg.shape, np.sum(np.isinf(preprocessed_eeg)))

    shifted_data = []

    for shift_idx in range(n_shifts):
        start = shift_idx * shift_step
        finish = n_times - max_shift + start
        shifted_data.append(preprocessed_eeg[:, :, start:finish])

    shifted_data = np.stack(shifted_data, axis=1)  # (n_bands, n_shifts, n_channels, shifted_n_times)
    assert shifted_data.shape == (n_bands, n_shifts, n_channels, shifted_n_times), (
        shifted_data.shape, (n_bands, n_shifts, n_channels, shifted_n_times)
    )

    shifted_data = normalize_time_series(series=shifted_data, axis=-1)

    fmri = fmri[:, fmri_shift:]
    fmri = normalize_time_series(series=fmri, axis=-1)

    flattened_eeg, orig_indices, indices_per_channel = flatten_eeg(eeg=shifted_data,
                                                                   eeg_ch_names=eeg_ch_names)

    return shifted_data, fmri, flattened_eeg, orig_indices, indices_per_channel


def preprocess_multiple_datasets(
        datasets: dict, fmri_sampling_rate: int, eeg_ch_names: Optional[list[str]], roi: str,
        separate_global_trend: bool,
        orig_indices: Optional[list] = None, indices_per_channel: Optional[dict] = None, disable_tqdm: bool = False,
        window_size_sec: float = 1, overlap_sec: float = 0.5, n_shifts: int = 60, shift_step_sec: float = 0.5,
        bands: tuple[tuple[int, int]] = (
                (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )
):
    """
    Does EEG preprocessing for multiple dataset parts. Preprocessing according to the algorithm described in
    Singer N, Poker G, Dunsky-Moran N, Nemni S, Reznik Balter S, Doron M, Baker T, Dagher A, Zatorre RJ, Hendler T.
    Development and validation of an fMRI-informed EEG model of reward-related ventral striatum activation.
    Neuroimage. 2023 Aug 1;276:120183. doi: 10.1016/j.neuroimage.2023.120183.

    Parameters
    ----------
    datasets : dict
        A dictionary with data. Has the following structure:
        (path_to_eeg_file, path_to_fmri_file):
            eeg: np.ndarray with EEG data
            fmri: np.ndarray with fMRI data
    fmri_sampling_rate : int
        fMRI sampling rate
    eeg_ch_names : list[str] or None
        A list of names of EEG channels in order of appearance in eeg array. If None, creates '0', '1', '2', ... as
        names
    roi : str
        Desired ROI to predict
    separate_global_trend : bool
        Whether a global trend should be separated
    orig_indices : list or None
        A list where 1d index of a current feature corresponds to the 3d index of an original feature
    indices_per_channel : dict or None
        A dictionary with channel names as keys and list of indices of EEG features which refer to this channel as
        values
    disable_tqdm : bool
        If True, does not print tqdm progress bar. Default: False
    window_size_sec : float
        Window size for bandpower calculation in seconds. Default: 1
    overlap_sec : float
        Overlap in seconds between window. Default: 0.5
    n_shifts : int
        Amount of lag shifts to use for prediction. Default: 60
    shift_step_sec : float
        Time difference in seconds for every shift backwards. Default: 0.5
    bands : tuple[tuple[int, int]]
        A sequence of bands to use for bandpower calculations. Default: (
                (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )

    Returns
    -------
    all_eeg : np.ndarray
        A concatenated array of EEG features of shape (n_bands, n_shifts, n_channels, sum_i(run_i_times))
    all_fmri : np.ndarray
        A concatenated array of respective fMRI
    run_labels : np.ndarray
        A concatenated array of run labels (each run has a unique integer label from 0 to n_runs - 1)
    all_flattened_eeg : np.ndarray
        A concatenated array of EEG features flattened over features axis; shape
        (n_bands * n_shifts * n_channels, sum_i(run_i_times))
    orig_indices : list
        A list where 1d index of a current feature corresponds to the 3d index of an original feature
    indices_per_channel : dict
        A dictionary with channel names as keys and list of indices of EEG features which refer to this channel as
        values
    """
    all_eeg = []
    all_flattened_eeg = []
    all_fmri = []
    run_labels = []
    for ds_idx, (ds_name, ds) in tqdm(enumerate(datasets.items()), total=len(datasets), disable=disable_tqdm):
        if separate_global_trend and (roi == ' Global Trend'):
            ds_fmri = ds['fmri'][-1:, :]  # Global Trend is the final ROI
        elif separate_global_trend:
            assert ds['fmri'].shape[0] == 2, ds['fmri'].shape  # First element - roi, second - global trend
            ds_fmri = ds['fmri'][:1, :]
        else:
            assert ds['fmri'].shape[0] == 1, ds['fmri'].shape
            ds_fmri = ds['fmri']
        eeg_run, fmri_run, flattened_eeg_run, orig_indices_run, indices_per_channel_run = preprocess_eeg_for_pls(
            eeg=ds['eeg'], fmri=ds_fmri, fmri_sampling_rate=fmri_sampling_rate, eeg_ch_names=eeg_ch_names,
            window_size_sec=window_size_sec, overlap_sec=overlap_sec, n_shifts=n_shifts, shift_step_sec=shift_step_sec,
            bands=bands
        )
        if orig_indices is None:
            orig_indices = orig_indices_run
        else:
            assert orig_indices == orig_indices_run, (ds_name, orig_indices, orig_indices_run)
        if indices_per_channel is None:
            indices_per_channel = indices_per_channel_run
        else:
            assert indices_per_channel == indices_per_channel_run, (ds_name, indices_per_channel,
                                                                    indices_per_channel_run)
        all_eeg.append(eeg_run)
        all_fmri.append(fmri_run)
        all_flattened_eeg.append(flattened_eeg_run)
        assert eeg_run.shape[-1] == fmri_run.shape[-1] == flattened_eeg_run.shape[-1], (
            ds_name, eeg_run.shape, fmri_run.shape, flattened_eeg_run.shape
        )
        run_labels = run_labels + (flattened_eeg_run.shape[-1] * [ds_idx])

    all_eeg = np.concatenate(all_eeg, axis=-1)
    all_flattened_eeg = np.concatenate(all_flattened_eeg, axis=-1)
    all_fmri = np.concatenate(all_fmri, axis=-1)
    run_labels = np.array(run_labels)

    assert all_eeg.shape[-1] == all_fmri.shape[-1] == run_labels.shape[0] == all_flattened_eeg.shape[-1], (
        all_eeg.shape, all_fmri.shape, run_labels.shape, all_flattened_eeg.shape
    )

    return all_eeg, all_fmri, run_labels, all_flattened_eeg, orig_indices, indices_per_channel
