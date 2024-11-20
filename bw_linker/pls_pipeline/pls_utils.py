import datetime
import os
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as sn


def get_results_folder(project_root: str, experiment_type: str, proj_name: str):
    """
    Get a folder for PLS experiment results

    Parameters
    ----------
    project_root : str
        A path to the root of the project
    experiment_type : str
        A type of the experiment (used to create subfolder with this name for results)
    proj_name : str
        A name of an experiment

    Returns
    -------
    save_root : str
        A path to a saving folder
    """
    now = str(datetime.datetime.now())
    date, time = now.split(' ')
    time = time.split('.')[0]
    time = '-'.join(time.split(':'))
    save_root_base = os.path.join(project_root, 'pls_logs', experiment_type, proj_name, date, time)
    save_root = save_root_base
    idx = 0
    while Path(save_root).exists():
        save_root = save_root_base + f'-v{idx}'
        idx += 1
    Path(save_root).mkdir(parents=True, exist_ok=False)
    return save_root


def bandpower(signal: np.ndarray, fs: int, band: tuple[int, int]):
    """
    Computes signal bandpower over a set band. Implementation imitates Matlab bandpower function as
    (Singer et al., 2023) were using a Matlab bandpower

    Parameters
    ----------
    signal : np.ndarray
        An EEG signal, shape (n_channels, n_times)
    fs : int
        Sampling rate
    band : tuple[int, int]
        A tuple or list of 2 ints with low and high frequency of the desired band

    Returns
    -------
    bandpow : float
        Bandpower value
    """
    n = signal.shape[-1]
    win = sn.windows.hamming(M=n, sym=True)
    f, Pxx = sn.periodogram(x=signal, fs=fs, window=win, nfft=n, axis=-1)
    freq_indices = np.where((f >= band[0]) & (f <= band[1]))[0]
    width = np.diff(f, axis=-1, append=0)
    pwr = width[freq_indices] @ Pxx[:, freq_indices].T
    return pwr


def flatten_eeg(eeg: np.ndarray, ch_axis: int = 2, eeg_ch_names: Optional[list[str]] = None):
    """
    Flattens EEG features and records correspondence of channels to indices

    Parameters
    ----------
    eeg : np.ndarray
        EEG features time series
    ch_axis : int
        Axis corresponding to channels
    eeg_ch_names : list[str] or None
        A list of names of EEG channels in order of appearance in eeg array. If None, creates '0', '1', '2', ... as
        names

    Returns
    -------
    eeg2 : np.ndarray
        Flattened EEG features array
    orig_indices : list
        A list where 1d index of a current feature corresponds to the 3d index of an original feature
    indices_per_channel : dict
        A dictionary with channel names as keys and list of indices of EEG features which refer to this channel as
        values
    """
    if eeg_ch_names is None:
        eeg_ch_names = [str(ch_idx) for ch_idx in range(eeg.shape[ch_axis])]

    # n_bands, n_shifts, n_channels, shifted_n_times
    eeg2 = eeg.reshape(eeg.shape[0] * eeg.shape[1] * eeg.shape[2], eeg.shape[3])
    assert eeg2.shape[-1] == eeg.shape[-1], (eeg2.shape, eeg.shape)
    orig_shape = eeg.shape[:-1]
    n_times = eeg.shape[-1]
    orig_indices = []
    indices_per_channel = {ch_idx: [] for ch_idx in eeg_ch_names}
    for idx in range(eeg2.shape[0]):
        orig_idx = np.unravel_index(idx, orig_shape)
        orig_indices.append(orig_idx)
        indices_per_channel[eeg_ch_names[orig_idx[ch_axis]]].append(idx)
        for t_idx in range(n_times):
            assert eeg2[idx, t_idx] == eeg[:, :, :, t_idx][orig_idx], (idx, orig_idx, t_idx, eeg2[idx, t_idx],
                                                                       eeg[:, :, :, t_idx][orig_idx])

    new_shape = (np.prod(eeg.shape[:-2]), eeg.shape[-1])
    for ch_idx, new_indices in indices_per_channel.items():
        assert np.equal(
            eeg2[new_indices, :], np.take(eeg, eeg_ch_names.index(ch_idx), axis=ch_axis).reshape(new_shape)
        ).all()

    return eeg2, orig_indices, indices_per_channel
