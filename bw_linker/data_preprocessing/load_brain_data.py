import os
from typing import Optional

import pandas as pd
import numpy as np
import mne
from scipy.interpolate import interp1d

from bw_linker.utils.constants import EEG_SAMPLING_RATE, FMRI_TR, RUNS, TEST_SIZES
from bw_linker.utils.helpers import is_integer_with_custom_precision


def load_fmri_data(root: str, sub: str, ses: str, task: str, desired_fmri_sampling_rate: int, interpolation_type: str,
                   rois: Optional[list[str]], rois_for_global_trend: Optional[list[str]],
                   roi_folder: str = 'roi'):
    """
    Loads and interpolates pre-processed fMRI data. The original fMRI is taken at 2.1 sec per scan.

    Parameters
    ----------
    root : str
        A directory where the dataset is
    sub : str
        A subject id
    ses : str
        A session id
    task : str
        A task id
    desired_fmri_sampling_rate : int
        Sampling rate fMRI should be upsampled to
    interpolation_type : str
        A type of interpolation to perform for fMRI upsampling
    rois : list[str] or None
        A list of fMRI ROIs to load from file. If None, extracts every ROI available
    rois_for_global_trend : list[str] or None
        A list of fMRI ROIs to load from file for Global Trend calculation. If None, uses rois
    roi_folder : str
        A folder with ROIs. 'roi' is a folder with subcortical ROIs, 'cortrois' is a folder with cortical ROIs.
        Default: 'roi'

    Returns
    -------
    df_fmri_interp : pd.DataFrame
        A DataFrame with interpolated fMRI data. Columns are the ROI names
    rois : list[str]
        A list of fMRI ROIs loaded from file
    csv_path : str
        A path to the fMRI data loaded
    fmri_required_times : np.ndarray
        Time-steps to which fMRI was upsampled
    fmri_times : np.ndarray
        Time-steps of original fMRI scans
    rois_for_global_trend : list[str]
        A list of fMRI ROIs loaded from file for Global Trend calculation
    """
    filename = f'sub-{sub}_ses-{ses}_task-{task}.csv'
    csv_path = os.path.join(root, f'sub-{sub}', f'ses-{ses}', roi_folder, filename)
    fmri_data = pd.read_csv(csv_path)

    fmri_times = fmri_data[' Tstart'].to_numpy()
    step = 1 / desired_fmri_sampling_rate
    fmri_required_times = np.arange(start=fmri_times[0], stop=fmri_times[-1], step=step)

    fmri_interpolate_func = interp1d(fmri_times,
                                     fmri_data.to_numpy(),
                                     kind=interpolation_type,
                                     axis=0)  # [time, n_regions]

    fmri_interpolate = fmri_interpolate_func(fmri_required_times)
    df_fmri_interp = pd.DataFrame(fmri_interpolate, columns=fmri_data.columns)

    if rois is None:
        rois = [column for column in df_fmri_interp.columns if column not in ['Nvol', ' Tstart']]

    if rois_for_global_trend is None:
        rois_for_global_trend = list(rois)

    all_rois = [roi for roi in list(rois)]
    all_rois = all_rois + [roi for roi in rois_for_global_trend if roi not in rois]

    df_fmri_interp = df_fmri_interp[all_rois]

    return df_fmri_interp, rois, csv_path, fmri_required_times, fmri_times, rois_for_global_trend


def load_eeg_data(root: str, sub: str, ses: str, task: str, eeg_channels: Optional[list[str]]):
    """
    Loads EEG data, checks it and crops by first fMRI scan

    Parameters
    ----------
    root : str
        A directory where the dataset is
    sub : str
        A subject id
    ses : str
        A session id
    task : str
        A task id
    eeg_channels : list[str] or None
        A list of EEG channels to load from file. If None, extracts every channel available

    Returns
    -------
    df_eeg : pd.DataFrame
        A DataFrame with EEG data. Columns are EEG channels
    eeg_channels : list[str]
        A list of EEG channels loaded from file
    vhdr_path : str
        A path to the .vhdr file from which the data was extracted
    eeg_times : np.ndarray
        Time-steps of the EEG signal
    """
    filename = f'sub-{sub}_ses-{ses}_task-{task}_eegMRbvCBbviiR250.vhdr'
    vhdr_path = os.path.join(root, f'sub-{sub}', f'ses-{ses}', 'eeg', filename)
    eeg_raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    eeg_raw = eeg_raw.pick('eeg')

    events, event_ids = mne.events_from_annotations(eeg_raw)

    r128_scans = events[events[:, -1] == event_ids['Stimulus/R128']]

    unique_scan_diffs, counts_diffs = np.unique(np.diff(r128_scans[:, 0]), return_counts=True)
    assert len(unique_scan_diffs) == 1, (unique_scan_diffs, counts_diffs)
    assert unique_scan_diffs[0] == 525  # 2.1 sec (fMRI TR) * 250 Hz (EEG sampling rate)

    assert eeg_raw.info['sfreq'] == EEG_SAMPLING_RATE
    first_r128 = r128_scans[0, 0] / EEG_SAMPLING_RATE

    cropped_eeg = eeg_raw.crop(tmin=first_r128)

    df_eeg = cropped_eeg.to_data_frame()

    if eeg_channels is None:
        eeg_channels = [column for column in df_eeg.columns if column != 'time']

    eeg_times = df_eeg['time']
    df_eeg = df_eeg[eeg_channels]

    return df_eeg, eeg_channels, vhdr_path, eeg_times


def load_data(root: str, sub: str, ses: str, task: str, desired_fmri_sampling_rate: int, fmri_interpolation_type: str,
              rois: Optional[list[str]], eeg_channels: Optional[list[str]], delay_eeg_samples: int,
              delay_fmri_samples: int, sampling_rates_ratio: int, separate_global_trend: bool = True,
              starting_point_sec: int = 0, rois_for_global_trend: Optional[list[str]] = None, roi_folder: str = 'roi'):
    """
    Loads, pre-processes, aligns and returns EEG and fMRI data for a specific subject, session and task

    Parameters
    ----------
    root : str
        A directory where the dataset is
    sub : str
        A subject id
    ses : str
        A session id
    task : str
        A task id
    desired_fmri_sampling_rate : int
        Sampling rate fMRI should be upsampled to
    fmri_interpolation_type : str
        A type of interpolation to perform for fMRI upsampling
    rois : list[str] or None
        A list of fMRI ROIs to load from file. If None, extracts every ROI available
    eeg_channels : list[str] or None
        A list of EEG channels to load from file. If None, extracts every channel available
    delay_eeg_samples : int
        A delay for shifting fMRI from EEG in samples in EEG sampling rate
    delay_fmri_samples : int
        A delay for shifting fMRI from EEG in samples in desired fMRI sampling rate
    sampling_rates_ratio : int
        A ratio of EEG sampling rate to the desired fMRI sampling rate
    separate_global_trend : bool
        If True will calculate and separate Global Trend from ROIs. Default: True
    starting_point_sec : int
        An amount of seconds to skip in the beginning of the record. Useful to avoid any issues with the beginning of
        files (e.g. device on but experiment did not start yet). Default: 0
    rois_for_global_trend : list[str] or None
        A list of fMRI ROIs to load from file for Global Trend calculation. If None, uses rois
    roi_folder : str
        A folder with ROIs. 'roi' is a folder with subcortical ROIs, 'cortrois' is a folder with cortical ROIs.
        Default: 'roi'

    Returns
    -------
    eeg_array : np.ndarray
        Pre-processed and aligned (including delay) EEG array
    fmri_array : np.ndarray
        Pre-processed and aligned (including delay) fMRI array
    eeg_channels : list[str]
        A list of EEG channels loaded from file
    rois : list[str]
        A list of fMRI ROIs loaded from file
    eeg_path : str
        A path to the .vhdr file from which the data was extracted
    fmri_path : str
        A path to the fMRI data loaded
    eeg_times : np.ndarray
        Time-steps of the EEG signal
    fmri_required_times : np.ndarray
        Time-steps to which fMRI was upsampled
    """
    df_fmri, rois, fmri_path, fmri_required_times, fmri_original_times, rois_for_global_trend = load_fmri_data(
        root=root, sub=sub, ses=ses, task=task, desired_fmri_sampling_rate=desired_fmri_sampling_rate,
        interpolation_type=fmri_interpolation_type, rois=rois, rois_for_global_trend=rois_for_global_trend,
        roi_folder=roi_folder
    )

    df_eeg, eeg_channels, eeg_path, eeg_times = load_eeg_data(
        root=root, sub=sub, ses=ses, task=task, eeg_channels=eeg_channels
    )

    fmri_in_eeg_sampling_rate = fmri_original_times.shape[0] * FMRI_TR * EEG_SAMPLING_RATE
    integer_check, fmri_in_eeg_sampling_rate = is_integer_with_custom_precision(
        num=fmri_in_eeg_sampling_rate, eps=1e-6, int_value=None
    )
    assert integer_check, (f'fMRI and EEG should already aligned after preprocessing. Only couple of '
                           f'milliseconds could be of a difference: {fmri_in_eeg_sampling_rate}, '
                           f'{fmri_original_times.shape}, {EEG_SAMPLING_RATE}, {df_eeg.shape}.\nRoot: {root}, '
                           f'sub: {sub}, ses: {ses}, task: {task}.')
    assert abs(df_eeg.shape[0] - fmri_in_eeg_sampling_rate) <= 25, (f'fMRI and EEG should already '
                                                                    f'aligned after preprocessing. Only '
                                                                    f'couple of milliseconds could be of a '
                                                                    f'difference: {fmri_in_eeg_sampling_rate},'
                                                                    f' {fmri_original_times.shape}, '
                                                                    f'{EEG_SAMPLING_RATE}, {df_eeg.shape}.'
                                                                    f'\nRoot: {root},  sub: {sub}, ses: {ses},'
                                                                    f' task: {task}.')

    assert (EEG_SAMPLING_RATE % desired_fmri_sampling_rate) == 0
    assert sampling_rates_ratio == (EEG_SAMPLING_RATE // desired_fmri_sampling_rate)
    n_eeg_samples = df_fmri.shape[0] * sampling_rates_ratio
    assert n_eeg_samples <= df_eeg.shape[0], (f'Required: {n_eeg_samples}, available: {df_eeg.shape}. '
                                              f'fMRI shape: {df_fmri.shape}, sampling rates ratio: '
                                              f'{sampling_rates_ratio}')
    df_eeg = df_eeg.iloc[:n_eeg_samples]

    assert df_eeg.shape[0] == (df_fmri.shape[0] * sampling_rates_ratio), (f'EEG size: {df_eeg.shape}, fMRI '
                                                                          f'size: {df_fmri.shape}, sr ratio: '
                                                                          f'{sampling_rates_ratio}.\nRoot: {root},  '
                                                                          f'sub: {sub}, ses: {ses}, task: {task}.')

    assert delay_eeg_samples >= 0, f'Delay should be non-negative, got: delay_eeg_samples=={delay_eeg_samples}'
    assert delay_fmri_samples >= 0, f'Delay should be non-negative, got: delay_fmri_samples=={delay_fmri_samples}'
    if delay_eeg_samples > 0:
        df_eeg = df_eeg.iloc[:-delay_eeg_samples]
        df_fmri = df_fmri.iloc[delay_fmri_samples:]

    assert df_eeg.shape[0] == (df_fmri.shape[0] * sampling_rates_ratio), (f'EEG size: {df_eeg.shape}, fMRI '
                                                                          f'size: {df_fmri.shape}, sr ratio: '
                                                                          f'{sampling_rates_ratio}.\nRoot: {root},  '
                                                                          f'sub: {sub}, ses: {ses}, task: {task}.')

    eeg_array = df_eeg.to_numpy().T
    fmri_array = df_fmri[rois].to_numpy().T
    fmri_for_global_trend = df_fmri[rois_for_global_trend].to_numpy().T

    # cut some initial data with potential issues
    eeg_array = eeg_array[:, starting_point_sec * EEG_SAMPLING_RATE:]
    fmri_array = fmri_array[:, starting_point_sec * desired_fmri_sampling_rate:]
    fmri_for_global_trend = fmri_for_global_trend[:, starting_point_sec * desired_fmri_sampling_rate:]

    # center BOLD around 0
    fmri_array = fmri_array - np.mean(fmri_array, axis=1, keepdims=True)
    fmri_for_global_trend = fmri_for_global_trend - np.mean(fmri_for_global_trend, axis=1, keepdims=True)

    if separate_global_trend:
        assert fmri_array.ndim == 2, fmri_array.shape
        assert fmri_for_global_trend.ndim == 2, fmri_for_global_trend.shape
        global_trend_array = np.mean(fmri_for_global_trend, axis=0)

        inner = np.inner(global_trend_array, global_trend_array)
        outer = np.outer(global_trend_array, global_trend_array)

        detrend = fmri_array @ outer / inner
        fmri_array = fmri_array - detrend

        for idx in range(fmri_array.shape[0]):
            corr_coeff = np.corrcoef(fmri_array[idx, :], global_trend_array)[0, 1]
            assert abs(corr_coeff) < 1e-6, (
                f'Sub: {sub}, ses: {ses}, task: {task}, ROI: {rois[idx]} has correlation above threshold: {corr_coeff}'
            )

        fmri_array = np.concatenate((fmri_array, global_trend_array[None, :]), axis=0)
        assert np.equal(fmri_array[-1], global_trend_array).all()

    return eeg_array, fmri_array, eeg_channels, rois, eeg_path, fmri_path, eeg_times, fmri_required_times


def load_multiple_files(root: str, subjects: list[str], desired_fmri_sampling_rate: int, fmri_interpolation_type: str,
                        rois: Optional[list[str]], eeg_channels: Optional[list[str]], delay_sec: int,
                        separate_global_trend: bool = True, starting_point_sec: int = 0,
                        rois_for_global_trend: Optional[list[str]] = None, roi_folder: str = 'roi'):
    """
    Loads, pre-processes, aligns and returns EEG and fMRI data for multiple files, runs

    Parameters
    ----------
    root : str
        A directory where the dataset is
    subjects : list[str]
        A list of subjects to process
    desired_fmri_sampling_rate : int
        Sampling rate fMRI should be upsampled to
    fmri_interpolation_type : str
        A type of interpolation to perform for fMRI upsampling
    rois : list[str] or None
        A list of fMRI ROIs to load from file. If None, extracts every ROI available
    eeg_channels : list[str] or None
        A list of EEG channels to load from file. If None, extracts every channel available
    delay_sec : int
        A delay for shifting fMRI from EEG in samples in seconds
    separate_global_trend : bool
        If True will calculate and separate Global Trend from ROIs. Default: True
    starting_point_sec : int
        An amount of seconds to skip in the beginning of the record. Useful to avoid any issues with the beginning of
        files (e.g. device on but experiment did not start yet). Default: 0
    rois_for_global_trend : list[str] or None
        A list of fMRI ROIs to load from file for Global Trend calculation. If None, uses rois
    roi_folder : str
        A folder with ROIs. 'roi' is a folder with subcortical ROIs, 'cortrois' is a folder with cortical ROIs.
        Default: 'roi'

    Returns
    -------
    datasets : dict
        A dictionary with loaded data. Has the following structure:
        (path_to_eeg_file, path_to_fmri_file):
            eeg: np.ndarray with EEG data
            fmri: np.ndarray with fMRI data
            task: task name
            eeg_times: time-steps of the EEG signal
            fmri_required_times: time-steps to which fMRI was upsampled
            delay_fmri_samples: a delay for shifting fMRI from EEG in samples in desired fMRI sampling rate
    rois : list[str]
        A list of fMRI ROIs loaded from file
    eeg_channels : list[str]
        A list of EEG channels loaded from file
    sampling_rates_ratio : int
        A ratio of EEG sampling rate to the desired fMRI sampling rate
    """
    assert (EEG_SAMPLING_RATE % desired_fmri_sampling_rate) == 0
    sampling_rates_ratio = (EEG_SAMPLING_RATE // desired_fmri_sampling_rate)
    delay_eeg_samples = delay_sec * EEG_SAMPLING_RATE
    delay_fmri_samples = delay_sec * desired_fmri_sampling_rate

    datasets = {}
    if rois is None:
        overall_rois = None
    else:
        overall_rois = list(rois)

    if eeg_channels is None:
        overall_eeg_channels = None
    else:
        overall_eeg_channels = list(eeg_channels)

    for subject_id in subjects:
        for ses, task in RUNS:
            eeg_array, fmri_array, eeg_channels, rois, eeg_path, fmri_path, eeg_times, fmri_required_times = load_data(
                root=root, sub=subject_id, ses=ses, task=task,
                desired_fmri_sampling_rate=desired_fmri_sampling_rate,
                fmri_interpolation_type=fmri_interpolation_type, rois=rois,
                eeg_channels=eeg_channels, delay_eeg_samples=delay_eeg_samples,
                delay_fmri_samples=delay_fmri_samples, sampling_rates_ratio=sampling_rates_ratio,
                separate_global_trend=separate_global_trend,
                starting_point_sec=starting_point_sec, rois_for_global_trend=rois_for_global_trend,
                roi_folder=roi_folder
            )

            if overall_rois is None:
                overall_rois = list(rois)
            else:
                assert overall_rois == rois

            if overall_eeg_channels is None:
                overall_eeg_channels = list(eeg_channels)
            else:
                assert overall_eeg_channels == eeg_channels

            datasets[(eeg_path, fmri_path)] = {
                'eeg': eeg_array, 'fmri': fmri_array, 'task': task.split('_')[0], 'eeg_times': eeg_times,
                'fmri_required_times': fmri_required_times, 'delay_fmri_samples': delay_fmri_samples
            }

    rois = list(rois)
    if separate_global_trend:
        rois.append(' Global Regressive Trend')

    return datasets, rois, eeg_channels, sampling_rates_ratio


def split_single_dataset(eeg: np.ndarray, fmri: np.ndarray, sampling_rates_ratio: int, task: str):
    """
    Returns data split into train, validation and test sections contiguously. The function takes test size from the
    latest time-steps. Then, takes validation from time-steps before test. Then, uses the rest as train. We measure
    train/validation/test size in fMRI sampling rate and then scale by sampling_rates_ratio to stay in the realm of
    integers

    Parameters
    ----------
    eeg : np.ndarray
        Pre-processed and aligned (including delay) EEG array
    fmri : np.ndarray
        Pre-processed and aligned (including delay) fMRI array
    sampling_rates_ratio : int
        A ratio of EEG sampling rate to the desired fMRI sampling rate
    task : str
        A task id

    Returns
    -------
    train_data : dict
        A dictionary of train data with keys: 'eeg' and 'fmri'
    val_data : dict
        A dictionary of validation data with keys: 'eeg' and 'fmri'
    test_data : dict
        A dictionary of test data with keys: 'eeg' and 'fmri'
    """
    fmri_size = fmri.shape[1]
    fmri_test_size = TEST_SIZES[task]
    fmri_valid_size = TEST_SIZES[task]
    fmri_train_size = fmri_size - fmri_valid_size - fmri_test_size
    fmri_train_val_size = fmri_train_size + fmri_valid_size

    assert min(fmri_train_size, fmri_valid_size, fmri_test_size) >= 0, (f'One of {fmri_train_size}, {fmri_valid_size}, '
                                                                        f'{fmri_test_size} is negative')

    train_data = {'eeg': eeg[:, :fmri_train_size * sampling_rates_ratio],
                  'fmri': fmri[:, :fmri_train_size]}

    val_data = {'eeg': eeg[:, fmri_train_size * sampling_rates_ratio:fmri_train_val_size * sampling_rates_ratio],
                'fmri': fmri[:, fmri_train_size:fmri_train_val_size]}

    test_data = {'eeg': eeg[:, fmri_train_val_size * sampling_rates_ratio:],
                 'fmri': fmri[:, fmri_train_val_size:]}

    return train_data, val_data, test_data


def split_datasets(datasets: dict, sampling_rates_ratio: int):
    """
    Splits all datasets into train/validation/test parts. Splits according to time steps: latest ones go to test,
    intermediate into validation and early ones into train

    Parameters
    ----------
    datasets : dict
        A dictionary with loaded data. Has the following structure of required keys:
        (path_to_eeg_file, path_to_fmri_file):
            eeg: np.ndarray with EEG data
            fmri: np.ndarray with fMRI data
            task : str with task id
    sampling_rates_ratio : int
        A ratio of EEG sampling rate to the desired fMRI sampling rate

    Returns
    -------
    train_datasets : dict
        A dictionary with train data of the same structure as datasets
    validation_datasets : dict
        A dictionary with validation data of the same structure as datasets
    test_datasets : dict
        A dictionary with test data of the same structure as datasets
    """
    assert isinstance(sampling_rates_ratio, int), f'Sampling rates ratio {sampling_rates_ratio} should be an integer'
    train_datasets = {}
    validation_datasets = {}
    test_datasets = {}

    for set_path, single_dataset in datasets.items():

        train_data, val_data, test_data = split_single_dataset(
            eeg=single_dataset['eeg'], fmri=single_dataset['fmri'], sampling_rates_ratio=sampling_rates_ratio,
            task=single_dataset['task']
        )

        train_datasets[set_path] = train_data

        if val_data is not None:
            validation_datasets[set_path] = val_data

        if test_data is not None:
            test_datasets[set_path] = test_data

    return train_datasets, validation_datasets, test_datasets
