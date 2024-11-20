import argparse
import itertools
import os

import numpy as np
import pandas as pd
import scipy.signal as sn
from joblib import delayed, Parallel

from bw_linker.data_preprocessing.load_brain_data import split_datasets
from bw_linker.pls_pipeline.pls_pipeline import run_pls_pipeline
from bw_linker.utils.constants import EEG_SAMPLING_RATE, PROJECT_ROOT


def generate_filtered_noise(n_samples: int, important_channels: list[int], n_ch: int,
                            bands: tuple[tuple[int, int]]):
    """
    Generate a synthetic signal with signal in specific frequency bands

    Parameters
    ----------
    n_samples : int
        Amount of samples to generate
    important_channels : list[int]
        A list with indices of main channels
    n_ch : int
        Amount of EEG channels to generate
    bands : tuple[tuple[int, int]]
        Frequency bands with signal

    Returns
    -------
    eeg : np.ndarray
        A generated synthetic EEG array
    fmri : np.ndarray
        A generated synthetic fMRI array
    """
    n_bands = len(bands)
    n_samples += EEG_SAMPLING_RATE  # accommodating for removal after filtering

    low_b, low_a = sn.butter(3, Wn=3, btype='low', fs=EEG_SAMPLING_RATE, output='ba')

    eeg = np.random.randn(n_ch, n_samples)
    band_power_data = np.full((n_ch * n_bands, n_samples), fill_value=np.inf)

    for band_idx, band in enumerate(bands):
        b, a = sn.butter(3, Wn=band, btype='bandpass', fs=EEG_SAMPLING_RATE, output='ba')
        filtered_data = sn.filtfilt(b, a, eeg, axis=-1)
        filtered_data = np.abs(sn.hilbert(filtered_data, axis=-1))
        filtered_data = sn.filtfilt(low_b, low_a, filtered_data, axis=-1)
        band_power_data[band_idx * n_ch:(band_idx + 1) * n_ch] = filtered_data

    assert not np.isinf(band_power_data).any(), np.sum(np.isinf(band_power_data))

    W = np.zeros(n_ch * n_bands)
    for ch in important_channels:
        W[np.arange(0, n_bands) * n_ch + ch] = np.random.randn(n_bands)

    fmri = W @ band_power_data

    # remove edge effects

    fmri = fmri[EEG_SAMPLING_RATE:]
    eeg = eeg[:, EEG_SAMPLING_RATE:]

    return eeg, fmri


def add_eeg_noise(eeg: np.ndarray, snr: float, print_magnitudes: bool):
    """
    Adds a noise to the EEG to get desired SNR

    Parameters
    ----------
    eeg : np.ndarray
        A generated synthetic EEG array
    snr : float
        A desired SNR
    print_magnitudes : bool
        If True will print information about SNR and norms of signal and noise

    Returns
    -------
    eeg : np.ndarray
        A generated synthetic EEG array with noise added and desired SNR
    """
    noise = np.random.randn(*eeg.shape)
    orig_norm = np.linalg.norm(eeg)
    noise_norm = np.linalg.norm(noise)
    # snr = orig_norm / noise_norm
    # therefore to make desired snr, noise norm should be orig_norm / snr
    desired_noise_norm = orig_norm / snr
    noise_coeff = desired_noise_norm / noise_norm
    noise *= noise_coeff
    new_noise_norm = np.linalg.norm(noise)

    eeg = eeg + noise

    new_norm = np.linalg.norm(eeg)

    if print_magnitudes:
        print(
            f'Signal to noise ratio: {snr}, '
            f'noiseless EEG norm: {orig_norm}, '
            f'noise norm: {new_noise_norm}, '
            f'EEG with noise norm: {new_norm}'
        )

    return eeg


def generate_signals(n_ch: int, main_channels: list[int], length: float, fmri_fs: int, snr: float,
                     bands: tuple[tuple[int, int]], print_magnitudes: bool = False):
    """
    Generate a synthetic signal with additive noise and desired SNR

    Parameters
    ----------
    n_ch : int
        Amount of EEG channels to generate
    main_channels : list[int]
        A list with indices of main channels
    length : float
        Length of a signal in seconds to generate
    fmri_fs : int
        Sampling rate of synthetic fMRI
    snr : float
        A desired SNR
    bands : tuple[tuple[int, int]]
        Frequency bands with signal for band_power generation_type
    print_magnitudes : bool
        If True will print information about SNR and norms of signal and noise. Default: False

    Returns
    -------
    eeg : np.ndarray
        A generated synthetic EEG array
    fmri : np.ndarray
        A generated synthetic fMRI array
    sampling_rates_ratio : int
        Ratio of EEG_SAMPLING_RATE and fmri_fs
    """
    eeg_samples = int(length * EEG_SAMPLING_RATE)
    eeg, fmri = generate_filtered_noise(
        n_samples=eeg_samples, important_channels=main_channels, n_ch=n_ch, bands=bands
    )

    eeg = add_eeg_noise(eeg=eeg, snr=snr, print_magnitudes=print_magnitudes)

    if fmri_fs is not None:
        assert (EEG_SAMPLING_RATE % fmri_fs) == 0, (EEG_SAMPLING_RATE, fmri_fs, EEG_SAMPLING_RATE / fmri_fs)
        sampling_rates_ratio = EEG_SAMPLING_RATE // fmri_fs
        fmri = sn.decimate(fmri, sampling_rates_ratio)
    else:
        sampling_rates_ratio = None

    return eeg, fmri[None, :], sampling_rates_ratio


def generate_dataset(n_ch: int, n_runs: int, length: float, fmri_fs: int, start_t: int, fin_t: int,
                     main_channels: list[int], snr: float, bands: tuple[tuple[int, int]],
                     print_magnitudes: bool = False, independent_runs: bool = False):
    """
    Generate a synthetic signal with additive noise and desired SNR

    Parameters
    ----------
    n_ch : int
        Amount of EEG channels to generate
    n_runs : int
        Amount of synthetic runs to create
    length : float
        Length of a signal in seconds to generate
    fmri_fs : int
        Sampling rate of synthetic fMRI
    start_t : int
        A time step in seconds to start run from (cuts stuff before, used to avoid edge effects from generation)
    fin_t : int
        A time step in seconds to finish run at (cuts stuff after, used to avoid edge effects from generation)
    main_channels : list[int]
        A list with indices of main channels
    snr : float
        A desired SNR
    bands : tuple[tuple[int, int]]
        Frequency bands with signal for band_power generation_type
    print_magnitudes : bool
        If True will print information about SNR and norms of signal and noise. Default: False
    independent_runs : bool
        If True generates runs separately, if False generates everything at once and cuts into chunks. Default: False

    Returns
    -------
    datasets : dict
        A dictionary with generated data. Has the following structure:
        (path_to_eeg_file, path_to_fmri_file):
            eeg: np.ndarray with EEG data
            fmri: np.ndarray with fMRI data
            task: task name
    ch_names : list[str]
        A list of EEG channels loaded from file
    sampling_rates_ratio : int
        A ratio of EEG sampling rate to the fMRI sampling rate
    """
    assert (EEG_SAMPLING_RATE % fmri_fs) == 0, (EEG_SAMPLING_RATE, fmri_fs, EEG_SAMPLING_RATE % fmri_fs)
    sampling_rates_ratio = EEG_SAMPLING_RATE // fmri_fs

    ch_names = [f'EEG {idx}' for idx in range(n_ch)]

    datasets = {}

    if independent_runs:

        for run_idx in range(n_runs):

            eeg, fmri, sampling_rates_ratio = generate_signals(
                n_ch=n_ch, main_channels=main_channels, length=length, fmri_fs=fmri_fs, snr=snr, bands=bands,
                print_magnitudes=print_magnitudes
            )

            datasets[(f'synthetic_eeg_{run_idx}', f'synthetic_fmri_{run_idx}')] = {
                'eeg': eeg[:, start_t * EEG_SAMPLING_RATE:eeg.shape[1] - fin_t * EEG_SAMPLING_RATE],
                'fmri': fmri[:, start_t * fmri_fs:fmri.shape[1] - fin_t * fmri_fs],
                'task': 'dme'
            }

    else:

        eeg_samples = int(length * EEG_SAMPLING_RATE)
        if fmri_fs is None:
            fmri_samples = eeg_samples
        else:
            fmri_samples = int(length * fmri_fs)

        eeg, fmri, sampling_rates_ratio = generate_signals(
            n_ch=n_ch, main_channels=main_channels, length=n_runs * length, fmri_fs=fmri_fs, snr=snr,
            bands=bands, print_magnitudes=print_magnitudes
        )

        assert (eeg.shape[-1] // n_runs) == eeg_samples, (eeg.shape[-1], n_runs, eeg.shape[-1] // n_runs, eeg_samples)
        assert (fmri.shape[-1] // n_runs) == fmri_samples, (
            fmri.shape[-1], n_runs, fmri.shape[-1] // n_runs, fmri_samples
        )

        for run_idx in range(n_runs):

            eeg_run = eeg[:, run_idx * eeg_samples:(run_idx + 1) * eeg_samples]
            fmri_run = fmri[:, run_idx * fmri_samples:(run_idx + 1) * fmri_samples]

            datasets[(f'synthetic_eeg_{run_idx}', f'synthetic_fmri_{run_idx}')] = {
                'eeg': eeg_run[:, start_t * EEG_SAMPLING_RATE:eeg_run.shape[1] - fin_t * EEG_SAMPLING_RATE],
                'fmri': fmri_run[:, start_t * fmri_fs:fmri_run.shape[1] - fin_t * fmri_fs],
                'task': 'dme'
            }

    return datasets, ch_names, sampling_rates_ratio


def train_pls_synthetic_model(
        n_ch: int, n_runs: int, length: float, fmri_fs: int, main_channels: list[int], snr: float,
        bands: tuple[tuple[int, int]], project_root: str, proj_name: str, roi: str, separate_global_trend: bool,
        start_t: int, fin_t: int, min_n_channels: int = 1, max_n_channels: int = 10,
        min_n_components: int = 1, max_n_components: int = 10, disable_tqdm: bool = False,
        print_magnitudes: bool = False, save_model: bool = True, independent_runs: bool = False,
        preprocess_window_size_sec: float = 1, preprocess_overlap_sec: float = 0.5, preprocess_n_shifts: int = 60,
        preprocess_shift_step_sec: float = 0.5, preprocess_bands: tuple[tuple[int, int]] = (
            (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )
):
    """
    Generate a synthetic signal with additive noise and desired SNR

    Parameters
    ----------
    n_ch : int
        Amount of EEG channels to generate
    n_runs : int
        Amount of synthetic runs to create
    length : float
        Length of a signal in seconds to generate
    fmri_fs : int
        Sampling rate of synthetic fMRI
    main_channels : list[int]
        A list with indices of main channels
    snr : float
        A desired SNR
    bands : tuple[tuple[int, int]]
        Frequency bands with signal for band_power generation_type
    project_root : str
        A path to the root of the project
    proj_name : str
        A name of this experiment
    roi : str
        Desired ROI to predict
    separate_global_trend : bool
        If True will calculate and separate Global Trend from ROIs. Default: True
    start_t : int
        A time step in seconds to start run from (cuts stuff before, used to avoid edge effects from generation)
    fin_t : int
        A time step in seconds to finish run at (cuts stuff after, used to avoid edge effects from generation)
    min_n_channels : int
        Minimal amount of EEG channels for gridsearch. Default: 1
    max_n_channels : int
        Maximal amount of EEG channels for gridsearch. Default: 10
    min_n_components : int
        Minimal amount of PLS components for gridsearch. Default: 1
    max_n_components : int
        Maximal amount of PLS components for gridsearch. Default: 10
    disable_tqdm : bool
        If True, does not print tqdm progress bar. Default: False
    print_magnitudes : bool
        If True will print information about SNR and norms of signal and noise. Default: False
    save_model : bool
        If True, saves final PLS model in ONNX format (requires additional dependency). Default: True
    independent_runs : bool
        If True generates runs separately, if False generates everything at once and cuts into chunks. Default: False
    preprocess_window_size_sec : float
        Window size for bandpower calculation in seconds. Default: 1
    preprocess_overlap_sec : float
        Overlap in seconds between window. Default: 0.5
    preprocess_n_shifts : int
        Amount of lag shifts to use for prediction. Default: 60
    preprocess_shift_step_sec : float
        Time difference in seconds for every shift backwards. Default: 0.5
    preprocess_bands : tuple[tuple[int, int]]
        A sequence of bands to use for bandpower calculations. Default: (
                (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )

    Returns
    -------
    grid_search_results : pd.DataFrame
        Summary of the GridSearch results
    test_results : dict
        Summary of the test results
    """
    datasets, eeg_channels, sampling_rates_ratio = generate_dataset(
        n_ch=n_ch,
        n_runs=n_runs,
        length=length,
        fmri_fs=fmri_fs,
        start_t=start_t,
        fin_t=fin_t,
        main_channels=main_channels,
        snr=snr,
        bands=bands,
        print_magnitudes=print_magnitudes,
        independent_runs=independent_runs
    )

    train_datasets, validation_datasets, test_datasets = split_datasets(
        datasets=datasets, sampling_rates_ratio=sampling_rates_ratio
    )

    for ds_name, ds in [('train', train_datasets), ('val', validation_datasets), ('test', test_datasets)]:
        for key, value in ds.items():
            print(ds_name, key, value['eeg'].shape, value['fmri'].shape)

    grid_search_results, test_results = run_pls_pipeline(
        train_datasets=train_datasets, validation_datasets=validation_datasets, test_datasets=test_datasets,
        fmri_sampling_rate=fmri_fs,
        eeg_ch_names=eeg_channels, roi=roi, roi_folder='syntheticroi',
        separate_global_trend=separate_global_trend, sampling_rates_ratio=sampling_rates_ratio,
        project_root=project_root, proj_name=proj_name, min_n_channels=min_n_channels,
        max_n_channels=max_n_channels, min_n_components=min_n_components, max_n_components=max_n_components,
        disable_tqdm=disable_tqdm, save_model=save_model,
        window_size_sec=preprocess_window_size_sec, overlap_sec=preprocess_overlap_sec, n_shifts=preprocess_n_shifts,
        shift_step_sec=preprocess_shift_step_sec, bands=preprocess_bands,
        experiment_type='Synthetic'
    )

    return grid_search_results, test_results


def run_one_test(
        n_runs, target_channels,
        snr, bands,
        min_n_channels,
        max_n_channels,
        iter_num, target_channels_str,
        preprocess_window_size_sec,
        preprocess_overlap_sec,
        preprocess_n_shifts,
        preprocess_shift_step_sec,
        preprocess_bands,
        disable_tqdm
):
    """
    Generate a synthetic signal with additive noise and desired SNR

    Parameters
    ----------
    n_runs : int
        Amount of synthetic runs to create
    target_channels : list[int]
        A list with indices of main channels
    snr : float
        A desired SNR
    bands : tuple[tuple[int, int]]
        Frequency bands with signal for band_power generation_type
    min_n_channels : int
        Minimal amount of EEG channels for gridsearch. Default: 1
    max_n_channels : int
        Maximal amount of EEG channels for gridsearch. Default: 10
    iter_num : int
        Number of test iteration
    target_channels_str : str
        Target channels as a single string with separators ', '
    preprocess_window_size_sec : float
        Window size for bandpower calculation in seconds. Default: 1
    preprocess_overlap_sec : float
        Overlap in seconds between window. Default: 0.5
    preprocess_n_shifts : int
        Amount of lag shifts to use for prediction. Default: 60
    preprocess_shift_step_sec : float
        Time difference in seconds for every shift backwards. Default: 0.5
    preprocess_bands : tuple[tuple[int, int]]
        A sequence of bands to use for bandpower calculations. Default: (
                (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )
    disable_tqdm : bool
        If True, does not print tqdm progress bar

    Returns
    -------
    results : dict
        Aggregated test results
    """

    grid_search_results, test_results = train_pls_synthetic_model(
        n_ch=20,
        n_runs=n_runs,
        length=1000,
        fmri_fs=2,
        main_channels=target_channels,
        snr=snr,
        bands=bands,
        project_root=PROJECT_ROOT,
        proj_name=f'SyntheticSNR{snr}',
        roi=f'Synthetic ROI',
        separate_global_trend=False,
        start_t=5,
        fin_t=5,
        min_n_channels=min_n_channels,
        max_n_channels=max_n_channels,
        min_n_components=1,
        max_n_components=10,
        disable_tqdm=disable_tqdm,
        print_magnitudes=True,
        save_model=True,
        independent_runs=False,
        preprocess_window_size_sec=preprocess_window_size_sec,
        preprocess_overlap_sec=preprocess_overlap_sec,
        preprocess_n_shifts=preprocess_n_shifts,
        preprocess_shift_step_sec=preprocess_shift_step_sec,
        preprocess_bands=preprocess_bands
    )

    results = {
        'SNR': snr,
        'Number of shifts': preprocess_n_shifts,
        'Iteration number': iter_num,
        'Target channels': target_channels_str,
        'Predicted channels': ', '.join(test_results['best_channels']),
        'Train average correlation': test_results['train_correlation'],
        'Test average correlation': test_results['test_mean_correlation'],
        'GridSearch test average correlation': test_results['grid_search_mean_correlation']
    }

    for n_run in range(n_runs):
        assert f'Test correlation, run {n_run}' not in results
        assert f'GridSearch train correlation, run {n_run}' not in results
        assert f'GridSearch test correlation, run {n_run}' not in results

        results[f'Test correlation, run {n_run}'] = test_results['test_per_run_correlations'][f'synthetic_fmri_{n_run}']
        results[f'GridSearch train correlation, run {n_run}'] = test_results['grid_search_train_correlations'][f'synthetic_fmri_{n_run}']
        results[f'GridSearch test correlation, run {n_run}'] = test_results['grid_search_test_correlations'][f'synthetic_fmri_{n_run}']

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Synthetic data Partial least squares baseline')

    parser.add_argument(
        '--n-workers', type=int, help='Number of workers to use.', required=False, default=1
    )

    args = parser.parse_args()

    n_iters = 3
    n_runs = 4
    min_n_channels = 1
    max_n_channels = 10
    results = {
        'SNR': [],
        'Number of shifts': [],
        'Iteration number': [],
        'Target channels': [],
        'Predicted channels': [],
        'Train average correlation': [],
        'Test average correlation': []
    }
    for n_run in range(n_runs):
        results[f'Test correlation, run {n_run}'] = []
        results[f'GridSearch train correlation, run {n_run}'] = []
        results[f'GridSearch test correlation, run {n_run}'] = []
    results['GridSearch test average correlation'] = []
    target_channels = [7, 11, 14, 15, 18]
    target_channels_list = [f'EEG {idx}' for idx in target_channels]
    target_channels_str = ', '.join(target_channels_list)
    bands = [(1, 3), (4, 8), (9, 14), (15, 25)]

    preprocess_window_size_sec = 1
    preprocess_overlap_sec = 0.5
    preprocess_shift_step_sec = 0.5
    preprocess_bands = (
        (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
    )

    secondary_channels = [3, 4, 5]

    all_snrs = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    all_n_shifts = [2, 3, 5, 7, 10, 60]  # default 60
    all_iter_indices = list(range(n_iters))

    if args.n_workers > 1:
        disable_tqdm = True
    else:
        disable_tqdm = False

    all_results = Parallel(n_jobs=args.n_workers)(delayed(run_one_test)(
        n_runs=n_runs,
        target_channels=target_channels,
        snr=snr,
        bands=bands,
        min_n_channels=min_n_channels,
        max_n_channels=max_n_channels,
        iter_num=iter_num,
        target_channels_str=target_channels_str,
        preprocess_window_size_sec=preprocess_window_size_sec,
        preprocess_overlap_sec=preprocess_overlap_sec,
        preprocess_n_shifts=preprocess_n_shifts,
        preprocess_shift_step_sec=preprocess_shift_step_sec,
        preprocess_bands=preprocess_bands,
        disable_tqdm=disable_tqdm
    ) for snr, preprocess_n_shifts, iter_num in itertools.product(all_snrs, all_n_shifts, all_iter_indices))

    for exp in all_results:
        for key in results:
            results[key].append(exp[key])

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(PROJECT_ROOT, 'pls_logs', 'FullChCorrResults.csv'), index=False)
