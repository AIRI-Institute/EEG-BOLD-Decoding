import argparse
import itertools
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

from bw_linker.data_preprocessing.load_brain_data import load_multiple_files, split_datasets
from bw_linker.pls_pipeline.pls_utils import get_results_folder
from bw_linker.pls_pipeline.preprocess_eeg import preprocess_multiple_datasets
from bw_linker.pls_pipeline.sparce_lasso_cca import get_covariance_matrix, find_best_channels
from bw_linker.utils.constants import ALL_SUBJECTS, EEG_CHANNELS, PROJECT_ROOT, CORT_ROIS, SUBCORT_ROIS
from bw_linker.utils.helpers import get_run_name, save_json


def pls_grid_search(
        train_flattened_eeg: np.ndarray, train_fmri: np.ndarray, train_run_labels: np.ndarray,
        lasso_indices_per_channel: dict, train_run_keys: list[str], min_n_channels: int = 1,
        max_n_channels: int = 10, min_n_components: int = 1, max_n_components: int = 10, disable_tqdm: bool = False
):
    """
    Performs a grid search and finds the best number of channels and number of PLS components using leave-one-run-out
    cross-validation

    Parameters
    ----------
    train_flattened_eeg : np.ndarray
        Flattened over features dimensions time-series of EEG features
    train_fmri : np.ndarray
        BOLD signal
    train_run_labels : np.ndarray
        Array with integer indices of EEG recording runs
    lasso_indices_per_channel : dict
        A dictionary with channel names as keys and list of indices of EEG features which refer to this channel as
        values
    train_run_keys : list
        A list with run ids for every run
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

    Returns
    -------
    results : pd.DataFrame
        A DataFrame with grid search results
    best_grid_search_correlations : dict
        Train and test correlations per run of the best grid search parameter combo
    best_n_channels : int
        Amount of channels associated with the largest mean correlation
    best_n_components : int
        Amount of components associated with the largest mean correlation
    best_corr : float
        The largest mean correlation
    """
    results = {
        'split_idx': [],
        'split_id': [],
        'sparse_lasso_coefficient': [],
        'n_channels': [],
        'selected_channels': [],
        'n_components': [],
        'train_correlation': [],
        'test_correlation': []
    }
    grid_search_correlations = {}

    # find channels
    splits = LeaveOneGroupOut()
    for split_idx, (train_index, test_index) in enumerate(
            splits.split(train_flattened_eeg.T, train_fmri, groups=train_run_labels)
    ):
        train_eeg_split = train_flattened_eeg[:, train_index]
        train_fmri_split = train_fmri[train_index]

        test_eeg_split = train_flattened_eeg[:, test_index]
        test_fmri_split = train_fmri[test_index]

        # Covariance matrix
        covariance_matrix = get_covariance_matrix(fmri=train_fmri_split, flattened_eeg=train_eeg_split)

        for n_ch in tqdm(range(min_n_channels, max_n_channels + 1), desc='Iterating over channels',
                         disable=disable_tqdm):

            ch_to_use, ch_to_use_names, lasso_cca = find_best_channels(
                lasso_indices_per_channel=lasso_indices_per_channel,
                desired_n_channels=n_ch,
                covariance_matrix=covariance_matrix,
                print_convergence=not disable_tqdm
            )

            filtered_train_channels_eeg = train_eeg_split[ch_to_use, :]
            filtered_test_channels_eeg = test_eeg_split[ch_to_use, :]

            for n_components in range(min_n_components, max_n_components + 1):

                pls = PLSRegression(n_components=n_components)
                pls.fit(X=np.copy(filtered_train_channels_eeg.T), y=np.copy(train_fmri_split))

                train_preds = pls.predict(X=np.copy(filtered_train_channels_eeg.T)).flatten()
                train_corr = np.corrcoef(train_preds, train_fmri_split)[0, 1]

                test_preds = pls.predict(X=np.copy(filtered_test_channels_eeg.T)).flatten()
                test_corr = np.corrcoef(test_preds, test_fmri_split)[0, 1]

                results['split_idx'].append(split_idx)
                results['split_id'].append(train_run_keys[split_idx])
                results['sparse_lasso_coefficient'].append(lasso_cca.group_lasso_coef)
                results['n_channels'].append(n_ch)
                results['selected_channels'].append(', '.join(ch_to_use_names))
                results['n_components'].append(n_components)
                results['train_correlation'].append(train_corr)
                results['test_correlation'].append(test_corr)

                grid_search_key = (n_ch, n_components)
                if grid_search_key not in grid_search_correlations:
                    grid_search_correlations[grid_search_key] = {'train': {}, 'test': {}}
                run_name = get_run_name(train_run_keys[split_idx][1])
                grid_search_correlations[grid_search_key]['train'][run_name] = train_corr
                grid_search_correlations[grid_search_key]['test'][run_name] = test_corr

    best_n_channels = None
    best_n_components = None
    best_corr = - np.inf
    for (n_ch, n_components), correlation_dicts in grid_search_correlations.items():
        mean_corr = np.mean(list(correlation_dicts['test'].values()))
        if mean_corr > best_corr:
            best_corr = mean_corr
            best_n_channels = n_ch
            best_n_components = n_components
    best_grid_search_correlations = grid_search_correlations[(best_n_channels, best_n_components)]

    results = pd.DataFrame(results)

    return results, best_grid_search_correlations, best_n_channels, best_n_components, best_corr


def run_pls_pipeline(
        train_datasets: dict, validation_datasets: dict, test_datasets: dict, fmri_sampling_rate: int,
        eeg_ch_names: list[str], project_root: str, proj_name: str, experiment_type: str, roi: str, roi_folder: str,
        separate_global_trend: bool, sampling_rates_ratio: int, min_n_channels: int = 1, max_n_channels: int = 10,
        min_n_components: int = 1, max_n_components: int = 10, disable_tqdm: bool = False, save_model: bool = True,
        window_size_sec: float = 1, overlap_sec: float = 0.5, n_shifts: int = 60, shift_step_sec: float = 0.5,
        bands: tuple[tuple[int, int]] = (
                (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )
):
    """
    Runs the whole PLS pipeline. Preprocessing EEG, gridsearch to chose optimal n_components and n_channels. Final
    fit and predict.
    This pipeline tries to reproduce the pipeline from the paper:
    Singer N, Poker G, Dunsky-Moran N, Nemni S, Reznik Balter S, Doron M, Baker T, Dagher A, Zatorre RJ, Hendler T.
    Development and validation of an fMRI-informed EEG model of reward-related ventral striatum activation.
    Neuroimage. 2023 Aug 1;276:120183. doi: 10.1016/j.neuroimage.2023.120183.

    Parameters
    ----------
    train_datasets : dict
        A dictionary with train data. Has the following structure:
        (path_to_eeg_file, path_to_fmri_file):
            eeg: np.ndarray with EEG data
            fmri: np.ndarray with fMRI data
    validation_datasets : dict
        A dictionary with validation data. Has the following structure:
        (path_to_eeg_file, path_to_fmri_file):
            eeg: np.ndarray with EEG data
            fmri: np.ndarray with fMRI data
    test_datasets : dict
        A dictionary with test data. Has the following structure:
        (path_to_eeg_file, path_to_fmri_file):
            eeg: np.ndarray with EEG data
            fmri: np.ndarray with fMRI data
    fmri_sampling_rate : int
        fMRI sampling rate
    eeg_ch_names : list[str]
        A list of names of EEG channels in order of appearance in eeg array
    project_root : str
        A path to the root of the project
    proj_name : str
        A name of this experiment
    experiment_type : str
        A type of the experiment (used to create subfolder with this name for results)
    roi : str
        Desired ROI to predict
    roi_folder : str
        A folder with ROIs. 'roi' is a folder with subcortical ROIs, 'cortrois' is a folder with cortical ROIs
    separate_global_trend : bool
        Whether a global trend should be separated
    sampling_rates_ratio : int
        A ratio of EEG sampling rate to the fMRI sampling rate
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
    save_model : bool
        If True, saves final PLS model in ONNX format (requires additional dependency). Default: True
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
    results : pd.DataFrame
        Summary of the GridSearch results
    test_results : dict
        Summary of the test results
    """
    # merge train and validation
    for ds_name in train_datasets.keys():
        for ds_type in ['eeg', 'fmri']:
            train_datasets[ds_name][ds_type] = np.concatenate(
                (train_datasets[ds_name][ds_type], validation_datasets[ds_name][ds_type]), axis=-1
            )

    train_run_keys = list(train_datasets.keys())
    test_run_keys = list(test_datasets.keys())

    # prepare for crossvalidation
    (train_eeg, train_fmri, train_run_labels, train_flattened_eeg, train_orig_indices,
     train_indices_per_channel) = preprocess_multiple_datasets(
        datasets=train_datasets, fmri_sampling_rate=fmri_sampling_rate,
        eeg_ch_names=list(eeg_ch_names), roi=roi, separate_global_trend=separate_global_trend,
        orig_indices=None, indices_per_channel=None,
        disable_tqdm=disable_tqdm,
        window_size_sec=window_size_sec, overlap_sec=overlap_sec, n_shifts=n_shifts, shift_step_sec=shift_step_sec,
        bands=bands
    )

    (test_eeg, test_fmri, test_run_labels, test_flattened_eeg, test_orig_indices,
     test_indices_per_channel) = preprocess_multiple_datasets(
        datasets=test_datasets, fmri_sampling_rate=fmri_sampling_rate,
        eeg_ch_names=list(eeg_ch_names), roi=roi, separate_global_trend=separate_global_trend,
        orig_indices=deepcopy(train_orig_indices),
        indices_per_channel=deepcopy(train_indices_per_channel),
        disable_tqdm=disable_tqdm,
        window_size_sec=window_size_sec, overlap_sec=overlap_sec, n_shifts=n_shifts, shift_step_sec=shift_step_sec,
        bands=bands
    )

    train_fmri = train_fmri.flatten()
    test_fmri = test_fmri.flatten()
    assert train_fmri.ndim == test_fmri.ndim == 1, (train_fmri.shape, test_fmri.shape)

    assert train_indices_per_channel == test_indices_per_channel, (train_indices_per_channel,
                                                                   test_indices_per_channel)
    lasso_indices_per_channel = deepcopy(train_indices_per_channel)

    assert train_orig_indices == test_orig_indices, (train_orig_indices, test_orig_indices)
    assert train_indices_per_channel == test_indices_per_channel, (train_indices_per_channel, test_indices_per_channel)

    results, best_grid_search_correlations, best_n_channels, best_n_components, best_corr = pls_grid_search(
        train_flattened_eeg=train_flattened_eeg, train_fmri=train_fmri, train_run_labels=train_run_labels,
        lasso_indices_per_channel=lasso_indices_per_channel, train_run_keys=train_run_keys,
        min_n_channels=min_n_channels, max_n_channels=max_n_channels,
        min_n_components=min_n_components, max_n_components=max_n_components, disable_tqdm=disable_tqdm
    )

    save_root = get_results_folder(project_root=project_root, experiment_type=experiment_type, proj_name=proj_name)
    results.to_csv(os.path.join(save_root, 'search_results.csv'), index=False)
    save_json(save_path=os.path.join(save_root, 'config.json'), data=dict(
        fmri_sampling_rate=fmri_sampling_rate,
        eeg_ch_names=eeg_ch_names,
        project_root=str(project_root),
        proj_name=proj_name,
        experiment_type=experiment_type,
        roi=roi,
        roi_folder=roi_folder,
        separate_global_trend=separate_global_trend,
        sampling_rates_ratio=sampling_rates_ratio,
        min_n_channels=min_n_channels,
        max_n_channels=max_n_channels,
        min_n_components=min_n_components,
        max_n_components=max_n_components,
        disable_tqdm=disable_tqdm,
        window_size_sec=window_size_sec,
        overlap_sec=overlap_sec,
        n_shifts=n_shifts,
        shift_step_sec=shift_step_sec,
        bands=bands
    ))

    ch_to_use, best_channels, lasso_cca = find_best_channels(
        lasso_indices_per_channel=lasso_indices_per_channel, desired_n_channels=best_n_channels, covariance_matrix=None,
        fmri=train_fmri, flattened_eeg=train_flattened_eeg, print_convergence=not disable_tqdm
    )

    filtered_train_eeg = train_flattened_eeg[ch_to_use, :]
    filtered_test_eeg = test_flattened_eeg[ch_to_use, :]

    pls = PLSRegression(n_components=best_n_components)
    pls.fit(X=np.copy(filtered_train_eeg.T), y=np.copy(train_fmri))

    train_preds = pls.predict(X=np.copy(filtered_train_eeg.T)).flatten()
    train_corr = np.corrcoef(train_preds, train_fmri)[0, 1]

    per_run_test_correlations = {}
    time_series_root = os.path.join(save_root, 'time_series', 'test')
    Path(time_series_root).mkdir(parents=True, exist_ok=True)

    test_splits = LeaveOneGroupOut()
    for split_idx, (_, test_index) in enumerate(
            test_splits.split(filtered_test_eeg.T, test_fmri, groups=test_run_labels)
    ):
        run_id = test_run_keys[split_idx]
        run_name = get_run_name(run_id[1])

        test_eeg_split = filtered_test_eeg[:, test_index]
        test_fmri_split = test_fmri[test_index]

        preds = pls.predict(X=np.copy(test_eeg_split.T)).flatten()
        corr = np.corrcoef(preds, test_fmri_split)[0, 1]

        ds_results = {f'{roi}_pred': preds, f'{roi}_gt': test_fmri_split}
        ds_results = pd.DataFrame(ds_results)
        ds_results.to_csv(os.path.join(time_series_root, f'{run_name}.csv'), index=False)
        per_run_test_correlations[run_name] = float(corr)

    test_results = {
        'best_n_components': best_n_components,
        'best_channels': best_channels,
        'best_channels_len': len(best_channels),
        'best_n_channels': best_n_channels,
        'sparse_lasso_coefficient': float(lasso_cca.group_lasso_coef),
        'grid_search_train_correlations': best_grid_search_correlations['train'],
        'grid_search_test_correlations': best_grid_search_correlations['test'],
        'grid_search_mean_correlation': float(best_corr),
        'train_correlation': float(train_corr),
        'test_per_run_correlations': per_run_test_correlations,
        'test_mean_correlation': float(np.mean(list(per_run_test_correlations.values())))
    }

    save_json(save_path=os.path.join(save_root, 'test_results.json'), data=test_results)

    np.savez(os.path.join(save_root, 'group_lasso_weights.npz'), **lasso_cca.weights)

    if save_model:
        from skl2onnx import to_onnx

        onx_model = to_onnx(pls, filtered_train_eeg[:1])
        with open(os.path.join(save_root, 'pls_model.onnx'), 'wb') as f:
            f.write(onx_model.SerializeToString())

    return results, test_results


def train_pls_model(
        data_root: str, subjects: list[str], desired_fmri_sampling_rate: int, fmri_interpolation_type: str, roi: str,
        eeg_channels: Optional[list[str]], roi_folder: str, separate_global_trend: bool,
        starting_point_sec: int, rois_for_global_trend: list[str],
        project_root: str, proj_name: str, experiment_type: str, min_n_channels: int = 1, max_n_channels: int = 10,
        min_n_components: int = 1, max_n_components: int = 10, disable_tqdm: bool = False, save_model: bool = True,
        window_size_sec: float = 1, overlap_sec: float = 0.5, n_shifts: int = 60, shift_step_sec: float = 0.5,
        bands: tuple[tuple[int, int]] = (
                (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )
):
    """
    Loads the data and runs the whole PLS pipeline.
    Preprocessing EEG, gridsearch to chose optimal n_components and n_channels. Final fit and predict

    Parameters
    ----------
    data_root : str
        A directory where the dataset is
    subjects : list[str]
        A list of subjects to process
    desired_fmri_sampling_rate : int
        Sampling rate fMRI should be upsampled to
    fmri_interpolation_type : str
        A type of interpolation to perform for fMRI upsampling
    roi : str
        Desired ROI to predict
    eeg_channels : list[str] or None
        A list of EEG channels to load from file. If None, extracts every channel available
    roi_folder : str
        A folder with ROIs. 'roi' is a folder with subcortical ROIs, 'cortrois' is a folder with cortical ROIs.
        Default: 'roi'
    separate_global_trend : bool
        If True will calculate and separate Global Trend from ROIs. Default: True
    starting_point_sec : int
        An amount of seconds to skip in the beginning of the record. Useful to avoid any issues with the beginning of
        files (e.g. device on but experiment did not start yet). Default: 0
    rois_for_global_trend : list[str]
        A list of fMRI ROIs to load from file for Global Trend calculation
    project_root : str
        A path to the root of the project
    proj_name : str
        A name of this experiment
    experiment_type : str
        A type of the experiment (used to create subfolder with this name for results)
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
    save_model : bool
        If True, saves final PLS model in ONNX format (requires additional dependency). Default: True
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
    grid_search_results : pd.DataFrame
        Summary of the GridSearch results
    test_results : dict
        Summary of the test results
    """
    if separate_global_trend:
        assert rois_for_global_trend is not None
    if separate_global_trend and roi == ' Global Trend':
        # some roi so that function doesn't fail extracting some rois before global trend is extracted
        rois_for_extraction = list(rois_for_global_trend)
    else:
        rois_for_extraction = [roi]

    datasets, rois, eeg_channels, sampling_rates_ratio = load_multiple_files(
        root=data_root, subjects=subjects, desired_fmri_sampling_rate=desired_fmri_sampling_rate,
        fmri_interpolation_type=fmri_interpolation_type, rois=rois_for_extraction,
        eeg_channels=eeg_channels, delay_sec=0,
        separate_global_trend=separate_global_trend, starting_point_sec=starting_point_sec,
        rois_for_global_trend=rois_for_global_trend, roi_folder=roi_folder
    )

    train_datasets, validation_datasets, test_datasets = split_datasets(
        datasets=datasets, sampling_rates_ratio=sampling_rates_ratio
    )

    grid_search_results, test_results = run_pls_pipeline(
        train_datasets=train_datasets, validation_datasets=validation_datasets, test_datasets=test_datasets,
        fmri_sampling_rate=desired_fmri_sampling_rate,
        eeg_ch_names=eeg_channels, project_root=project_root, proj_name=proj_name, experiment_type=experiment_type,
        roi=roi, roi_folder=roi_folder,
        separate_global_trend=separate_global_trend, sampling_rates_ratio=sampling_rates_ratio,
        min_n_channels=min_n_channels, max_n_channels=max_n_channels,
        min_n_components=min_n_components, max_n_components=max_n_components,
        disable_tqdm=disable_tqdm, save_model=save_model, window_size_sec=window_size_sec, overlap_sec=overlap_sec,
        n_shifts=n_shifts, shift_step_sec=shift_step_sec, bands=bands
    )

    return grid_search_results, test_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='EEG based fMRI Digital Twin. Partial least squares baseline')

    parser.add_argument(
        '--subjects', type=str, nargs='+', help='IDs of subjects to use.',
        required=False, default=ALL_SUBJECTS
    )
    parser.add_argument(
        '--rois', type=str, nargs='+', help='ROIs to predict.',
        required=False, default=CORT_ROIS
    )
    parser.add_argument(
        '--roi-folder', type=str, help='ROI folder ("roicort" for cortical ROIs or "roi" for subcortical)',
        required=False, choices=['roi', 'roicort'], default='roi'
    )
    parser.add_argument(
        '--n-workers', type=int, help='Number of workers to use.', required=False, default=1
    )
    parser.add_argument(
        '--window-size-sec', type=int, help='Size of a window for bandpower calculation in seconds.',
        required=False, default=1
    )
    parser.add_argument(
        '--overlap-sec', type=float, help='Overlap of windows for bandpower calculation in seconds.',
        required=False, default=0.5
    )
    parser.add_argument(
        '--n-shifts', type=int, help='Number of shifts in EEG data.', required=False, default=60
    )
    parser.add_argument(
        '--shift-step-sec', type=float, help='Size of a single shift of EEG data in seconds.',
        required=False, default=0.5
    )
    parser.add_argument(
        '-save', '--save-model', action=argparse.BooleanOptionalAction,
        help='If True, saves final PLS model in ONNX format (requires additional dependency)',
        default=True
    )
    parser.add_argument(
        '-trend', '--separate-global-trend', action=argparse.BooleanOptionalAction,
        help='If True will separate Global Trend',
        default=True
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    subjects = args.subjects
    rois = list(args.rois)
    roi_folder = args.roi_folder
    n_workers = args.n_workers
    window_size_sec = args.window_size_sec
    overlap_sec = args.overlap_sec
    n_shifts = args.n_shifts
    shift_step_sec = args.shift_step_sec
    save_model = args.save_model
    separate_global_trend = args.separate_global_trend
    if n_workers > 1:
        disable_tqdm = True
    else:
        disable_tqdm = False
    if roi_folder == 'roi':
        experiment_type = 'Subcort'
        rois_for_global_trend = list(SUBCORT_ROIS)
    elif roi_folder == 'roicort':
        experiment_type = 'Cort'
        rois_for_global_trend = list(CORT_ROIS)
    else:
        raise NotImplementedError
    if separate_global_trend:
        experiment_type = experiment_type + 'Trend'
        if ' Global Trend' not in rois:
            rois.append(' Global Trend')
    results = Parallel(n_jobs=n_workers)(delayed(train_pls_model)(
        data_root=os.path.join(PROJECT_ROOT, 'NaturalViewingDataset'),
        subjects=[sub],
        desired_fmri_sampling_rate=2,
        fmri_interpolation_type='cubic',
        roi=roi,
        eeg_channels=EEG_CHANNELS,
        roi_folder=roi_folder,
        separate_global_trend=separate_global_trend,
        starting_point_sec=5,
        rois_for_global_trend=rois_for_global_trend,
        project_root=PROJECT_ROOT,
        proj_name=f'Sub{sub}Roi{roi}',
        experiment_type=experiment_type,
        min_n_channels=1,
        max_n_channels=10,
        min_n_components=1,
        max_n_components=10,
        disable_tqdm=disable_tqdm,
        save_model=save_model,
        window_size_sec=window_size_sec,
        overlap_sec=overlap_sec,
        n_shifts=n_shifts,
        shift_step_sec=shift_step_sec,
        bands=(
            (0, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 40)
        )
    ) for sub, roi in itertools.product(subjects, rois))
