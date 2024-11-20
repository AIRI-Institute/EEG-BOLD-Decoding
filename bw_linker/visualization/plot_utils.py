import math
import os
from glob import glob
from typing import Optional

import pandas as pd
from tqdm import tqdm

from bw_linker.utils.constants import CORT_ROIS, SUBCORT_ROIS
from bw_linker.utils.helpers import load_json
from bw_linker.visualization.get_wandb_mapping import load_wandb_mapping


def get_rename_dict(rois: list[str], max_row_length: Optional[int]):
    """
    Returns a dictionary that maps ROI names in experiments to more compact for plots and tables

    Parameters
    ----------
    rois : list[str]
        A list of ROIs which need to be renamed
    max_row_length : int or None
        If given, will split long ROI names into several rows of maximal length max_row_length

    Returns
    -------
    rename_dict : dict
        A dictionary with {name: plot_name} pairs
    """
    rename_dict = {}
    for roi in list(rois):
        if roi == ' Global Trend':
            rename_dict[roi] = 'Global Trend'
        else:
            new_roi = roi.lstrip()
            new_roi = new_roi.split(' ')
            if new_roi[0] == 'Left':
                prefix = 'L.'
            elif new_roi[0] == 'Right':
                prefix = 'R.'
            else:
                rename_dict[roi] = roi
                continue
            roi_name = ' '.join(new_roi[1:])
            roi_name = shorten_roi_name(roi=roi_name)
            new_roi = f'{prefix} {roi_name}'
            if max_row_length:
                parts = new_roi.split(' ')
                rows = []
                current_row = parts[0]
                for part in parts[1:]:
                    if (len(current_row) + len(part) + 1) <= max_row_length:
                        current_row = current_row + ' ' + part
                    else:
                        rows.append(current_row)
                        current_row = part
                rows.append(current_row)
                new_roi = '\n'.join(rows)
            rename_dict[roi] = new_roi
    return rename_dict


def shorten_roi_name(roi: str):
    """
    Returns the same ROI but with shortened name. This is used because the original names of many
    cortical ROIs do not fit on the graphs

    Parameters
    ----------
    roi : str
        An original name of ROI

    Returns
    -------
    roi_sh : str
        The same ROI with shortened name
    """
    sh_names_mapping = {
        'Inferior Frontal Gyrus in pars triangularis': 'Inferior Frontal Gyrus, triangularis',
        'Inferior Frontal Gyrus in pars opercularis': 'Inferior Frontal Gyrus, opercularis',
        'Superior Temporal Gyrus in anterior division': 'Superior Temporal Gyrus, anterior',
        'Superior Temporal Gyrus in posterior division': 'Superior Temporal Gyrus, posterior',
        'Middle Temporal Gyrus in anterior division': 'Middle Temporal Gyrus, anterior',
        'Middle Temporal Gyrus in posterior division': 'Middle Temporal Gyrus, posterior',
        'Middle Temporal Gyrus in temporooccipital part': 'Middle Temporal Gyrus, temporooccipital',
        'Inferior Temporal Gyrus in anterior division': 'Inferior Temporal Gyrus, anterior',
        'Inferior Temporal Gyrus in posterior division': 'Inferior Temporal Gyrus, posterior',
        'Inferior Temporal Gyrus in temporooccipital part': 'Inferior Temporal Gyrus, temporooccipital',
        'Supramarginal Gyrus in anterior division': 'Supramarginal Gyrus, anterior',
        'Supramarginal Gyrus in posterior division': 'Supramarginal Gyrus, posterior',
        'Lateral Occipital Cortex in superior division': 'Lateral Occipital Cortex, superior',
        'Lateral Occipital Cortex in inferior division': 'Lateral Occipital Cortex, inferior',
        'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)': 'Juxtapositional Lobule Cortex',
        'Cingulate Gyrus in anterior division': 'Cingulate Gyrus, anterior',
        'Cingulate Gyrus in posterior division': 'Cingulate Gyrus, posterior',
        'Parahippocampal Gyrus in anterior division': 'Parahippocampal Gyrus, anterior',
        'Parahippocampal Gyrus in posterior division': 'Parahippocampal Gyrus, posterior',
        'Temporal Fusiform Cortex in anterior division': 'Temporal Fusiform Cortex, anterior',
        'Temporal Fusiform Cortex in posterior division': 'Temporal Fusiform Cortex, posterior',
        "Heschl's Gyrus (includes H1 and H2)": "Heschl's Gyrus",
    }
    if roi in sh_names_mapping:
        return sh_names_mapping[roi]
    else:
        return roi


def get_batched_rois(suffix: str, batch_size: Optional[int] = 7):
    """
    Returns a list of lists with expected fMRI ROIs. Every sublist consists of batch_size * 2 ROIs (Left/Right for each
    ROI in batch_size)

    Parameters
    ----------
    suffix : str
        A suffix in the name of the experiment (e.g. 'Subcort', 'SubcortTrend', ...)
    batch_size : int or None
        Amount of ROIs in a single group. If None, adds all to a single group. Default: 7

    Returns
    -------
    batched_rois : list[list[str]]
        A list of batches with expected ROIs for the experiment
    rois : list[str]
        A list of all expected ROIs for the experiment
    """
    rois_map = {
        'Subcort': list(SUBCORT_ROIS),
        'SubcortTrend': list(SUBCORT_ROIS) + [' Global Trend'],
        'Cort': list(CORT_ROIS),
        'CortTrend': list(CORT_ROIS) + [' Global Trend']
    }
    if batch_size is None:
        return [rois_map[suffix]], rois_map[suffix]
    roi_pairs = get_roi_pairs(rois=rois_map[suffix])
    n_batches = math.ceil(len(roi_pairs) / batch_size)
    rois = []
    for b_idx in range(n_batches):
        batch = get_rois_from_pairs(
            roi_pairs=roi_pairs[b_idx * batch_size:(b_idx + 1) * batch_size], add_global_trend=suffix.endswith('Trend')
        )
        rois.append(batch)
    return rois, rois_map[suffix]


def get_roi_pairs(rois: list[str]):
    """
    Returns a list of ROI pairs (ROIs without Left/Right indication) from rois (i.e. if there is
    ' Left Hippocampus' and ' Right Hippocampus' the output list will only have 'Hippocampus').
    Ignores ' Global Trend'

    Parameters
    ----------
    rois : list[str]
        A list of ROIs

    Returns
    -------
    roi_pairs : list[str]
        A list of ROI pairs (ROIs without Left/Right indication)
    """
    roi_pairs = []
    for roi in rois:
        if roi != ' Global Trend':
            roi = ' '.join(roi.lstrip().split(' ')[1:])
            if roi not in roi_pairs:
                roi_pairs.append(roi)
    return roi_pairs


def get_rois_from_pairs(roi_pairs: list[str], add_global_trend: bool):
    """
    Makes a list of ROIs from ROI pairs by adding Left/Right and optionally Global Trend

    Parameters
    ----------
    roi_pairs : list[str]
        A list of ROI pairs (ROIs without Left/Right indication)
    add_global_trend
        If True, adds ' Global Trend' ROI

    Returns
    -------
    rois : list[str]
        A list of ROIs
    """
    rois = []
    for roi in roi_pairs:
        rois.append(f' Left {roi}')
        rois.append(f' Right {roi}')
    if add_global_trend:
        rois.append(' Global Trend')
    return rois


def get_series(results_dir: str, split_name: str):
    """
    Returns a dictionary of time series per task extracted from experiments

    Parameters
    ----------
    results_dir : str
        A path to the directory with experiment results
    split_name : str
        A name of the split for extraction (e.g. 'train', 'validation, 'test')

    Returns
    -------
    series : dict
        A dictionary of the structure {task_name: time_series_DataFrame}
    """
    series = {}
    trend_rename = {f' Global Regressive Trend_{tag}': f' Global Trend_{tag}' for tag in ['pred', 'gt']}
    for p in glob(os.path.join(results_dir, 'time_series', split_name, '*.csv')):
        series_df = pd.read_csv(p)
        series_df = series_df.rename(columns=trend_rename)
        series[os.path.splitext(os.path.basename(p))[0]] = series_df
    return series


def extract_all_brain_wave_runs(
        results_root: str, subjects: list[str], delays: list[int], suffixes: list[str], split_name: str,
        project_name: str = 'EEG-BOLD-Decoding'
):
    """
    Return all correlations and times series for all requested runs for BrainWaveLinker experiments

    Parameters
    ----------
    results_root : str
        A root to the folder with all results
    subjects : list[str]
        A list of subjects to include
    delays : list[int]
        A list of delays to include
    suffixes : list[str]
        A list of suffixes which to include (e.g. 'Subcort', 'SubcortTrend', ...)
    split_name : str
        A name of the split for extraction (e.g. 'train', 'validation, 'test')
    project_name : str
        A name of the Weights and Biases project. Default: 'EEG-BOLD-Decoding'

    Returns
    -------
    all_corrs : dict
        A dictionary with all correlations of structure {(run_delay, run_sub, run_suffix): correlations_DataFrame}
    all_series : dict
        A dictionary with all time series of structure
        {(run_delay, run_sub, run_suffix): {task_name: time_series_DataFrame}}
    """
    results_root = os.path.join(results_root, project_name, project_name)
    matched_paths = load_wandb_mapping()
    subjects = set(subjects)
    suffixes = set(suffixes)
    delays = set(delays)
    all_corrs = {}
    all_series = {}
    for run_parameters, run_id in tqdm(
            matched_paths.items(), total=len(matched_paths), desc='Extracting BrainWaveLinker metrics'
    ):
        delay, sub, suffix = run_parameters
        if (delay not in delays) or (sub not in subjects) or (suffix not in suffixes):
            continue
        _, rois_run = get_batched_rois(suffix=suffix, batch_size=None)
        results_dir = os.path.join(results_root, run_id)
        corrs = pd.read_csv(os.path.join(results_dir, 'test_full_correlations.csv'))
        corrs = corrs.rename(columns={' Global Regressive Trend': ' Global Trend'})
        corrs = corrs[['task'] + rois_run]
        series = get_series(results_dir=results_dir, split_name=split_name)
        all_corrs[run_parameters] = corrs
        all_series[run_parameters] = series
    return all_corrs, all_series


def merge_pls_correlations(sub_corrs: dict, rois: list[str]):
    """
    Return a DataFrame of correlations with concatenated ROIs from PLS experiments

    Parameters
    ----------
    sub_corrs : dict
        A dictionary with per run data of the structure {task: {roi: correlation}}
    rois : list[str]
        A list of required ROIs

    Returns
    -------
    sub_corrs_df : pd.DataFrame
        A DataFrame with all requested ROIs
    """
    sub_corrs_df = {'task': []}
    for roi in rois:
        sub_corrs_df[roi] = []

    for task, task_corrs in sub_corrs.items():
        sub_corrs_df['task'].append(task)
        for roi, roi_corr in task_corrs.items():
            sub_corrs_df[roi].append(roi_corr)
    sub_corrs_df = pd.DataFrame(sub_corrs_df)
    sub_corrs_df = sub_corrs_df[['task'] + rois]
    return sub_corrs_df


def merge_pls_series(sub_series: dict):
    """
    Return a per task dictionary of DataFrames with time series with concatenated ROIs from PLS experiments

    Parameters
    ----------
    sub_series : dict
        A dictionary with per task time series DataFrames of the structure {task: [roi1_df, roi2_df, ...]}

    Returns
    -------
    sub_series_merged : dict
        A dictionary with merged DataFrames with all ROIs per task
    """
    sub_series_merged = {}
    for task, series_list in sub_series.items():
        sub_series_merged[task] = pd.concat(series_list, axis='columns')
    return sub_series_merged


def extract_all_pls_runs(results_root: str, subjects: list[str], suffixes: list[str], split_name: str):
    """
    Return all correlations and times series for all requested runs for PLS experiments

    Parameters
    ----------
    results_root : str
        A root to the folder with all results
    subjects : list[str]
        A list of subjects to include
    suffixes : list[str]
        A list of suffixes which to include (e.g. Subcort, SubcortTrend, ...)
    split_name : str
        A name of the split for extraction (e.g. 'train', 'validation, 'test')

    Returns
    -------
    all_corrs : dict
        A dictionary with all correlations of structure {(None, run_sub, run_suffix): correlations_DataFrame}
    all_series : dict
        A dictionary with all time series of structure
        {(None, run_sub, run_suffix): {task_name: time_series_DataFrame}}
    """
    all_corrs = {}
    all_series = {}

    for suffix in suffixes:
        _, all_rois = get_batched_rois(suffix=suffix, batch_size=None)

        for sub in subjects:
            sub_corrs = {}
            sub_series = {}

            for roi in tqdm(
                    all_rois, total=len(all_rois), desc=f'Extracting PLS metrics for {suffix}, sub {sub}'
            ):
                run_name = f'Sub{sub}Roi{roi}'
                results_path = glob(os.path.join(results_root, suffix, run_name, '**/**/test_results.json'))
                assert len(results_path) == 1, (results_path, len(results_path))
                results_path = results_path[0]
                results_dir = os.path.dirname(results_path)
                test_results = load_json(json_path=results_path)

                for task, corr in test_results['test_per_run_correlations'].items():
                    if task not in sub_corrs:
                        sub_corrs[task] = {}
                    sub_corrs[task][roi] = corr

                series = get_series(results_dir=results_dir, split_name=split_name)
                for task, df in series.items():
                    if task not in sub_series:
                        sub_series[task] = []
                    sub_series[task].append(df)

            sub_corrs_df = merge_pls_correlations(sub_corrs=sub_corrs, rois=all_rois)
            sub_series_merged = merge_pls_series(sub_series=sub_series)

            run_parameters = (None, sub, suffix)
            all_corrs[run_parameters] = sub_corrs_df
            all_series[run_parameters] = sub_series_merged

    return all_corrs, all_series
