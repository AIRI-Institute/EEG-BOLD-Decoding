import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bw_linker.utils.constants import ALL_SUBJECTS, PROJECT_ROOT, RUNS
from bw_linker.utils.helpers import normalize_time_series
from bw_linker.visualization.plot_utils import (extract_all_brain_wave_runs, extract_all_pls_runs, get_rename_dict,
                                                get_batched_rois, get_roi_pairs)


def get_latex_table(all_corrs_bw: dict, all_corrs_pls: dict, subjects: list[str], delay: int, suffixes: list[str],
                    project_root: str, table_name: str):
    """
    Generate and save as .txt LaTeX table with final test metrics

    Parameters
    ----------
    all_corrs_bw : dict
        A dictionary for BrainWaveLinker with all correlations of structure
        {(run_delay, run_sub, run_suffix): correlations_DataFrame}
    all_corrs_pls : dict
        A dictionary for PLS with all correlations of structure
        {(run_delay, run_sub, run_suffix): correlations_DataFrame}
    subjects : list[str]
        A list of subjects to include
    delay : int
        A delay to use
    suffixes : list[str]
        A list of suffixes which to include (e.g. 'Subcort', 'SubcortTrend', ...)
    project_root : str
        Path to the project root
    table_name : str
        A name of a file table will be saved into
    """
    models = {'BWL': (all_corrs_bw, delay), 'PLS': (all_corrs_pls, None)}

    table = {'Average correlation': {}}

    for suffix in suffixes:

        for model_name, (model_corrs, model_delay) in models.items():
            aggregated_correlations = {}
            all_correlations = []

            for sub in subjects:
                corrs = model_corrs[(model_delay, sub, suffix)].drop('task', axis='columns')

                for roi, corrs_list in corrs.items():
                    if roi not in aggregated_correlations:
                        aggregated_correlations[roi] = []
                    aggregated_correlations[roi] += list(corrs_list)
                    all_correlations += list(corrs_list)

            key = (suffix, model_name)
            for roi, corrs_list in aggregated_correlations.items():
                if roi not in table:
                    table[roi] = {}
                table[roi][key] = np.round(100 * np.mean(corrs_list), 2)

            if ' Global Trend' not in table:
                table[' Global Trend'] = {}
            if key not in table[' Global Trend']:
                table[' Global Trend'][key] = np.nan

            table['Average correlation'][key] = np.round(100 * np.mean(all_correlations), 2)

    rename_dict = get_rename_dict(rois=list(table.keys()), max_row_length=None)
    latex_table = {('ROI', ): []}
    for roi, roi_dict in table.items():
        latex_table[('ROI', )].append(rename_dict[roi])

        for suffix in suffixes:
            for model_name in models.keys():
                key = (suffix, model_name)
                if key not in latex_table:
                    latex_table[key] = []
                latex_table[key].append(roi_dict[key])

    latex_table = pd.DataFrame(latex_table).to_latex(index=False, na_rep='', float_format='%.2f')
    save_dir = os.path.join(project_root, 'visualizations')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_dir, f'{table_name}.txt'), 'w') as output:
        output.write(latex_table)


def get_latex_tables(all_corrs_bw: dict, all_corrs_pls: dict, subjects: list[str], delay: int, suffixes: list[str],
                     project_root: str):
    """
    Generate and save as .txt LaTeX separate tables with final test metrics for cortical and subcortical suffixes

    Parameters
    ----------
    all_corrs_bw : dict
        A dictionary for BrainWaveLinker with all correlations of structure
        {(run_delay, run_sub, run_suffix): correlations_DataFrame}
    all_corrs_pls : dict
        A dictionary for PLS with all correlations of structure
        {(run_delay, run_sub, run_suffix): correlations_DataFrame}
    subjects : list[str]
        A list of subjects to include
    delay : int
        A delay to use
    suffixes : list[str]
        A list of suffixes which to include (e.g. 'Subcort', 'SubcortTrend', ...)
    project_root : str
        Path to the project root
    """
    cort = [suffix for suffix in suffixes if suffix.startswith('Cort')]
    subcort = [suffix for suffix in suffixes if suffix.startswith('Subcort')]
    if len(cort) > 0:
        get_latex_table(
            all_corrs_bw=all_corrs_bw, all_corrs_pls=all_corrs_pls, subjects=subjects, delay=delay, suffixes=cort,
            project_root=project_root, table_name='CortLaTeXTable'
        )
    if len(subcort) > 0:
        get_latex_table(
            all_corrs_bw=all_corrs_bw, all_corrs_pls=all_corrs_pls, subjects=subjects, delay=delay, suffixes=subcort,
            project_root=project_root, table_name='SubcortLaTeXTable'
        )


def plot_boxplots(all_corrs: dict, subjects: list[str], delay: int, suffix: str, model_name: str,
                  title: str, add_average_correlations: bool, savename: str, project_root: str,
                  batch_size: Optional[int] = 7):
    """
    Plots and saves boxplots with correlation metrics per ROI

    Parameters
    ----------
    all_corrs : dict
        A dictionary with all correlations of structure {(run_delay, run_sub, run_suffix): correlations_DataFrame}
    subjects : list[str]
        A list of subjects to include
    delay : int
        A delay to use
    suffix : str
        An experiment suffix to use (e.g. 'Subcort', 'SubcortTrend', ...)
    model_name : str
        A name of the model which generated these metrics
    title : str
        A title of the graph
    add_average_correlations : bool
        If True, will add average correlation into the title of the graph
    savename : str
        A name of the file to save plot to
    project_root : str
        Path to the project root
    batch_size : int or None
        A maximal amount of ROI pairs to plot on a single plot. If None, plots all available from rois. Default: 7
    """
    roi_batches, all_rois = get_batched_rois(suffix=suffix, batch_size=batch_size)
    all_correlations = []
    for sub in subjects:
        corrs = all_corrs[(delay, sub, suffix)]
        for roi in all_rois:
            corrs_roi = (100 * corrs[roi]).to_list()
            all_correlations += list(corrs_roi)
    average_correlation = np.mean(all_correlations)

    for batch_idx, rois in enumerate(roi_batches):

        rename_dict = get_rename_dict(rois=rois, max_row_length=25)
        correlations = {rename_dict[roi]: [] for roi in rois}

        for sub in subjects:
            corrs = all_corrs[(delay, sub, suffix)]
            for roi in rois:
                corrs_roi = (100 * corrs[roi]).to_list()
                correlations[rename_dict[roi]] += list(corrs_roi)

        correlations = pd.DataFrame(correlations)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        sns.boxenplot(correlations, order=[rename_dict[roi] for roi in rois], orient='h', ax=ax)
        if add_average_correlations:
            batch_title = f'{title}. Average correlation: {average_correlation:.2f}%'
        else:
            batch_title = title
        ax.set_title(batch_title, position=(0.35, 0))
        plt.tight_layout()
        save_dir = os.path.join(project_root, 'visualizations', 'correlations', suffix, model_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'{savename}_{batch_idx}.pdf'))
        plt.close(fig)


def plot_delays_heatmaps(heatmap_array: np.ndarray, delay_names: list[str], roi_names: list[str], suffix: str,
                         model_name: str, normalize_rois: bool, title: str, project_root: str, savename: str):
    """
    Plots and saves a regular or normalized heatmap with delays as X axis and ROIs as Y axis

    Parameters
    ----------
    heatmap_array : np.ndarray
        A 2D array with correlations of shape (n_rois, n_delays)
    delay_names : list[str]
        A list with delay names for the graph of length n_delays
    roi_names : list[str]
        A list with ROI names for the graph of length n_rois
    suffix : str
        An experiment suffix to use (e.g. 'Subcort', 'SubcortTrend', ...)
    model_name : str
        A name of the model which generated these metrics
    normalize_rois : bool
        If True, will normalize rows of the heatmap_array (ROIs)
    title : str
        A title of the graph
    savename : str
        A name of the file to save plot to
    project_root : str
        Path to the project root
    """
    if normalize_rois:
        dir_name = 'normalized_heatmaps'
        title = title + '. Normalized per ROI'
        min_array = np.min(heatmap_array, axis=1, keepdims=True)
        max_array = np.max(heatmap_array, axis=1, keepdims=True)
        heatmap_array = (heatmap_array - min_array) / (max_array - min_array)
    else:
        dir_name = 'heatmaps'

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.heatmap(heatmap_array, xticklabels=delay_names, yticklabels=roi_names, ax=ax, annot=True,
                annot_kws={'fontsize': 8})
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    save_dir = os.path.join(project_root, 'visualizations', 'correlations_vs_delay', dir_name, suffix, model_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{savename}.pdf'))
    plt.close(fig)


def plot_delay_curves(heatmap_array: np.ndarray, delay_names: list[str], roi_names: list[str], suffix: str,
                      model_name: str, title: str, project_root: str, savename: str):
    """
    Plots and saves delay curves with delays as X axis and ROIs as Y axis

    Parameters
    ----------
    heatmap_array : np.ndarray
        A 2D array with correlations of shape (n_rois, n_delays)
    delay_names : list[str]
        A list with delay names for the graph of length n_delays
    roi_names : list[str]
        A list with ROI names for the graph of length n_rois
    suffix : str
        An experiment suffix to use (e.g. 'Subcort', 'SubcortTrend', ...)
    model_name : str
        A name of the model which generated these metrics
    title : str
        A title of the graph
    savename : str
        A name of the file to save plot to
    project_root : str
        Path to the project root
    """

    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5), layout='constrained')

    for roi_idx, roi_name in enumerate(roi_names):
        ax.plot(delay_names, heatmap_array[roi_idx, :], label=roi_name)
    ax.set_title(title)
    ax.set_xlabel('Delay, sec')
    ax.set_ylabel('Correlation, %')
    fig.legend(loc='outside lower center', ncol=3, frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1)
    save_dir = os.path.join(project_root, 'visualizations', 'correlations_vs_delay', 'delay_curves', suffix, model_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f'{savename}.pdf'))
    plt.close(fig)


def plot_delay_dependency(all_corrs: dict, subjects: list[str], delays: list[int], suffix: str, model_name: str,
                          title: str, savename: str, project_root: str, batch_size: Optional[int] = 7):
    """
    Plots and saves a regular and normalized heatmaps, and delay curves of correlations with delays as X axis and ROIs
    as Y axis

    Parameters
    ----------
    all_corrs : dict
        A dictionary with all correlations of structure {(run_delay, run_sub, run_suffix): correlations_DataFrame}
    subjects : list[str]
        A list of subjects to include
    delays : list[int]
        A list of delays to include
    suffix : str
        An experiment suffix to use (e.g. 'Subcort', 'SubcortTrend', ...)
    model_name : str
        A name of the model which generated these metrics
    title : str
        A title of the graph
    savename : str
        A name of the file to save plot to
    project_root : str
        Path to the project root
    batch_size : int or None
        A maximal amount of ROI pairs to plot on a single plot. If None, plots all available from rois. Default: 7
    """
    roi_batches, all_rois = get_batched_rois(suffix=suffix, batch_size=batch_size)
    per_delay_correlations = {}
    for delay_idx, delay in enumerate(sorted(delays)):
        delay_correlations = []

        for roi_idx, roi in enumerate(all_rois):
            for sub in subjects:
                corrs = all_corrs[(delay, sub, suffix)]
                delay_correlations = delay_correlations + corrs[roi].to_list()
        per_delay_correlations[delay] = 100 * np.mean(delay_correlations)

    for batch_idx, rois in enumerate(roi_batches):

        rename_dict_22 = get_rename_dict(rois=rois, max_row_length=22)
        rename_dict_25 = get_rename_dict(rois=rois, max_row_length=25)

        order_delays = []
        heatmap_array = np.full((len(rois) + 1, len(delays)), fill_value=np.inf)

        for delay_idx, delay in enumerate(sorted(delays)):
            name = f'{delay - 5} sec'
            order_delays.append(name)

            for roi_idx, roi in enumerate(rois):
                per_roi_correlations = []
                for sub in subjects:
                    corrs = all_corrs[(delay, sub, suffix)]
                    per_roi_correlations = per_roi_correlations + corrs[roi].to_list()
                heatmap_array[roi_idx, delay_idx] = 100 * np.mean(per_roi_correlations)
            heatmap_array[-1, delay_idx] = per_delay_correlations[delay]

        assert not np.isinf(heatmap_array).any()

        display_rois = [rename_dict_25[roi] for roi in rois]
        display_rois.append('Average')
        for normalize_rois in [True, False]:
            plot_delays_heatmaps(
                heatmap_array=heatmap_array, delay_names=order_delays, roi_names=display_rois, suffix=suffix,
                model_name=model_name, normalize_rois=normalize_rois, title=title, project_root=project_root,
                savename=f'{savename}_{batch_idx}'
            )

        display_rois = [rename_dict_22[roi] for roi in rois]
        display_rois.append('Average')
        plot_delay_curves(
            heatmap_array=heatmap_array, delay_names=[d.split(' ')[0] for d in order_delays], roi_names=display_rois,
            suffix=suffix, model_name=model_name, title=title, project_root=project_root,
            savename=f'{savename}_{batch_idx}'
        )


def plot_single_pred_vs_gt(df: pd.DataFrame, roi: str, roi_display_name: str, ax: plt.Axes, idx1: int, idx2: int,
                           xaxis_off: bool):
    """
    Plots a single predictions vs ground truth plot for a single ROI into an axis of the larger plot

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with time series
    roi : str
        A name of the ROI to plot
    roi_display_name : str
        A name of the ROI to plot which will be displayed on a graph
    ax : plt.Axes
        An axis of the overall plot
    idx1 : int
        The first index of the ax to plot to
    idx2 : int
        The second index of the ax to plot to
    xaxis_off : bool
        If True, will remove ticks and labels from the x axis
    """
    gt = df[f'{roi}_gt'].to_numpy()
    pred = df[f'{roi}_pred'].to_numpy()
    corr = np.corrcoef(gt, pred)[0, 1]

    x_axis = np.arange(gt.shape[0]) / 2
    ax[idx1, idx2].plot(x_axis, normalize_time_series(gt, axis=None, keepdims=False), label='Ground Truth')
    ax[idx1, idx2].plot(x_axis, normalize_time_series(pred, axis=None, keepdims=False), label='Prediction')
    ax[idx1, idx2].set_title(f'{roi_display_name}\nCorrelation: {corr * 100:.2f}%')
    ax[idx1, idx2].yaxis.set_visible(False)
    if xaxis_off:
        ax[idx1, idx2].xaxis.set_visible(False)
    if not xaxis_off:
        ax[idx1, idx2].set_xlabel('Time, sec')


def plot_preds_vs_gt(all_series: dict, sub: str, delay: int, suffix: str, model_name: str,
                     title: str, task_name: str, project_root: str, batch_size: Optional[int] = 7):
    """
    Plots and saves a predictions vs ground truth time series plot with all ROIs

    Parameters
    ----------
    all_series : dict
        A dictionary with all time series of structure
        {(run_delay, run_sub, run_suffix): {task_name: time_series_DataFrame}}
    sub : str
        A subject to plot for
    delay : int
        A delay to use
    suffix : str
        An experiment suffix to use (e.g. 'Subcort', 'SubcortTrend', ...)
    model_name : str
        A name of the model which generated these metrics
    title : str
        A title of the graph
    task_name : str
        A name of the task. Will also save plot into the file with the same name
    project_root : str
        Path to the project root
    batch_size : int or None
        A maximal amount of ROI pairs to plot on a single plot. If None, plots all available from rois. Default: 7
    """
    global_trend_name = ' Global Trend'
    roi_batches, all_rois = get_batched_rois(suffix=suffix, batch_size=batch_size)
    rename_dict = get_rename_dict(rois=all_rois, max_row_length=25)

    df = all_series[(delay, sub, suffix)][task_name]

    for batch_idx, rois in enumerate(roi_batches):

        roi_pairs = get_roi_pairs(rois=rois)

        figsize = (7, 15)
        if (len(rois) % 2) == 1:
            assert global_trend_name in rois
            fig, ax = plt.subplots(1 + len(rois) // 2, 2, figsize=figsize, layout='constrained')
            fig.delaxes(ax[0, 1])
            start_idx = 1
        else:
            assert global_trend_name not in rois
            fig, ax = plt.subplots(len(rois) // 2, 2, figsize=figsize, layout='constrained')
            start_idx = 0

        for roi in roi_pairs:
            if (((len(rois) % 2) == 0) and (start_idx == (len(roi_pairs) - 1)) or
                    ((len(rois) % 2) == 1) and (start_idx == len(roi_pairs))):
                l_xaxisoff = False
                r_xaxisoff = False
            else:
                l_xaxisoff = True
                r_xaxisoff = True

            left_roi = f' Left {roi}'
            right_roi = f' Right {roi}'
            plot_single_pred_vs_gt(
                df=df, roi=left_roi, roi_display_name=rename_dict[left_roi], ax=ax, idx1=start_idx, idx2=0,
                xaxis_off=l_xaxisoff
            )
            plot_single_pred_vs_gt(
                df=df, roi=right_roi, roi_display_name=rename_dict[right_roi], ax=ax, idx1=start_idx, idx2=1,
                xaxis_off=r_xaxisoff
            )
            start_idx += 1

        if global_trend_name in rois:
            plot_single_pred_vs_gt(
                df=df, roi=global_trend_name, roi_display_name='Global Trend', ax=ax, idx1=0, idx2=0, xaxis_off=True
            )
            legend_location = (0.512574, 0.897368)
        else:
            legend_location = 'outside lower center'

        handles, labels = ax[-1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=legend_location, ncols=2, frameon=True, fancybox=True, framealpha=1,
                   shadow=True, borderpad=0.4)

        fig.suptitle(title)
        save_dir = os.path.join(project_root, 'visualizations', 'predictions_vs_gt', suffix, model_name,
                                f'delay-{delay}', f'sub-{sub}')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'{task_name}_{batch_idx}.pdf'))
        plt.close(fig)


def plot_all_graphs(subjects: list[str], delays: list[int], suffixes: list[str], project_root: str, wandb_root: str,
                    pls_root: str, split_name: str = 'test', main_delay: int = 9,
                    project_name: str = 'EEG-BOLD-Decoding', batch_size: Optional[int] = 7):
    """
    Plots and saves the following graphs: boxplots for every subject and all combined; delay-correlation heatmaps for
    every subject; predictions vs ground truth time series for every subject and task. Also, saves a LaTeX code for a
    comparison table with metrics

    Parameters
    ----------
    subjects : list[str]
        A list of subjects to include
    delays : list[int]
        A list of delays to include
    suffixes : list[str]
        A list of suffixes which to include (e.g. 'Subcort', 'SubcortTrend', ...)
    project_root : str
        Path to the project root
    wandb_root : str
        Path to the Weights and Biases logs and checkpoints
    pls_root : str
        Path to the PLS logs and results
    split_name : str
        A name of the split for extraction (e.g. 'train', 'validation, 'test'). Default: 'test'
    main_delay : int
        Main delay to use. It will be used to plot boxplots, time series graphs and for the table with metrics.
        Default: 9
    project_name : str
        A name of the Weights and Biases project. Default: 'EEG-BOLD-Decoding'
    batch_size : int or None
        A maximal amount of ROI pairs to plot on a single plot. If None, plots all available from rois. Default: 7
    """

    all_corrs_bw, all_series_bw = extract_all_brain_wave_runs(
        results_root=wandb_root, subjects=subjects, delays=delays, suffixes=suffixes, split_name=split_name,
        project_name=project_name
    )

    all_corrs_pls, all_series_pls = extract_all_pls_runs(
        results_root=pls_root, subjects=subjects, suffixes=suffixes, split_name=split_name
    )

    for (model_name, all_corrs_model, all_series_model, delay_model) in (
            ('BrainWaveLinker', all_corrs_bw, all_series_bw, main_delay), ('PLS', all_corrs_pls, all_series_pls, None)
    ):
        for suffix in suffixes:
            if suffix.endswith('Trend'):
                prefix = 'Relative'
            else:
                prefix = 'Absolute'

            plot_boxplots(
                all_corrs=all_corrs_model, subjects=subjects, delay=delay_model, suffix=suffix,
                model_name=model_name, title=f'{prefix} BOLD correlations',
                add_average_correlations=True, savename=f'{prefix}CorrsAllSubjects', project_root=project_root,
                batch_size=batch_size
            )

            if model_name == 'BrainWaveLinker':
                plot_delay_dependency(all_corrs=all_corrs_model, subjects=subjects, delays=delays, suffix=suffix,
                                      model_name=model_name,
                                      title=f'{prefix} correlations against delays',
                                      savename=f'{prefix}DelayCorrsAllSubjects', project_root=project_root,
                                      batch_size=batch_size)

            for sub in tqdm(subjects, desc=f'Processing per subject graphs for {suffix} and model {model_name}'):
                plot_boxplots(
                    all_corrs=all_corrs_model, subjects=[sub], delay=delay_model, suffix=suffix,
                    model_name=model_name, title=f'{prefix} BOLD correlations. Subject {sub}',
                    add_average_correlations=True, savename=f'{prefix}CorrsSub{sub}', project_root=project_root,
                    batch_size=batch_size
                )

                if model_name == 'BrainWaveLinker':

                    plot_delay_dependency(all_corrs=all_corrs_model, subjects=[sub], delays=delays, suffix=suffix,
                                          model_name=model_name,
                                          title=f'{prefix} correlations against delays, subject {sub}',
                                          savename=f'{prefix}DelayCorrsSub{sub}', project_root=project_root,
                                          batch_size=batch_size)

                for session_name, task_name in RUNS:

                    plot_preds_vs_gt(all_series=all_series_model, sub=sub, delay=delay_model, suffix=suffix,
                                     model_name=model_name,
                                     title=f'Subject: {sub}. Session: {session_name}. Task: {task_name}',
                                     task_name=f'sub-{sub}_ses-{session_name}_task-{task_name}',
                                     project_root=project_root,
                                     batch_size=batch_size)

    get_latex_tables(
        all_corrs_bw=all_corrs_bw, all_corrs_pls=all_corrs_pls, subjects=subjects, delay=main_delay,
        suffixes=suffixes, project_root=project_root
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot topographies for BrainWaveLinker spatial filters.')

    parser.add_argument(
        '--subjects', type=str, nargs='+', help='IDs of subjects to use.',
        required=False, default=ALL_SUBJECTS
    )
    parser.add_argument(
        '--delays', type=int, nargs='+', help='Delays to use.',
        required=False, default=list(range(0, 16))
    )
    parser.add_argument(
        '--suffixes', type=str, nargs='+', help='Experiment suffixes to use.', required=False,
        default=('SubcortTrend', 'Subcort', 'CortTrend', 'Cort')
    )
    parser.add_argument(
        '--project-root', type=str, help='Path to the project root.', required=False, default=PROJECT_ROOT
    )
    parser.add_argument(
        '--wandb-root', type=str, help='Path to the Weights and Biases logs and checkpoints.',
        required=False, default=os.path.join(PROJECT_ROOT, 'wandb_logs')
    )
    parser.add_argument(
        '--pls-root', type=str, help='Path to the PLS logs and results.',
        required=False, default=os.path.join(PROJECT_ROOT, 'pls_logs')
    )
    parser.add_argument(
        '--split-name', type=str, help='Name of the split to plot.', required=False,
        default='test'
    )
    parser.add_argument(
        '-d', '--main-delay', type=int,
        help='Main delay to use. It will be used to plot boxplots, time series graphs and for the table with metrics.',
        required=False, default=9
    )
    parser.add_argument(
        '--project-name', type=str, help='Name of the project in Weights and Biases.', required=False,
        default='EEG-BOLD-Decoding'
    )
    parser.add_argument(
        '-b', '--batch-size', type=int,
        help='A maximal amount of ROI names to plot on a single graph. The actual amount of ROIs will be double that'
             'because it will include Left and Right ROI for every index in range(batch_size). Also, if there is a'
             'Global Trend in an experiment, it will be added to every plot.',
        required=False, default=7
    )

    return parser.parse_args()


if __name__ == '__main__':
    plt.rc('font', size=12)

    args = parse_arguments()

    plot_all_graphs(
        subjects=args.subjects, delays=args.delays, suffixes=args.suffixes, project_root=args.project_root,
        wandb_root=args.wandb_root, pls_root=args.pls_root, split_name=args.split_name,
        main_delay=args.main_delay, project_name=args.project_name, batch_size=args.batch_size
    )
