import argparse
import os
from glob import glob
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

from bw_linker.brain_wave_pipeline.brain_wave_linker import BrainWaveLinker
from bw_linker.brain_wave_pipeline.training_pipeline import get_torch_datasets
from bw_linker.utils.constants import ALL_SUBJECTS, EEG_SAMPLING_RATE, PROJECT_ROOT, RUNS
from bw_linker.utils.helpers import get_window_sizes_sample, load_json
from bw_linker.visualization.get_wandb_mapping import load_wandb_mapping


def get_experiment_metadata(
        delay: int, sub: str, suffix: str, wandb_root: str, project_name: str = 'EEG-BOLD-Decoding'
):
    """
    Extract a metadata from the experiment (a.k.a. model config, experiment config and checkpoint path)

    Parameters
    ----------
    delay : int
        A delay of the experiment
    sub : str
        A subject of the experiment
    suffix : str
        An experiment suffix (e.g. 'SubcortTrend', 'Subcort', ...)
    wandb_root : str
        Path to the Weights and Biases logs and checkpoints
    project_name : str
        Name of the project in Weights and Biases. Default: 'EEG-BOLD-Decoding'

    Returns
    -------
    ckpt_path : str
        A path to the best checkpoint
    config : dict
        An experiment configuration dict
    model_config: dict
        A configuration file for the neural network
    """
    matched_runs = load_wandb_mapping()
    run_id = matched_runs[(delay, sub, suffix)]
    ckpt_path = list(glob(os.path.join(wandb_root, project_name, project_name, run_id, 'checkpoints', '*.ckpt')))
    assert len(ckpt_path) == 1, ckpt_path
    ckpt_path = ckpt_path[0]
    dir_mapping = {
        'Subcort': 'subcort',
        'SubcortTrend': 'subcort-trend',
        'Cort': 'cort',
        'CortTrend': 'cort-trend'
    }
    config_path = os.path.join(PROJECT_ROOT, 'configs', dir_mapping[suffix], f'delay-{delay}', f'sub-{sub}-train.json')
    full_config = load_json(config_path)
    model_config = full_config['model_config']
    config = full_config['config']

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
    return ckpt_path, config, model_config


def plot_topographies(eeg_data_path: str, eeg_channels: list[str], filters_data: np.ndarray, suffix: str,
                      project_root: str, task_info: str, n_patterns: int = 6):
    """
    Plots topographies for a single task

    Parameters
    ----------
    eeg_data_path : str
        A path to the EEG file with metadata for plots
    eeg_channels : list[str]
        A list of EEG channels
    filters_data : np.ndarray
        An array with spatial brain pattern data from spatial filters
    suffix : str
        An experiment suffix (e.g. 'SubcortTrend', 'Subcort', ...)
    project_root : str
        Path to the project root
    task_info : str
        A name of the EEG file from which info for savefile name and title will be extracted
    n_patterns : int
        Amount of strongest patterns to plot per task. Please use numbers divisible by 3. Default: 6

    Returns
    -------
    status : int
        Return 0 if no errors happened
    """
    assert (n_patterns % 3) == 0, 'Function plots patterns into 3 columns. Please use n_patterns divisible by 3'
    eeg = mne.io.read_raw_brainvision(eeg_data_path, verbose='WARNING')
    eeg.pick(list(eeg_channels))

    evoked = mne.EvokedArray(
        data=filters_data,  # brain_pattern,
        info=eeg.info
    )

    fig = evoked.plot_topomap(
        times=np.arange(n_patterns) / evoked.info['sfreq'], ch_type='eeg', units='', scalings={'eeg': 1},
        time_format='', ncols=3, nrows=n_patterns // 3, show=False
    )

    axes = fig.axes
    for idx in range(n_patterns):
        axes[idx].set_xlabel(str(idx + 1))

    task_info = task_info.split('_')
    sub = task_info[0].split('-')[-1]
    ses = task_info[1].split('-')[-1]
    task = task_info[2].split('-')[-1]

    fig.suptitle(f'Patterns. Subject {sub}, session {ses}, task {task}')

    savename = '_'.join(task_info[:-1])
    save_root = os.path.join(project_root, 'visualizations', 'brain_patterns', suffix, f'sub-{sub}')

    Path(save_root).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(save_root, f'{savename}.pdf'))
    plt.close(fig)

    return 0


def plot_topographies_all_tasks(sub: str, delay: int, suffix: str, wandb_root: str,
                                project_root: str, project_name: str = 'EEG-BOLD-Decoding', n_patterns: int = 6):
    """
    Plots and saves graphs for topographies for all tasks

    Parameters
    ----------
    sub : str
        A subject of the experiment
    delay : int
        A delay of the experiment
    suffix : str
        An experiment suffix (e.g. 'SubcortTrend', 'Subcort', ...)
    wandb_root : str
        Path to the Weights and Biases logs and checkpoints
    project_root : str
        Path to the project root
    project_name : str
        Name of the project in Weights and Biases. Default: 'EEG-BOLD-Decoding'
    n_patterns : int
        Amount of strongest patterns to plot per task. Please use numbers divisible by 3. Default: 6
    """
    ckpt_path, config, model_config = get_experiment_metadata(
        delay=delay, sub=sub, suffix=suffix, wandb_root=wandb_root, project_name=project_name
    )

    train_dataset, validation_dataset, test_dataset, eeg_channels, rois = get_torch_datasets(
        project_root=project_root, config=config
    )

    model = BrainWaveLinker(
        in_channels=len(eeg_channels),
        out_channels=len(rois),
        **model_config
    )

    checkpoint = torch.load(ckpt_path, weights_only=True, map_location='cpu')
    state_dict = {}

    # Original checkpoint for a torch.nn.Module was inside a lightning module, therefore removing additional prefix
    for key, value in checkpoint['state_dict'].items():
        key = key.split('.')
        assert key[0] == 'model'
        new_key = '.'.join(key[1:])
        state_dict[new_key] = value
    model.load_state_dict(state_dict)
    model.eval()

    filtering_weights = model.spatial_filer.weight
    assert filtering_weights.size(-1) == 1
    filtering_weights = filtering_weights.squeeze(-1)
    np_filters = filtering_weights.detach().cpu().numpy()

    for ds in train_dataset.datasets:

        eeg, gt_fmri, (eeg_data_path, fmri_data_path) = ds.get_all_data()
        np_eeg = eeg.detach().cpu().numpy()
        corr_matrix = np.corrcoef(np_eeg, rowvar=True)

        values_to_plot = []
        for f_idx in range(np_filters.shape[0]):
            brain_pattern = corr_matrix @ np_filters[f_idx, :]

            values_to_plot.append(
                (brain_pattern, np.linalg.norm(brain_pattern))
            )

        values_to_plot = sorted(values_to_plot, key=lambda i: i[1], reverse=True)
        filters_data = []
        for f_idx in range(n_patterns):
            brain_pattern = values_to_plot[f_idx][0]
            filters_data.append(brain_pattern)

        filters_data = np.stack(filters_data, axis=-1)

        try:
            plot_topographies(
                eeg_data_path=eeg_data_path, eeg_channels=eeg_channels, filters_data=filters_data,
                suffix=suffix, project_root=project_root, task_info=os.path.basename(eeg_data_path),
                n_patterns=n_patterns
            )
        except RuntimeError:
            res = None
            for ses_name, task_name in RUNS:
                new_eeg_path = os.path.join(
                    project_root, config['root'], f'sub-{sub}', f'ses-{ses_name}', 'eeg',
                    f'sub-{sub}_ses-{ses_name}_task-{task_name}_eegMRbvCBbviiR250.vhdr'
                )
                try:
                    res = plot_topographies(
                        eeg_data_path=new_eeg_path, eeg_channels=eeg_channels, filters_data=filters_data,
                        suffix=suffix, project_root=project_root, task_info=os.path.basename(eeg_data_path),
                        n_patterns=n_patterns
                    )
                except RuntimeError:
                    pass
                if res:
                    break
            if res is None:
                raise RuntimeError(f'None of the runs in {RUNS} have metadata with info about EEG channel coordinates')
            warn(f'{eeg_data_path} does not have information on locations of EEG channels. {new_eeg_path} '
                 f'was used for metadata')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot topographies for BrainWaveLinker spatial filters.')

    parser.add_argument(
        '--subjects', type=str, nargs='+', help='IDs of subjects to use.',
        required=False, default=ALL_SUBJECTS
    )
    parser.add_argument(
        '--delay', type=int, help='Delay to use.', required=False, default=9
    )
    parser.add_argument(
        '--suffix', type=str, help='Experiment suffix.', required=False, default='SubcortTrend'
    )
    parser.add_argument(
        '--wandb-root', type=str, help='Path to the Weights and Biases logs and checkpoints.',
        required=False, default=os.path.join(PROJECT_ROOT, 'wandb_logs')
    )
    parser.add_argument(
        '--project-root', type=str, help='Path to the project root.', required=False, default=PROJECT_ROOT
    )
    parser.add_argument(
        '--project-name', type=str, help='Name of the project in Weights and Biases.', required=False,
        default='EEG-BOLD-Decoding'
    )
    parser.add_argument(
        '--n-patterns', type=int,
        help='Amount of strongest patterns to plot per task. Please use numbers divisible by 3.', required=False,
        default=6
    )

    return parser.parse_args()


if __name__ == '__main__':
    mne.set_log_level('WARNING')

    args = parse_arguments()

    for sub in args.subjects:
        plot_topographies_all_tasks(
            sub=sub, delay=args.delay, suffix=args.suffix, wandb_root=args.wandb_root,
            project_root=args.project_root, project_name=args.project_name, n_patterns=args.n_patterns
        )
