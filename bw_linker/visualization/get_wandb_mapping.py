import argparse
import os
from pathlib import Path

import wandb

from bw_linker.utils.constants import ALL_SUBJECTS, PROJECT_ROOT
from bw_linker.utils.helpers import run_name_from_parameters, load_json, save_json


def get_wandb_paths(delays: list[int], subjects: list[str], suffixes: list[str], entity: str,
                    project_name: str = 'EEG-BOLD-Decoding'):
    """
    Returns dictionary with run name and run id correspondence

    Parameters
    ----------
    delays : list[int]
        A list of delays to include
    subjects : list[str]
        A list of subjects to include
    suffixes : list[str]
        A list of suffixes which to include (e.g. 'Subcort', 'SubcortTrend', ...)
    entity : str
        A name of the Weights and Biases entity
    project_name : str
        A name of the Weights and Biases project. Default: 'EEG-BOLD-Decoding'

    Returns
    -------
    matched_runs : dict
        A dictionary with runs information matched as {run_suffix: {run_delay: {run_sub: run_id}}} structure
    """
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project_name}')
    allowed_runs = {}
    matched_runs = {}
    for suffix in suffixes:
        matched_runs[suffix] = {}
        for delay in delays:
            matched_runs[suffix][delay] = {}
            for sub in subjects:
                matched_runs[suffix][delay][sub] = None
                allowed_runs[run_name_from_parameters(delay=delay, sub=sub, suffix=suffix)] = (delay, sub, suffix)

    for run in runs:
        run_name = run.name
        if run_name in allowed_runs:
            delay, sub, suffix = allowed_runs[run_name]
            matched_runs[suffix][delay][sub] = run.id
    return matched_runs


def load_wandb_mapping():
    """
    Loads and returns a flat dictionary with experiment parameters mapped to its Weights and Biases run identifiers

    Returns
    -------
    matched_runs_flat : dict
        A dictionary with runs information matched as {(run_delay, run_sub, run_suffix): run_id} key-value pairs
    """
    matched_runs = load_json(os.path.join(PROJECT_ROOT, 'visualizations', 'wandb_mapping.json'))
    matched_runs_flat = {}
    for suffix, suffix_runs in matched_runs.items():
        for delay, delay_runs in suffix_runs.items():
            assert delay.isdigit(), delay
            delay = int(delay)
            for sub, run_id in delay_runs.items():
                matched_runs_flat[(delay, sub, suffix)] = run_id
    return matched_runs_flat


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
        '--suffixes', type=str, help='Experiment suffixes to use.', required=False,
        default=('SubcortTrend', 'Subcort', 'CortTrend', 'Cort')
    )
    parser.add_argument(
        '-e', '--entity', type=str, help='Name of the entity in the Weights and Biases.',
        required=True
    )
    parser.add_argument(
        '--project-name', type=str, help='Name of the project in Weights and Biases.', required=False,
        default='EEG-BOLD-Decoding'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    matched_runs = get_wandb_paths(
        delays=args.delays, subjects=args.subjects, suffixes=args.suffixes, entity=args.entity,
        project_name=args.project_name
    )
    save_dir = os.path.join(PROJECT_ROOT, 'visualizations')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_json(save_path=os.path.join(save_dir, 'wandb_mapping.json'), data=matched_runs)
