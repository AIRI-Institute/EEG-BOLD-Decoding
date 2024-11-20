import argparse
import copy
import os
import traceback

import mne

from bw_linker.brain_wave_pipeline.training_pipeline import run_full_pipeline
from bw_linker.utils.constants import ALL_SUBJECTS, PROJECT_ROOT
from bw_linker.utils.helpers import load_json


def run_experiment():
    """
    Train EEG-based fMRI Digital Twin model
    """
    mne.set_log_level('WARNING')

    args = parse_arguments()
    json_config_root = args.config_root
    subjects = args.subjects
    delays = args.delays
    for subject in subjects:
        for delay in delays:
            try:
                json_config_path = os.path.join(json_config_root, f'delay-{delay}', f'sub-{subject}-train.json')
                print(f'Processing {json_config_path} config!')
                full_config = load_json(json_path=json_config_path)

                iter_config = copy.deepcopy(full_config)

                config = iter_config['config']

                model_config = iter_config['model_config']

                sampler_configs = iter_config['sampler_configs']

                dataloader_configs = iter_config['dataloader_configs']

                criterion_config = iter_config['criterion_config']

                optimizer_config = iter_config['optimizer_config']

                scheduler_config = iter_config['scheduler_config']

                trainer_config = iter_config['trainer_config']

                wandb_kwargs = iter_config['wandb_kwargs']

                run_full_pipeline(
                    config=config, criterion_config=criterion_config, optimizer_config=optimizer_config,
                    scheduler_config=scheduler_config, trainer_config=trainer_config,
                    sampler_configs=sampler_configs,
                    dataloader_configs=dataloader_configs,
                    wandb_kwargs=wandb_kwargs, project_root=PROJECT_ROOT,
                    model_config=model_config
                )
            except Exception:
                traceback.print_exc()


def parse_arguments():
    parser = argparse.ArgumentParser(description='EEG based fMRI Digital Twin')

    parser.add_argument(
        '--config-root', type=str,
        help='Path to a directory with config json files',
        required=True
    )
    parser.add_argument(
        '--subjects', type=str, nargs='+', help='IDs of subjects to use.',
        required=False, default=ALL_SUBJECTS
    )
    parser.add_argument(
        '--delays', type=int, nargs='+', help='Delays to use.',
        required=False, default=list(range(0, 16))
    )

    return parser.parse_args()


if __name__ == '__main__':
    run_experiment()
