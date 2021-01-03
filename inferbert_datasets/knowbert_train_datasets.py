import logging
import os
import subprocess
import sys
import time
import json
import shutil
from itertools import product

import torch

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level='INFO',
)

logger = logging.getLogger('kb-train')
logger.setLevel('INFO')

SEED = 42

BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

TRAIN_GRID = {
    'dataset': ['location', 'color', 'trademark'],
    'dataset_type': ['dataset_creative', 'mnli_10k_split', 'mnli_full'],
    'num_epochs': [3, 4],
    'lr': [2e-5, 3e-5, 5e-5],
}


def run_train_job(model_filename, logfile: str, overrides: dict):
    allennlp_cmd = shutil.which('allennlp')
    cmd = [
        'nohup',
        allennlp_cmd,
        'train',
        '--file-friendly-logging',
        '--include-package',
        'kb.include_all',
        '--overrides',
        json.dumps(overrides),
        'base-config.jsonnet',
        '-s',
        model_filename,
        '--force',
    ]
    logger.info(f'running job {cmd!r}')
    subprocess.Popen(
        cmd,
        stdout=open(logfile, 'w'),
        stderr=subprocess.STDOUT,
        env={'PYTHONWARNINGS': 'once'},
    )


def get_overrides(
    train_data_fname, dev_data_fname, cuda_device, num_epochs, lr, seed=SEED
):
    return {
        'random_seed': seed,
        'numpy_seed': seed,
        'pytorch_seed': seed,
        'trainer': {
            'cuda_device': cuda_device,
            'num_epochs': num_epochs,
            'lr': lr,
        },
        'train_data_path': train_data_fname,
        'validation_data_path': dev_data_fname,
    }


def get_grid():
    domains = [domain for param, domain in TRAIN_GRID.items()]
    for values in product(*domains):
        yield dict(zip(TRAIN_GRID.keys(), values))


def get_free_cuda_device():
    num_gpus = torch.cuda.device_count()

    while True:
        logger.info(f'checking free gpu from {num_gpus} devices')
        for device_id in range(num_gpus):
            if torch.cuda.memory_allocated(device_id) == 0:
                logger.info(f'found free gpu at {device_id}')
                return device_id

        logger.info(f'all gpus are busy... sleeping 1 minute...')
        time.sleep(60)


def main():
    hp_grid = list(get_grid())

    logger.info(f'got {len(hp_grid)} setups in the hyper params grid')
    for params in hp_grid:
        logger.info(f'trying to launch {params}')
        data_dir = os.path.join(
            DATASETS_DIR, params['dataset'], params['dataset_type']
        )

        # compressed dataset
        train_data_fname = os.path.join(
            data_dir, 'dataset_creative_train.json'
        )
        if not os.path.exists(train_data_fname):
            train_data_fname = train_data_fname + '.xz'

        dev_data_fname = os.path.join(data_dir, 'dataset_creative_dev.json')

        # this blocks until a gpu is free
        cuda_device = get_free_cuda_device()

        # prepare config
        overrides_dict = get_overrides(
            train_data_fname=train_data_fname,
            dev_data_fname=dev_data_fname,
            cuda_device=cuda_device,
            num_epochs=params['num_epochs'],
            lr=params['lr'],
        )

        # launch our model
        model_dirname = os.path.join(
            MODELS_DIR, params['dataset'], params['dataset_type']
        )
        fname = f'kbmodel__{params["dataset"]}__{params["dataset_type"]}__{params["num_epochs"]}__{params["lr"]}'
        os.makedirs(model_dirname, exist_ok=True)

        log_dirname = os.path.join(
            LOGS_DIR, params['dataset'], params['dataset_type']
        )
        os.makedirs(log_dirname, exist_ok=True)

        model_filename = os.path.join(model_dirname, fname)
        log_filename = os.path.join(log_dirname, fname + '.log')

        if os.path.exists(log_filename) or os.path.exists(model_filename):
            logger.info(f'skipping already exist job - {model_filename}')
            continue

        run_train_job(model_filename, log_filename, overrides_dict)

        # sleep so we dont have race conditions on get cuda
        logger.info('sleeping 30 secs before next schedule')
        time.sleep(30)


if __name__ == '__main__':
    main()
