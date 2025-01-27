import logging
import os
import subprocess
import sys
import time
import json
import shutil
from itertools import product

import nvsmi

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level='INFO',
)

logger = logging.getLogger('kb-train')
logger.setLevel('INFO')

SEED = 42
# on every dataset we have different delay (depends on size takes to load it)
SLEEP_BY_DATASET = {
    'regular': 240,
    'mnli10k': 300,
    'mnli100k': 600,
}

BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'best-logs')

TRAIN_GRID = {
    'dataset_type': ['regular', 'mnli10k', 'mnli100k'],
    'num_epochs': [3, 4],
    'lr': [2e-5, 3e-5, 5e-5],
    'dataset_name': ['combined', 'hypernymy'], #'location_common', 'location_rare', 'trademark', 'color', 'hypernymy'],
}


def run_cmd(cmd, logfile):
    env = {'PYTHONWARNINGS': 'once', 'TMPDIR': './tmp'}
    env.update(os.environ)
    logger.info(f'running job {cmd!r}')
    subprocess.Popen(
        cmd,
        stdout=open(logfile, 'w'),
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setpgrp,
    )


def run_train_job(model_filename, logfile: str, overrides: dict):
    allennlp_cmd = shutil.which('allennlp')
    py_cmd = sys.executable
    cmd = [
        'nohup',
        py_cmd,
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
    run_cmd(cmd, logfile)


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
            'optimizer': {'lr': lr},
        },
        'train_data_path': train_data_fname,
        'validation_data_path': dev_data_fname,
    }


def get_grid():
    domains = [domain for param, domain in TRAIN_GRID.items()]
    for values in product(*domains):
        yield dict(zip(TRAIN_GRID.keys(), values))


def get_grid_best_only():
    grid = [
        ['combined', 'mnli10k',	4, 2e-5],
        ['location_common', 'mnli10k', 4, 3e-5],
        ['hypernymy', 'mnli100k', 3, 5e-5],
        ['color', 'mnli100k', 4, 3e-5],
        ['trademark', 'mnli10k', 4, 2e-5],
    ]

    keys = [
        'dataset_name',
        'dataset_type',
        'num_epochs',
        'lr',
    ]
    return [dict(zip(keys, x)) for x in grid]


def get_free_cuda_device():
    while True:
        gpus = list(nvsmi.get_gpus())
        logger.info(f'checking free gpu from {len(gpus)} devices')
        for gpu in gpus:
            if gpu.mem_free > 11000:
                logger.info(f'found free gpu at {gpu}')
                return int(gpu.id)

        logger.info(f'all gpus are busy... sleeping 1 minute...')
        time.sleep(60)


def get_filenames(params: dict):
    model_dirname = os.path.join(
        MODELS_DIR, params['dataset_name'], params['dataset_type']
    )
    fname = f'kbmodel__{params["dataset_name"]}__{params["dataset_type"]}__{params["num_epochs"]}__{params["lr"]}'
    os.makedirs(model_dirname, exist_ok=True)

    log_dirname = os.path.join(
        LOGS_DIR, params['dataset_name'], params['dataset_type']
    )
    os.makedirs(log_dirname, exist_ok=True)

    model_filename = os.path.join(model_dirname, fname)
    log_filename = os.path.join(log_dirname, fname + '.log')
    return model_filename, log_filename


def get_dataset_filenames(dataset_name, dataset_type):
    data_dir = os.path.join(DATASETS_DIR, dataset_name.replace('_rare', '').replace('_common', ''))

    if dataset_type == 'regular':
        train_data_fname = os.path.join(data_dir, f'{dataset_name}_train.json')
    else:
        train_data_fname = os.path.join(data_dir, f'{dataset_name}_train_{dataset_type}.json.xz')

    dev_data_fname = os.path.join(data_dir, f'{dataset_name}_dev.json')

    assert os.path.exists(train_data_fname)
    assert os.path.exists(dev_data_fname)

    return train_data_fname, dev_data_fname


def main():
    hp_grid = list(get_grid())

    logger.info(f'got {len(hp_grid)} setups in the hyper params grid')
    for params in hp_grid:
        logger.info(f'trying to launch {params}')

        train_data_fname, dev_data_fname = get_dataset_filenames(params['dataset_name'], params['dataset_type'])

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
        model_filename, log_filename = get_filenames(params)

        if os.path.exists(log_filename):
            logger.info(f'skipping already exist job - {model_filename}')
            continue

        run_train_job(model_filename, log_filename, overrides_dict)

        # sleep so we dont have race conditions on get cuda
        sleep_time = SLEEP_BY_DATASET[params['dataset_type']]
        logger.info(f'sleeping {sleep_time} secs before next schedule')
        time.sleep(sleep_time)


if __name__ == '__main__':
    main()
