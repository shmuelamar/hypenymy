import os
import shutil
import sys
import time

from knowbert_train_datasets import (
    get_grid,
    get_filenames,
    logger,
    run_cmd,
    get_free_cuda_device,
    DATASETS_DIR,
)

SLEEP = 180

EVAL_DATASETS = {
    # mnli
    'mnli_dev_matched': os.path.join(DATASETS_DIR, 'mnli', 'mnli_dev_matched.jsonl.xz'),
    'mnli_dev_mismatched': os.path.join(DATASETS_DIR, 'mnli', 'mnli_dev_mismatched.jsonl.xz'),
    # location
    'location_rare_dev': os.path.join(DATASETS_DIR, 'location', 'location_rare_dev.json'),
    'location_rare_test': os.path.join(DATASETS_DIR, 'location', 'location_rare_test.json'),
    'location_common_test': os.path.join(DATASETS_DIR, 'location', 'location_common_test.json'),
    # trademark
    'trademark_dev': os.path.join(DATASETS_DIR, 'trademark', 'trademark_dev.json'),
    'trademark_test': os.path.join(DATASETS_DIR, 'trademark', 'trademark_test.json'),
    # color
    'color_dev': os.path.join(DATASETS_DIR, 'color', 'color_dev.json'),
    'color_test': os.path.join(DATASETS_DIR, 'color', 'color_test.json'),
    # hypernymy
    'hypernymy_dev': os.path.join(DATASETS_DIR, 'hypernymy', 'hypernymy_dev.json'),
    'hypernymy_test': os.path.join(DATASETS_DIR, 'hypernymy', 'hypernymy_test.json'),
}

DATASET_NAME_TO_EVAL = {
    'location_rare': (
        'location_rare_dev',
        'location_rare_test',
        'location_common_test',
        'mnli_dev_matched',
        'mnli_dev_mismatched',
    ),
    'trademark': (
        'trademark_dev',
        'trademark_test',
        'mnli_dev_matched',
        'mnli_dev_mismatched',
    ),
    'color': {
        'color_dev',
        'color_test',
        'mnli_dev_matched',
        'mnli_dev_mismatched',
    },
    'hypernymy': {
        'hypernymy_dev',
        'hypernymy_test',
        'mnli_dev_matched',
        'mnli_dev_mismatched',
    },
}


def is_finished(log_filename):
    if not os.path.exists(log_filename):
        return False

    # check if this phrase exist - it means it finished
    with open(log_filename) as fp:
        for line in fp:
            if '"validation_accuracy":' in line:
                return True
    return False


def run_eval_job(model_tar_filename, cuda_device, dataset_filename, logfile):
    allennlp_cmd = shutil.which('allennlp')
    py_cmd = sys.executable
    cmd = [
        'nohup',
        py_cmd,
        allennlp_cmd,
        'evaluate',
        '--include-package',
        'kb.include_all',
        '--cuda-device',
        str(cuda_device),
        model_tar_filename,
        dataset_filename,
    ]
    run_cmd(cmd, logfile)


def main():
    for params in get_grid():
        for eval_dataset_name in DATASET_NAME_TO_EVAL[params['dataset_name']]:
            eval_filename = EVAL_DATASETS[eval_dataset_name]
            assert os.path.exists(eval_filename), f'not exist {eval_filename}'

            model_filename, log_filename = get_filenames(params)
            if not is_finished(log_filename):
                logger.info(f'skipping {model_filename} - not finished')

            model_tar_filename = os.path.join(model_filename, 'model.tar.gz')
            eval_log_filename = (
                log_filename[: -len('.log')] + f'__eval__{eval_dataset_name}.log'
            )

            if os.path.exists(eval_log_filename):
                logger.info(
                    f'skipping eval for {eval_dataset_name}.'
                    f'file exists {eval_log_filename}'
                )
                continue

            cuda_device = get_free_cuda_device()

            logger.info(f'eval on {eval_filename} for {eval_log_filename}')
            run_eval_job(
                model_tar_filename,
                cuda_device,
                eval_filename,
                eval_log_filename,
            )

            logger.info(f'sleeping {SLEEP} seconds')
            time.sleep(SLEEP)


if __name__ == '__main__':
    main()
