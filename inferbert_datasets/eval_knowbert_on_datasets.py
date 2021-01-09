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

SLEEP_BY_DATASET = {'mnli_dev_matched': 150, None: 60}  # others

EVAL_DATASETS = {
    # FIXME: copy into this repo
    'mnli_dev_matched': '../../mnli_1.0/dev_matched.tsv.gz',
    'creative_test': 'dataset_creative/dataset_creative_test.json',
    'common_test': 'dataset_creative/dataset_creative_test_common.json',
    'rare_test': 'dataset_creative/dataset_creative_test_rare.json',
    # 'creative_dev': 'dataset_creative/dataset_creative_dev.json',
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
        for dataset_name, dataset_filename in EVAL_DATASETS.items():
            # FIXME: remove after move to project
            if dataset_name != 'mnli_dev_matched':
                dataset_filename = os.path.join(
                    DATASETS_DIR, params['dataset'], dataset_filename
                )

            if not os.path.exists(dataset_filename):
                logger.info(f'skipping {dataset_filename} - not exist')
                continue

            model_filename, log_filename = get_filenames(params)
            if not is_finished(log_filename):
                logger.info(f'skipping {model_filename} - not finished')

            model_tar_filename = os.path.join(model_filename, 'model.tar.gz')
            eval_log_filename = (
                log_filename[: -len('.log')] + f'__eval__{dataset_name}.log'
            )

            if os.path.exists(eval_log_filename):
                logger.info(
                    f'skipping eval for {dataset_name}.'
                    f'file exists {eval_log_filename}'
                )
                continue

            cuda_device = get_free_cuda_device()

            logger.info(f'eval on {dataset_filename} for {eval_log_filename}')
            run_eval_job(
                model_tar_filename,
                cuda_device,
                dataset_filename,
                eval_log_filename,
            )

            sleep_time = SLEEP_BY_DATASET.get(
                dataset_name, SLEEP_BY_DATASET[None]
            )
            logger.info(f'sleeping {sleep_time} seconds')
            time.sleep(sleep_time)


if __name__ == '__main__':
    main()