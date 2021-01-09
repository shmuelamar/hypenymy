import os
import time
from os import pardir

from eval_knowbert_on_datasets import run_eval_job, SLEEP_BY_DATASET
from knowbert_train_datasets import get_free_cuda_device, logger

ALL_EVAL_DATASETS = {
    'mnli_dev_matched': '../../mnli_1.0/dev_matched.tsv.gz',
    'locationv2_common': 'datasets/locationv2/dataset_creative/dataset_creative_test_common.json',
    'locationv2_rare': 'datasets/locationv2/dataset_creative/dataset_creative_test_rare.json',
    'trademark': 'datasets/trademark/dataset_creative/dataset_creative_test.json',
    'color': 'datasets/color/dataset_creative/dataset_creative_test.json',
}


def main():
    model_tar_filename = os.path.join(
        pardir, pardir, 'mnli_final_model', 'model.tar.gz'
    )

    os.makedirs('logs/mnli', exist_ok=True)
    for dataset_name, dataset_filename in ALL_EVAL_DATASETS.items():
        logger.info(f'predict mnli on {dataset_name}')
        gpu = get_free_cuda_device()

        run_eval_job(
            model_tar_filename,
            cuda_device=gpu,
            dataset_filename=dataset_filename,
            logfile=f'logs/mnli/kbmodel__mnli__{dataset_name}.log',
        )
        sleep_time = SLEEP_BY_DATASET.get(dataset_name, SLEEP_BY_DATASET[None])
        logger.info(f'sleeping {sleep_time} seconds')
        time.sleep(sleep_time)


if __name__ == '__main__':
    main()
