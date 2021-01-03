import logging
import sys

import cbox

from datasets_config import Dataset

import random
from itertools import combinations
from operator import itemgetter

import pandas as pd
import os

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level='INFO',
)

logger = logging.getLogger('inferbert')
logger.setLevel('INFO')


def load_raw_data(input_file, dataset: Dataset):
    df = pd.read_json(input_file)

    for field in dataset.metadata_fields:
        df[field] = df['metadata'].apply(itemgetter(field))

    original_len = len(df)
    dups_df = (
        df[['premise', 'hypothesis', 'label']]
        .groupby(['premise', 'hypothesis'])
        .nunique()
    )
    dups_and_multi_label = set(dups_df[dups_df['label'] > 1].index.tolist())

    # same premise hypothesis pair tagged twice with different labels
    df['has_multiple_labels'] = df.apply(
        lambda x: (x.premise, x.hypothesis) in dups_and_multi_label, axis=1
    )

    df = df[~df['has_multiple_labels']]
    df = df.drop_duplicates(['premise', 'hypothesis'])

    logger.info(
        f'dedup: found {original_len}/{len(df)} unique/total samples on '
        f'{input_file}. filtered {len(dups_and_multi_label)} samples with '
        f'conflicting labels for same pair'
    )
    return df


def split_data(df, dataset: Dataset, seed='inferbert'):
    df = df.copy()

    # we split by original row ids to avoid info leakage between train / test
    row_ids = sorted(set(df['row_id']))
    random.seed(seed)
    random.shuffle(row_ids)

    # split to train / test
    train_split_pos = int(len(row_ids) * dataset.train_size)
    dev_split_pos = train_split_pos + int(len(row_ids) * dataset.dev_size)

    train_row_ids = set(row_ids[:train_split_pos])
    dev_row_ids = set(row_ids[train_split_pos:dev_split_pos])
    test_row_ids = set(row_ids[dev_split_pos:])

    assert train_row_ids & test_row_ids == set()
    assert train_row_ids & dev_row_ids == set()
    assert dev_row_ids & test_row_ids == set()

    df['is_train'] = df['row_id'].isin(train_row_ids)
    df['is_dev'] = df['row_id'].isin(dev_row_ids)
    df['is_test'] = df['row_id'].isin(test_row_ids)

    assert (
        df[['is_train', 'is_dev', 'is_test']].values.sum(axis=1) == 1
    ).all(), 'more than one value'
    return df


def add_data_filters(df, dataset: Dataset):
    df = df.copy()

    # filter excluded workers
    assert dataset.excluded_workers < set(
        df['worker_id']
    ), 'invalid excluded workers'
    df = df[~df['worker_id'].isin(dataset.excluded_workers)]
    df['dataset_creative'] = True
    return df


def save_dataset(df, ds_dir, ds_name, dataset: Dataset):
    os.makedirs(ds_dir, exist_ok=True)

    fnames = []
    for set_name, select in [
        ('train', df['is_train']),
        ('dev', df['is_dev']),
        ('test', df['is_test']),
    ]:
        ds_fname = os.path.join(ds_dir, f'{ds_name}_{set_name}.json')
        fnames.append(ds_fname)
        set_df = df[select]
        logger.info(
            f'creating {set_name} set with {len(set_df)} samples from '
            f'{set_df["row_id"].nunique()} hits into {ds_fname}'
        )
        set_df[list(dataset.columns)].to_json(
            ds_fname, orient='records', indent=4
        )

    # paranoid test
    dataset_dfs = [pd.read_json(f) for f in fnames]
    for ds1, ds2 in combinations(dataset_dfs, 2):
        assert set(ds1['hypothesis']) & set(ds2['hypothesis']) == set()
        assert set(ds1['premise']) & set(ds2['premise']) == set()


def prepare_dataset(dataset: Dataset, output_dir: str, input_file=None):
    logger.info(f'preparing dataset {dataset.name}')

    input_file = input_file or dataset.input_file

    df = load_raw_data(input_file, dataset)
    df = split_data(df, dataset)
    df = add_data_filters(df, dataset)

    for dataset_name in dataset.field_types:
        dataset_df = df[df[dataset_name]].copy()
        ds_dir = os.path.join(output_dir, dataset_name)
        save_dataset(dataset_df, ds_dir, dataset_name, dataset)

    logger.info(f'finished prepare dataset {dataset.name}')


@cbox.cmd
def prepare_all_datasets(output_dir: str):
    logger.info(f'preparing all datasets into {output_dir}')
    for dataset in Dataset.get_all_datasets():
        dataset_dir = os.path.join(output_dir, dataset.name)
        prepare_dataset(dataset, dataset_dir)


if __name__ == '__main__':
    cbox.main(prepare_all_datasets)
