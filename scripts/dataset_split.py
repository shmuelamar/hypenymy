import random
from itertools import combinations
from operator import itemgetter

import cbox
import pandas as pd
import os

EXCLUDED_WORKERS = frozenset({'A2BK45LZGGWPLX', 'A3V916K8KYOTPS'})
METADATA_FIELDS = (
    'row_id',
    'worker_id',
    'is_complete',
    'hit_id',
    'section',
    'ptype',
    'htype',
    'pword',
    'hword',
)
DATASET_COLS = ['premise', 'hypothesis', 'label', *METADATA_FIELDS]

DATASETS_TYPES = ('dataset_simple', 'dataset_creative', 'dataset_full')


def load_raw_data(input_file):
    df = pd.read_json(input_file)

    for field in METADATA_FIELDS:
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

    print(
        f'dedup: found {original_len}/{len(df)} unique/total samples on '
        f'{input_file}. filtered {len(dups_and_multi_label)} samples with '
        f'conflicting labels for same pair'
    )
    return df


def split_data(df, train_size):
    df = df.copy()

    # we split by original row ids to avoid info leakage between train / test
    row_ids = sorted(set(df['row_id']))
    random.seed('hypernymy')
    random.shuffle(row_ids)

    # split to train / test
    split_pos = int(len(row_ids) * train_size)
    train_row_ids = set(row_ids[:split_pos])
    test_row_ids = set(row_ids[split_pos:])

    assert train_row_ids & test_row_ids == set()

    df['is_train'] = df['row_id'].isin(train_row_ids)
    return df


def add_data_filters(df):
    df = df.copy()

    # filter excluded workers
    assert EXCLUDED_WORKERS < set(df['worker_id']), 'invalid excluded workers'
    df = df[~df['worker_id'].isin(EXCLUDED_WORKERS)]

    # we have 3 dataset types
    df['dataset_simple'] = df['section'].str.match(r'^[1-3]')
    df['dataset_creative'] = ~df['dataset_simple']
    df['dataset_full'] = True
    return df


def save_dataset(df, ds_dir, ds_name):
    os.makedirs(ds_dir, exist_ok=True)

    fnames = []
    for set_name, select in [
        ('train', df['is_train']),
        ('test', ~df['is_train']),
    ]:
        ds_fname = os.path.join(ds_dir, f'{ds_name}_{set_name}.json')
        fnames.append(ds_fname)
        set_df = df[select]
        print(
            f'creating {set_name} set with {len(set_df)} samples from '
            f'{set_df["row_id"].nunique()} hits into {ds_fname}'
        )
        set_df[DATASET_COLS].to_json(ds_fname, orient='records', indent=4)

    # paranoid test
    dataset_dfs = [pd.read_json(f) for f in fnames]
    for ds1, ds2 in combinations(dataset_dfs, 2):
        assert set(ds1['hypothesis']) & set(ds2['hypothesis']) == set()
        assert set(ds1['premise']) & set(ds2['premise']) == set()


@cbox.cmd
def main(input_file, output_dir, train_size: float = 0.75):
    df = load_raw_data(input_file)
    df = split_data(df, train_size)
    df = add_data_filters(df)

    for dataset_name in DATASETS_TYPES:
        dataset_df = df[df[dataset_name]].copy()
        ds_dir = os.path.join(output_dir, dataset_name)
        save_dataset(dataset_df, ds_dir, dataset_name)


if __name__ == '__main__':
    cbox.main(main)
