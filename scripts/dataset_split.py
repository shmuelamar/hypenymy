import random
from itertools import combinations

import cbox
import pandas as pd
import os

DATASET_COLS = ['premise', 'hypothesis', 'label']


@cbox.cmd
def main(input_file, output_dir, train_size: float = 0.75):
    df = pd.read_json(input_file)
    df['row_id'] = df['metadata'].apply(lambda x: x['row_id'])

    # we split by original row ids to avoid leakage
    row_ids = sorted(set(df['row_id']))
    random.seed('hypernymy')
    random.shuffle(row_ids)

    # split to train / test
    split_pos = int(len(row_ids) * train_size)
    train_row_ids = set(row_ids[:split_pos])
    test_row_ids = set(row_ids[split_pos:])

    assert train_row_ids & test_row_ids == set()

    df['is_train'] = df['row_id'].apply(lambda rid: rid in train_row_ids)

    os.makedirs(output_dir, exist_ok=True)

    fnames = []
    for name, select in [('train', df['is_train']), ('test', ~df['is_train'])]:
        ds_fname = os.path.join(output_dir, f'{name}.json')
        fnames.append(ds_fname)
        set_df = df[select]
        print(f'creating {name} set with {len(set_df)} samples from {set_df["row_id"].nunique()} hits into {ds_fname}')
        set_df[DATASET_COLS].to_json(ds_fname, orient='records', indent=4)

    # paranoid test
    dataset_dfs = [pd.read_json(f) for f in fnames]
    for ds1, ds2 in combinations(dataset_dfs, 2):
        assert set(ds1['hypothesis']) & set(ds2['hypothesis']) == set()
        assert set(ds1['premise']) & set(ds2['premise']) == set()


if __name__ == '__main__':
    cbox.main(main)
