import os
import random

import pandas as pd


def main(fname, outdir, dataset_name, train_split, dev_split):
    df = pd.read_json(fname)
    for col in df.loc[0]['metadata'].keys():
        df[col] = df['metadata'].apply(lambda x: x[col])
    df.pop('metadata')

    row_ids = sorted(set(df['row_id']))
    random.seed('inferbert')
    random.shuffle(row_ids)

    # split to train / dev / test
    train_split_pos = int(len(row_ids) * train_split)
    dev_split_pos = int(len(row_ids) * (train_split + dev_split))

    train_row_ids = set(row_ids[:train_split_pos])
    dev_row_ids = set(row_ids[train_split_pos:dev_split_pos])
    test_row_ids = set(row_ids[dev_split_pos:])

    assert train_row_ids & test_row_ids == set()
    assert train_row_ids & dev_row_ids == set()
    assert dev_row_ids & test_row_ids == set()

    assert train_row_ids | dev_row_ids | test_row_ids == set(df['row_id'])

    train_df = df[df['row_id'].isin(train_row_ids)]
    dev_df = df[df['row_id'].isin(dev_row_ids)]
    test_df = df[df['row_id'].isin(test_row_ids)]

    os.makedirs(outdir, exist_ok=True)
    for dtype, ds in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        outfname = f'{outdir}/{dataset_name}_{dtype}.json'
        print(f'saving {dtype} to {outfname} with {len(ds)} lines')
        ds.to_json(outfname, indent=4, orient='records')


if __name__ == '__main__':
    pass
    # main(
    #     'amt-raw-hits/trademark_dataset/trademark_examples_200p.json',
    #     outdir='datasets/trademark',
    #     dataset_name='trademark',
    #     train_split=0.6975,
    #     dev_split=0.095,  # 252 test 114 dev
    # )

    # main(
    #     'amt-raw-hits/hypernymy_dataset/all_hypernymy_examples_229p.json',
    #     outdir='datasets/hypernymy',
    #     dataset_name='hypernymy',
    #     train_split=0.699,
    #     dev_split=0.083,  # 50 test 19 dev
    # )

    # main(
    #     'amt-raw-hits/color_dataset/color_examples_old_78p_and_new_133p.json',
    #     outdir='datasets/color',
    #     dataset_name='color',
    #     train_split=0.677,
    #     dev_split=0.08,
    # )
