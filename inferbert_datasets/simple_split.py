import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split


def main(fname, outdir, dataset_name, train_split, dev_split):
    df = pd.read_json(fname)
    for col in df.loc[0]['metadata'].keys():
        df[col] = df['metadata'].apply(lambda x: x[col])
    df.pop('metadata')

    # # we split by original row ids to avoid info leakage between train / test
    # train_row_ids, test_row_ids = train_test_split(df['row_id'].tolist(), stratify=df['item'], test_size=0.1, random_state=5)
    # train_row_ids = set(train_row_ids)
    # test_row_ids = set(test_row_ids)
    # print(len(train_row_ids), len(test_row_ids))
    #
    # train_row_ids = sorted(train_row_ids)
    # strat = df[df['row_id'].isin(train_row_ids)][['row_id', 'item']].drop_duplicates('row_id')
    #
    # train_row_ids, dev_row_ids = train_test_split(list(train_row_ids), stratify=strat, test_size=0.1, random_state=42)
    # train_row_ids = set(train_row_ids)
    # dev_row_ids = set(dev_row_ids)

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
    # main(
    #     'amt-raw-hits/color_dataset/color_examples_all.json',
    #     outdir='datasets/color',
    #     dataset_name='color',
    #     word_field='item',
    # )

    # main(
    #     'amt-raw-hits/trademark_dataset/trademark_examples.json',
    #     outdir='datasets/trademark',
    #     dataset_name='trademark',
    #     train_split=0.6975,
    #     dev_split=0.095,  # 252 test 114 dev
    #     # word_field='company',
    # )

    main(
        'amt-raw-hits/color_dataset/color_examples_old_78p_and_new_133p.json',
        outdir='datasets/color',
        dataset_name='color',
        train_split=0.677,
        dev_split=0.08,
    )
