import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

MNLI_FULL_DATASET = os.path.join(DATASETS_DIR, 'mnli', 'mnli_train_full.jsonl.xz')

MNLI_TRAIN_DATASETS = {
    'mnli10k': os.path.join(DATASETS_DIR, 'mnli', 'mnli_train_10k_split.json.xz'),
    'mnli100k': os.path.join(DATASETS_DIR, 'mnli', 'mnli_train_100k.json.xz'),
}

RENAME_COLS = {
    'sentence1': 'premise',
    'sentence2': 'hypothesis',
    'gold_label': 'label',
}


def sample_mnli(size, outfile, columns=('sentence1', 'sentence2', 'gold_label')):
    df = pd.read_json(MNLI_FULL_DATASET, lines=True)
    df = df[df['gold_label'] != '-'][list(columns)]
    print(set(df['gold_label']))

    sample_df = df.sample(n=size, random_state=42)
    sample_df.to_json(outfile, orient='records', indent=4)


def main(fname, out_fname_tmpl):
    for mnli_name, mnli_filename in MNLI_TRAIN_DATASETS.items():
        df = pd.read_json(fname)[list(RENAME_COLS.values())]
        print(f'read {len(df)} from {fname}')

        print(mnli_filename)
        mnli_df = pd.read_json(mnli_filename)
        mnli_df.rename(columns=RENAME_COLS, inplace=True)
        print(f'read {len(mnli_df)} from {mnli_filename}')

        final_df = pd.concat((df, mnli_df))

        print('concat length:', len(final_df))
        final_df.to_json(out_fname_tmpl.format(mnli_name), orient='records')


if __name__ == '__main__':
    pass
    # sample_mnli(
    #     10_000,
    #     outfile=MNLI_TRAIN_DATASETS['mnli10k'],
    # )
    # sample_mnli(
    #     100_000,
    #     outfile=MNLI_TRAIN_DATASETS['mnli100k'],
    # )
    # main(
    #     fname=os.path.join(DATASETS_DIR, 'location', 'location_rare_train.json'),
    #     out_fname_tmpl=os.path.join(DATASETS_DIR, 'location', 'location_rare_train_{}.json.xz'),
    # )
    main(
        fname=os.path.join(DATASETS_DIR, 'location', 'location_common_train.json'),
        out_fname_tmpl=os.path.join(DATASETS_DIR, 'location', 'location_common_train_{}.json.xz'),
    )
    # main(
    #     fname=os.path.join(DATASETS_DIR, 'trademark', 'trademark_train.json'),
    #     out_fname_tmpl=os.path.join(DATASETS_DIR, 'trademark', 'trademark_train_{}.json.xz'),
    # )
    # main(
    #     fname=os.path.join(DATASETS_DIR, 'hypernymy', 'hypernymy_train.json'),
    #     out_fname_tmpl=os.path.join(DATASETS_DIR, 'hypernymy', 'hypernymy_train_{}.json.xz'),
    # )
    # main(
    #     fname=os.path.join(DATASETS_DIR, 'color', 'color_train.json'),
    #     out_fname_tmpl=os.path.join(DATASETS_DIR, 'color', 'color_train_{}.json.xz'),
    # )
    main(
        fname=os.path.join(DATASETS_DIR, 'combined', 'combined_train.json'),
        out_fname_tmpl=os.path.join(DATASETS_DIR, 'combined', 'combined_train_{}.json.xz'),
    )
