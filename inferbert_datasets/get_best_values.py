import pandas as pd


def main(fname):
    key = ['dataset_name', 'trainset_name']
    df = pd.read_csv(fname)

    for key, grp_df in df.groupby(key):
        print(key)
        grp_df[grp_df['evalset_name'] == 'mnli_dev_matched']
        print(grp_df)


if __name__ == '__main__':
    main('./knowbert_eval_results.csv')
