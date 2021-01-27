from functools import partial
from glob import glob

from pytorch_pretrained_bert.tokenization import BertTokenizer
import pandas as pd
from tqdm import tqdm

import wordnet_parsing_utils as wn


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# FIXME: @ohad - what is the default kwargs for these functions
DATA2FIND_PAIRS_FN = {
    'hypernymy': partial(wn.find_hypernymy_pairs, filter_repeat_word=True),
    'color': partial(wn.find_color_pairs, local_wiki_features=wn.local_wiki_features),
    'location': partial(wn.find_location_country_pairs, local_wiki=wn.local_wiki, include_ORG=True),
    'trademark': partial(wn.find_trademark_country_pairs, local_wiki=wn.local_wiki),
}


def parse_mnli_sample(a, b, data_type):
    tokens_a = tokenizer.tokenize(a)
    tokens_b = tokenizer.tokenize(b)
    tokens_a_new = wn.word_peice_connected(tokens_a, [1] * len(tokens_a))[0]
    tokens_b_new = wn.word_peice_connected(tokens_b, [1] * len(tokens_b))[0]

    pairs_fn = DATA2FIND_PAIRS_FN[data_type]

    pairs = pairs_fn(a, b, tokens_a_new, tokens_b_new)
    return pairs


def download_main(data_dir):
    fnames = (
        list(glob(f'{data_dir}/**/*.json')) +
        list(glob(f'{data_dir}/**/*.jsonl.xz')) +
        list(glob(f'{data_dir}/**/*.json.xz'))
    )
    print(fnames)
    for fname in tqdm(fnames):
        df = pd.read_json(fname)
        print(f'found {len(df)} samples on {fname}')
        for row in tqdm(df.itertuples(), total=len(df)):

            for data_type in DATA2FIND_PAIRS_FN:
                parse_mnli_sample(row.premise, row.hypothesis, data_type)


def example_run():
    examples = [
        ("the frog and the cats playing outside in the sand on chair",
        "i see other animal inside with a furniture",),

        ("Although Trussville has its share of pubs, bars, discos, and even a well attended casino with revues, most visitors don't come to the island for evening entertainment.",
        "Alabama has many pubs, bars, and discos.",),

        ("and um they have little bar there so we sit there and um sipping on some appletinis before we get to eat",
        "I sipped on a bright green cocktail",),

        ("Cotral leather goods remain unequalled .",
        "There is an Italian company that produces leather goods.",),
    ]

    for a, b in examples:
        for dataset_type in list(DATA2FIND_PAIRS_FN):
            print(dataset_type, a[:20] + '...', b[:20] + '...')
            pairs = parse_mnli_sample(a, b, dataset_type)
            print(pairs)


if __name__ == '__main__':
    download_main('../datasets')
