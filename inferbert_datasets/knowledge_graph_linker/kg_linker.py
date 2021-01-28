import base64
import gzip
import json
from functools import partial
from hashlib import md5

import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm

import wordnet_parsing_utils as wn

KG_CACHE_FILENAME = 'knowledge_graph_cache.json.gz'


def load_cache_file():
    with gzip.open(KG_CACHE_FILENAME, 'rt') as fp:
        return json.load(fp)


def save_cache_file():
    d = json.dumps(KG_CACHE).encode('utf8')
    with gzip.open(KG_CACHE_FILENAME, 'w') as fp:
        fp.write(d)


KG_CACHE = load_cache_file()
KG_CACHE_UPDATES = 0
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# FIXME: @ohad - what is the default kwargs for these functions
DATA2FIND_PAIRS_FN = {
    'hypernymy': partial(wn.find_hypernymy_pairs, filter_repeat_word=True),
    'color': partial(wn.find_color_pairs, local_wiki_features=wn.local_wiki_features),
    'location': partial(wn.find_location_country_pairs, local_wiki=wn.local_wiki, include_ORG=True),
    'trademark': partial(wn.find_trademark_country_pairs, local_wiki=wn.local_wiki),
}


def hash_string(s):
    return base64.b64encode(md5(s.encode('utf8')).digest()).decode()


def parse_mnli_sample(a, b, data_type, cached_only=False):
    global KG_CACHE_UPDATES
    cache_key = '|'.join([hash_string(a), hash_string(b), data_type])

    if cache_key not in KG_CACHE:
        if cached_only:
            raise KeyError(f'cannot find {a} {b} {data_type} in cache')
        tokens_a = tokenizer.tokenize(a)
        tokens_b = tokenizer.tokenize(b)
        tokens_a_new = wn.word_peice_connected(tokens_a, [1] * len(tokens_a))[0]
        tokens_b_new = wn.word_peice_connected(tokens_b, [1] * len(tokens_b))[0]

        pairs_fn = DATA2FIND_PAIRS_FN[data_type]

        pairs = pairs_fn(a, b, tokens_a_new, tokens_b_new)
        KG_CACHE[cache_key] = pairs
        KG_CACHE_UPDATES += 1
        if KG_CACHE_UPDATES >= 100:
            save_cache_file()
            KG_CACHE_UPDATES = 0

    return KG_CACHE[cache_key]


def download_main(data_dir):
    # fnames = (
    #     list(glob(f'{data_dir}/color/*.json')) +
    #     list(glob(f'{data_dir}/location/*.json')) +
    #     list(glob(f'{data_dir}/hypernymy/*.json')) +
    #     list(glob(f'{data_dir}/trademark/*.json')) +
    #     list(glob(f'{data_dir}/mnli/*.jsonl.xz')) +
    #     list(glob(f'{data_dir}/mnli/*.json.xz'))
    # )

    fnames = [
        # '../datasets/color/color_dev.json',
        # '../datasets/color/color_train.json',
        # '../datasets/color/color_test.json',
        # '../datasets/location/location_common_test.json',
        # '../datasets/location/location_rare_dev.json',
        # '../datasets/location/location_rare_train.json',
        # '../datasets/location/location_rare_test.json',
        # '../datasets/hypernymy/hypernymy_train.json',
        # '../datasets/hypernymy/hypernymy_dev.json',
        # '../datasets/hypernymy/hypernymy_test.json',
        # '../datasets/trademark/trademark_test.json',
        # '../datasets/trademark/trademark_dev.json',
        # '../datasets/trademark/trademark_train.json',
        ###'../datasets/mnli/mnli_dev_mismatched.jsonl.xz',
        ### '../datasets/mnli/mnli_train_full.jsonl.xz',
        '../datasets/mnli/mnli_dev_matched.jsonl.xz',
        '../datasets/mnli/mnli_train_10k_split.json.xz',
        '../datasets/mnli/mnli_train_100k.json.xz',
    ]
    print(fnames)
    for fname in tqdm(fnames):
        df = pd.read_json(fname, lines=fname.endswith('jsonl') or fname.endswith('jsonl.xz'))
        print(df.columns)
        df.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis'}, inplace=True)

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
            print(json.dumps(pairs, indent=4, sort_keys=True))


if __name__ == '__main__':
    # example_run()
    download_main('../datasets')
