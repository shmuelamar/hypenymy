import os

import pandas as pd
from glob import glob
import inflect

SENT1_FIELD = 'premise'
SENT2_FIELD = 'hypothesis'
HITID_FIELD = 'hit_id'

inflecter = inflect.engine()


def is_plural(word):
    force_singular = ['process', 'bedclothes', 'address', 'compass', 'class', 'business', 'religious', 'judiciousness', 'attentiveness', 'access', 'cross',
                      'abbess', 'relatedness', 'illness', 'sameness', 'resoluteness', 'gas', 'kindness', 'orderliness', 'trustworthiness', 'activeness', 'inattentiveness',
                      'carelessness', 'loss']
    if not word.isalpha(): return word
    if word in force_singular:
        return False
    return not inflecter.singular_noun(word) is False


def to_singular(word):
    return singular_noun_fixed(word) if (word.isalpha() and is_plural(word)) else word


def singular_noun_fixed(word):
    fixes = {'proces':'process', 'bedclothe':'bedclothes', 'due_proces': 'due_process', 'addres':'address', 'compas':'compass', 'clas':'class', 'busines':'business', '':''
        , 'religiou': 'religious', 'judiciousnes':'judiciousness', 'attentivenes':'attentiveness', 'acces':'access', 'cros':'cross'}
    if not word.isalpha(): return word
    sing = inflecter.singular_noun(word)

    if sing in fixes:
        return fixes[sing]
    return sing


def main(dirname, word_field):
    fnames = sorted(f for f in glob(dirname + '/*.json', recursive=True) if os.path.isfile(f))
    print(f'checking {len(fnames)} files: ', fnames)
    dses = [pd.read_json(fname) for fname in fnames]

    for i, df1 in enumerate(dses):
        for j, df2 in enumerate(dses):
            if df1 is df2:
                continue

            fname1 = fnames[i]
            fname2 = fnames[j]

            set1 = {tuple(x) for x in df1[[SENT1_FIELD, SENT2_FIELD]].itertuples(index=False)}
            set2 = {tuple(x) for x in df2[[SENT1_FIELD, SENT2_FIELD]].itertuples(index=False)}

            print(f'checking {i} and {j} non-overlap sentences')
            assert set1 & set2 == set(), set1 & set2

            set1_word = set(df1[word_field].apply(to_singular).str.lower().tolist())
            set2_word = set(df2[word_field].apply(to_singular).str.lower().tolist())
            print(f'checking {i} and {j} non-overlap words')
            assert set1_word & set2_word == set(), set1_word & set2_word

            set1_hit_id = set(df1[HITID_FIELD].tolist())
            set2_hit_id = set(df2[HITID_FIELD].tolist())
            print(f'checking {i} and {j} non-overlap hit ids')

            if fname1.endswith('_test.json') and fname2.endswith('_test.json'):
                print('test files may overlap hit ids')
            else:
                assert set1_hit_id & set2_hit_id == set(), set1_hit_id & set2_hit_id


if __name__ == '__main__':
    main('datasets/location', word_field='location')
    main('datasets/trademark', word_field='company')
    main('datasets/hypernymy', word_field='item')
    main('datasets/color', word_field='item')
