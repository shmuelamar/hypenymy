"""From Ohad"""
from functools import lru_cache

import spacy
from nltk.corpus import wordnet as wn
import inflect

INFLECT = inflect.engine()
NLP_PARSER = spacy.load('en_core_web_lg')


def is_plural(word):
    try:
        return not INFLECT.singular_noun(word) == False
    except IndexError:
        return False


def to_singular(word_list):
    # make it hashable (immutable)
    return to_singular_cached(tuple(word_list))


@lru_cache(maxsize=1_000_000)
def to_singular_cached(word_list):
    return [INFLECT.singular_noun(w) if is_plural(w) else w for w in word_list]


def all_hypernyms2(word, noun_only=False):
    results = []
    if wn.synsets(word):
        for sense in wn.synsets(word):
            if not noun_only or sense.pos() == 'n':
                for path in sense.hypernym_paths():
                    results += path[:-1]
        return list(set(results))


def list_hypernyms2(word_list, noun_only=False):
    output = []
    for w in word_list:
        h = all_hypernyms2(w, noun_only=noun_only)
        if h:
            output.append((w, h))
    return output


def pick_nouns_for_hypernyms(sentence):
    doc = NLP_PARSER(sentence)
    entities = [a.text for a in doc.ents]  # entities from SpiCy NER
    nouns = []
    for chunk in doc.noun_chunks:  # looking for NP chunks
        cand = chunk.root
        loc_after_chunk = (
            (' ' + sentence + ' ').find(chunk.root.text)
            + len(chunk.root.text)
            - 1
        )
        loc_after_chunk = (
            loc_after_chunk - 1
            if loc_after_chunk >= len(sentence)
            else loc_after_chunk
        )  # In case the chunk ends where the sentence ends (so no character after it)
        has_dash = (
            sentence[(' ' + sentence + ' ').find(chunk.root.text) - 2] == '-'
        ) or (sentence[loc_after_chunk] == '-')
        if (
            cand.text not in entities
            and not cand.text[0].isupper()
            and cand.pos_ == 'NOUN'
            and not has_dash
        ):
            nouns.append(chunk.root.text)
    return nouns, doc


@lru_cache(maxsize=1_000_000)
def get_hypernym_candidates2(sentence_string, is_print=True):
    nouns, doc = pick_nouns_for_hypernyms(sentence_string)
    words_hypernyms = list_hypernyms2(nouns, noun_only=True)
    return words_hypernyms, doc


def find_hyper_hypo_pairs(s1, s2):
    # find hypernym-hyponym pairs, so one is in s1 and the other in s2 and vise versa
    excluded_classes = [
        'object',
        'thing',
        'attribute',
        'abstraction',
        'entity',
        'object',
        'whole',
        'artifact',
        'physical_entity',
        'substance',
        'psychological_feature',
        'matter',
        'cognitive_state',
        'causal_agent',
        'living_thing',
        'event',
        'act',
    ]

    pairs = []
    not_printed = True
    words_hypernyms_s1, doc1 = get_hypernym_candidates2(s1, is_print=False)
    words_hypernyms_s2, doc2 = get_hypernym_candidates2(s2, is_print=False)
    doc1 = [i.text for i in doc1]
    doc2 = [i.text for i in doc2]

    # looking for hypo in s1 and hyper in s2
    for (w, hs) in words_hypernyms_s1:
        # print('*** ' + w + ':')
        for hi in hs:
            hyper_s = hi.name().split('.')[0]  # should be always singular (?)
            # print(hyper_s + ' -> ')
            # if s2.find(hyper_s) > -1 and s1.find9(hyper_s) < 0 and s2.find(w) < 0 and hyper_s not in excluded_classes:
            if (
                hyper_s in to_singular(doc2)
                and hyper_s not in excluded_classes
            ):
                ind = [
                    i for i, x in enumerate(to_singular(doc2)) if x == hyper_s
                ][0]
                hyper_s_orig = doc2[ind]
                pairs.append((w, hyper_s_orig))
                if not_printed:
                    print('\n' + s1)
                    print(s2)
                    not_printed = False
                else:
                    print("-----------Multiple----------")
                print('*** Found in H:%s -> %s' % (w, hyper_s_orig))

    # looking for hypo in s2 and hyper in s1
    for (w, hs) in words_hypernyms_s2:
        # print('*** ' + w + ':')
        for hi in hs:
            hyper_s = hi.name().split('.')[0]
            # print(hyper_s + ' -> ')
            # if s2.find(hyper_s) > -1 and s1.find9(hyper_s) < 0 and s2.find(w) < 0 and hyper_s not in excluded_classes:
            if (
                hyper_s in to_singular(doc1)
                and hyper_s not in excluded_classes
            ):
                ind = [
                    i for i, x in enumerate(to_singular(doc1)) if x == hyper_s
                ][0]
                hyper_s_orig = doc1[ind]
                pairs.append((w, hyper_s_orig))
                if not_printed:
                    print('\n' + s1)
                    print(s2)
                    not_printed = False
                else:
                    print("-----------Multiple----------")
                print('*** Found in H:%s -> %s' % (w, hyper_s_orig))
    return pairs
