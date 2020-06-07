import os

import spacy
import json

SRCDIR = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(SRCDIR)
NLP = None


def json_load(fname):
    with open(fname) as fp:
        return json.load(fp)


def jsonl_dump(data, fname, mode='w'):
    with open(fname, mode) as fp:
        for line in data:
            fp.write(json.dumps(line))
            fp.write('\n')


def jsonl_load(fname):
    with open(fname) as fp:
        for line in fp:
            yield json.loads(line)


def get_nlp():
    """lazy load nlp"""
    global NLP

    if not NLP:
        NLP = spacy.load('en_core_web_lg')
    return NLP
