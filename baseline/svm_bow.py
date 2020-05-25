import re
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def only_one(it):
    seq = list(it)
    assert len(seq) == 1, f'too many items {seq}'
    return seq[0]


def load_dataset(datadir):
    return [
        pd.read_json(only_one(glob(f'{datadir}/*_{set_type}.json'))).to_dict(
            'records'
        )
        for set_type in ['train', 'test']
    ]


def array_to_labels(samples):
    labels = ('contradiction', 'neutral', 'entailment')
    return [labels.index(s['label']) for s in samples]


def build_train_clf(train_samples, model_params):
    print('building vectorizer')
    tokenizer = re.compile(r"(?u)\b\w\w+\b").findall
    vectorizers = [
        TfidfVectorizer(
            ngram_range=model_params['ngram_range'],
            max_df=float(model_params['max_df']),
            min_df=int(model_params['min_df']),
            max_features=int(model_params['max_features']),
            analyzer=lambda s: tokenizer(s[item]),
        )
        for item in ['premise', 'hypothesis', 'ptype', 'htype']
    ]

    x_train = np.hstack(
        v.fit_transform(train_samples).toarray() for v in vectorizers
    )
    print('done build vectorizer')

    clf = LogisticRegression(
        class_weight='balanced',
        multi_class='auto',
        C=model_params['C'],
        solver='lbfgs',
        verbose=1,
        dual=False,
        max_iter=10000,
    )
    # clf = SVC(kernel=model_params['kernel'], C=model_params['C'], class_weight='balanced', decision_function_shape='ovr')

    print('vectorizer vocab size:', [v.vocabulary_ for v in vectorizers])

    y_train = array_to_labels(train_samples)

    clf.fit(x_train, y_train)
    print('svm training done')
    return vectorizers, clf


def eval_clf(vectorizers, clf, samples):
    y = array_to_labels(samples)

    x = np.hstack([vec.transform(samples).toarray() for vec in vectorizers])
    y_pred = clf.predict(x)

    acc = metrics.accuracy_score(y, y_pred)
    return {
        'acc': acc,
        'f1_macro': metrics.f1_score(y, y_pred, average='macro'),
        'f1_micro': metrics.f1_score(y, y_pred, average='micro'),
        # 'status': 'ok',
        # 'loss': 1 - acc,
    }


def main(datadir):
    train_samples, test_samples = load_dataset(datadir)

    params = {
        'C': 1.0,
        'kernel': 'linear',
        'max_df': 1,
        'max_features': 843000.0,
        'min_df': 1.0,
        'ngram_range': (1, 3),
    }
    vecs, clf = build_train_clf(train_samples, params)
    pprint(eval_clf(vecs, clf, test_samples))


if __name__ == '__main__':
    main('/home/shmulik/biu/ibert/hypenymy/dataset/dataset_full')
