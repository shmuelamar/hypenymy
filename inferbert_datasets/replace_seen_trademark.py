import json
import random
import re


def replace_words(sent, mapping, flags=re.IGNORECASE):
    for old, new in mapping.items():
        new_sent = re.sub(rf'\b{re.escape(old)}\b', new, sent, flags=flags)
        if new_sent != sent:
            return new_sent
    return sent


def get_all_options(fname):
    with open(fname) as fp:
        samples = json.load(fp)

    all_options = set()
    for sample in samples:
        all_options.add((
            sample["company"],
            sample["country"],
            sample["country_adj"],
            sample["other_country"],
            sample["other_country_adj"],
        ))
    return list(all_options)


def main():
    outfile = './datasets/seen_test/trademarks_new.json'
    train_fname = './datasets/trademark/trademark_train.json'
    test_fname = './datasets/trademark/trademark_test.json'

    # build mapping from old to new words
    # random.seed()
    train_options = get_all_options(train_fname)

    random.shuffle(train_options)
    test_options = get_all_options(test_fname)

    old_new_mapping = {tst: list(zip(tst, trn)) for tst, trn in zip(test_options, train_options)}
    assert len(old_new_mapping) == len(test_options)

    # now replace
    with open(test_fname) as fp:
        samples = json.load(fp)

    new_test_samples = []
    for sample in samples:
        mapping = old_new_mapping[
            (
                sample["company"],
                sample["country"],
                sample["country_adj"],
                sample["other_country"],
                sample["other_country_adj"],
            )
        ]

        new_sample = sample.copy()
        new_sample.update({
            "premise": replace_words(sample["premise"], dict(mapping[:2])),
            "hypothesis": replace_words(sample["hypothesis"], dict(mapping[2:])),
            "company": mapping[0][1],
            "country": mapping[1][1],
            "country_adj": mapping[2][1],
            "other_country": mapping[3][1],
            "other_country_adj": mapping[4][1],
        })

        new_test_samples.append(new_sample)

    with open(outfile, 'w') as fp:
        json.dump(new_test_samples, fp, indent=4)


if __name__ == '__main__':
    main()
