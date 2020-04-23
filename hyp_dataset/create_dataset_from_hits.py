import argparse
import json
import re
from collections import namedtuple, Counter

import pandas as pd


class EmptyCellError(ValueError):
    pass


Sample = namedtuple('Sample', 'premise hypothesis label metadata')


def parse_answers_to_sections(df: pd.DataFrame):
    premises = df['Input.premise'].tolist()
    rows = df.to_dict(orient='records')
    # sanity check
    assert len(premises) == len(rows)
    return premises, rows


def parse_ans1_hyper_hypo(premise, row):
    main_hypo = row['Answer.1a. Original_hyponym (hyponym 1)']

    hyper = row['Answer.1a. Main Hypernym']
    label = row['Answer.1b. Label original Hypo->Hyper']
    validate_cell(label)

    return Sample(
        premise=premise,
        hypothesis=safe_replace(premise, old=main_hypo, new=hyper),
        label=label,
        metadata={'section': '1'},
    )


def parse_ans2_hyper_hypos(premise, row):
    main_hypo = row['Answer.1a. Original_hyponym (hyponym 1)']
    main_hyper = row['Answer.1a. Main Hypernym']

    hypos = str2list(row['Answer.2a. Hyponyms list'])

    # take premise sentence, replace it with the hyper word
    hyper_sentence = safe_replace(premise, main_hypo, main_hyper)

    # hypo -> hyper
    labels_hypo2hyper = str2list(row['Answer.2b-1. Label list Hypo->Hyper'])
    assert len(labels_hypo2hyper) == len(hypos)
    for hypo, label in zip(hypos, labels_hypo2hyper):
        yield Sample(
            premise=hyper_sentence,
            hypothesis=safe_replace(premise, old=main_hypo, new=hypo),
            label=label,
            metadata={'section': '2 hypo->hyper'},
        )

    # hyper -> hypo
    label_hyper2hypo = str2list(row['Answer.2b-2. Label list Hyper->Hypo'])
    assert len(label_hyper2hypo) == len(hypos)
    for hyper, label in zip(hypos, label_hyper2hypo):
        yield Sample(
            premise=safe_replace(premise, old=main_hypo, new=hyper),
            hypothesis=hyper_sentence,
            label=label,
            metadata={'section': '2 hyper->hypo'},
        )


def parse_ans3_non_hypos(premise, row):
    main_hypo = row['Answer.1a. Original_hyponym (hyponym 1)']
    main_hyper = row['Answer.1a. Main Hypernym']

    non_hypos = str2list(row['Answer.3a. Non-hyponyms list'])

    # take premise sentence, replace it with the hyper word
    hyper_sentence = safe_replace(premise, main_hypo, main_hyper)

    # non-hypo -> hyper
    labels_nonhypo2hyper = str2list(
        row['Answer.3b-1. Label list Nonhypo->Hyper']
    )
    assert len(labels_nonhypo2hyper) == len(non_hypos)
    for nonhypo, label in zip(non_hypos, labels_nonhypo2hyper):
        yield Sample(
            premise=safe_replace(premise, old=main_hypo, new=nonhypo),
            hypothesis=hyper_sentence,
            label=label,
            metadata={'section': '3 Nonhypo->hyper'},
        )

    # hyper -> non-hypo
    labels_hyper2nonhypo = str2list(
        row['Answer.3b-2. Label list Hyper->Nonhypo']
    )
    assert len(labels_hyper2nonhypo) == len(non_hypos)
    for nonhypo, label in zip(non_hypos, labels_hyper2nonhypo):
        yield Sample(
            premise=hyper_sentence,
            hypothesis=safe_replace(premise, old=main_hypo, new=nonhypo),
            label=label,
            metadata={'section': '3 Hyper->Nonhypo'},
        )


def parse_ans4_hyper_hypo_freestyle(_, row):
    premise = row['Answer.4. Premise (Hyper)']
    premise = premise[premise.find(')') + 1 :].strip()  # remove (1) at start
    validate_cell(premise)

    # hyper -> hypo
    yield Sample(
        premise=premise,
        hypothesis=row['Answer.4a. Hyper->Hypo text'],
        label=row['Answer.4al. Hyper->Hypo label'],
        metadata={'section': '4 hyper->hypo freestyle original'},
    )

    yield Sample(
        premise=premise,
        hypothesis=row['Answer.4b. Hyper->Nonhypo text'],
        label=row['Answer.4al. Hyper->Hypo label'],
        metadata={'section': '4 hyper->Nonhypo freestyle original'},
    )


def parse_ans5_hypo_hyper_freestyle(_, row):
    premise = row['Answer.5. Premise (Hypo)']
    premise = premise[premise.find(')') + 1 :].strip()  # remove (1) at start
    validate_cell(premise)

    # hyper -> hypo
    yield Sample(
        premise=premise,
        hypothesis=row['Answer.5a. Hypo->Hyper text'],
        label=row['Answer.5al. Hypo->Hyper label'],
        metadata={'section': '5 hypo->hyper freestyle original'},
    )

    yield Sample(
        premise=premise,
        hypothesis=row['Answer.5b. Hypo->Nonhypo text'],
        label=row['Answer.5bl. Hypo->Nonhypo label'],
        metadata={'section': '5 hypo->Nonhyper freestyle original'},
    )


def parse_ans4_and_5_freestyle_hypothesis_with_substitution(_, row):
    premise4 = row['Answer.4. Premise (Hyper)']
    premise4 = premise4[
        premise4.find(')') + 1 :
    ].strip()  # remove (1) at start
    premise5 = row['Answer.4. Premise (Hyper)']
    premise5 = premise5[
        premise5.find(')') + 1 :
    ].strip()  # remove (1) at start

    hypos = str2list(row['Answer.2a. Hyponyms list'])
    non_hypos = str2list(row['Answer.3a. Non-hyponyms list'])

    main_hyper = row['Answer.1a. Main Hypernym']
    main_hypo = row['Answer.1a. Original_hyponym (hyponym 1)']
    first_nonhypo = non_hypos[0]

    # 4a
    for hypo_word in hypos:
        yield Sample(
            premise=premise4,
            hypothesis=safe_replace(
                row['Answer.4a. Hyper->Hypo text'],
                old=main_hypo,
                new=hypo_word,
            ),
            label=row['Answer.4al. Hyper->Hypo label'],
            metadata={
                'section': 'dataset3 hyper->hypo freestyle substitution'
            },
        )

    # 4b
    for nonhypo_word in non_hypos[1:]:
        yield Sample(
            premise=premise4,
            hypothesis=safe_replace(
                row['Answer.4b. Hyper->Nonhypo text'],
                old=first_nonhypo,
                new=nonhypo_word,
            ),
            label=row['Answer.4bl. Hyper->Nonhypo label'],
            metadata={
                'section': 'dataset3 hyper->nonhypo freestyle substitution'
            },
        )

    # 5a
    for hypo_word in hypos:
        yield Sample(
            premise=premise5,
            hypothesis=safe_replace(
                row['Answer.5a. Hypo->Hyper text'],
                old=main_hyper,
                new=hypo_word,
            ),
            label=row['Answer.5al. Hypo->Hyper label'],
            metadata={
                'section': 'dataset3 hypo->hyper freestyle substitution'
            },
        )

    # 5b
    for nonhypo_word in non_hypos[1:]:
        yield Sample(
            premise=premise5,
            hypothesis=safe_replace(
                row['Answer.5b. Hypo->Nonhypo text'],
                old=first_nonhypo,
                new=nonhypo_word,
            ),
            label=row['Answer.5bl. Hypo->Nonhypo label'],
            metadata={
                'section': 'dataset3 hypo->nonhypo freestyle substitution'
            },
        )


PARSERS = [
    parse_ans1_hyper_hypo,
    parse_ans2_hyper_hypos,
    parse_ans3_non_hypos,
    parse_ans4_hyper_hypo_freestyle,
    parse_ans5_hypo_hyper_freestyle,
    parse_ans4_and_5_freestyle_hypothesis_with_substitution,
]


def parse_row(premise, row):
    for parser in PARSERS:
        try:
            res = parser(premise, row)
            if isinstance(res, Sample):
                yield res
            else:
                yield from res
        except EmptyCellError:
            continue
        except Exception as e:
            print(repr(e), premise, row)
            raise


def add_metadata(sample: Sample, row, row_id):
    """adds metadata inplace"""
    sample.metadata.update(
        {
            'row_id': row_id,
            'worker_id': row['WorkerId'],
            'hit_id': row['HITId'],
            'is_complete': row['Answer.Submit Location'].strip() == '{}',
        }
    )


def save_samples(samples, output_file):
    with open(output_file, 'w') as fp:
        json.dump([s._asdict() for s in samples], fp, indent=4)


def build_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    premises, rows = parse_answers_to_sections(df)
    all_samples = []

    for row_id, (premise, row) in enumerate(zip(premises, rows)):
        for sample in parse_row(premise, row):
            add_metadata(sample, row, row_id)
            all_samples.append(sample)

    unique_samples = Counter((s.premise, s.hypothesis) for s in all_samples)
    unique_samples_labels = Counter(
        (s.premise, s.hypothesis, s.label) for s in all_samples
    )
    assert len(unique_samples) == len(
        unique_samples_labels
    ), 'contradicting samples'
    print(f'found {len(unique_samples)}/{len(all_samples)} unique/samples')

    print('some duplicates:')
    print(json.dumps(unique_samples.most_common(5), indent=4))

    save_samples(all_samples, output_file)


def safe_replace(s, old, new):
    """replaces old word by new word ensuring no ambiguity"""
    return sentence2template(s, old).format(new)


def sentence2template(s, word):
    validate_cell(s)
    validate_cell(word)
    start, end = get_word_position(s, word)
    assert '{}' not in s
    return s[:start] + '{}' + s[end:]


def get_word_position(s, word):
    matches = list(re.finditer(fr'\b{re.escape(word)}', s))
    if not matches:
        raise ValueError(f'{word!r} found {len(matches)} times on {s!r}')
    return matches[0].start(), matches[0].end()


def validate_cell(s):
    if '{}' in s:
        raise EmptyCellError(f'got empty cell as list - {s}')


def str2list(s):
    validate_cell(s)
    return s.split(',')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-csv')
    parser.add_argument('-o', '--output-dataset')
    args = parser.parse_args()
    build_dataset(args.input_csv, args.output_dataset)


if __name__ == '__main__':
    main()
