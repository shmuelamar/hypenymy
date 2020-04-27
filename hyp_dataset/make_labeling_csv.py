import csv
import json
import re

import cbox


@cbox.cmd
def main(input_file, output_file):
    with open(input_file) as fp:
        dataset = json.load(fp)

    fields = [
        'type',
        'text',
        'label',
        'is_label_correct',
        'is_grammatically',
        'is_makes_sense',
        'ptype->htype',
        'from->to',
        'section',
        'worker',
        'is_completed',
        'row_id',
        'hit_id',
    ]
    with open(output_file, 'w') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in dataset:

            # write premise
            meta = row['metadata']
            for (sent_type, sent) in [('p', 'premise'), ('h', 'hypothesis')]:
                writer.writerow(
                    {
                        'type': sent_type.upper(),
                        'text': make_bold(
                            row[sent], word=meta[f'{sent_type}word']
                        ),
                        'label': row['label'],
                        'is_label_correct': None,
                        'is_grammatically': None,
                        'is_makes_sense': None,
                        'ptype->htype': f'{meta["ptype"]}->{meta["htype"]}',
                        'from->to': f'{meta["pword"]}->{meta["hword"]}',
                        'section': meta['section'],
                        'worker': meta['worker_id'],
                        'is_completed': meta['is_complete'],
                        'row_id': meta['row_id'],
                        'hit_id': meta['hit_id'],
                    }
                )

            # write empty line
            writer.writerow({f: None for f in fields})


def make_bold(s, word):
    new_sent = re.sub(fr'\b{re.escape(word)}', f'**{word}**', s, 1)
    assert s != new_sent, 'word not found'
    return new_sent


if __name__ == '__main__':
    cbox.main(main)
