import re

import cbox
import pandas as pd


def repl(s):
    pos = s.lower().find('the **')
    if pos == -1:
        return re.sub(r'\*\*', '}}}', re.sub(r' ?\*\*', ' {{{', s, 1), 1).strip()
    return re.sub(r'\*\*', '}}}', re.sub(r'[Tt]he \*\*', '{{{the ', s, 1), 1).strip()


@cbox.cmd
def main(infile, outfile):
    df = pd.read_csv(infile)
    df = df[df['Approved'] == 1]
    df['formatted_sentence'] = df['s'].apply(repl)
    df.to_csv(outfile, index=False)


if __name__ == '__main__':
    cbox.main(main)
