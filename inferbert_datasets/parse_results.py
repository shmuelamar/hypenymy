import re
from glob import glob

import pandas as pd


def main():
    eval_data = []
    for fname in sorted(glob('logs/**/*.log', recursive=True)):
        with open(fname) as fp:
            content = fp.read()
        if '__eval__' in fname:
            acc = [x.group(1) for x in re.finditer(r'allennlp.commands.evaluate - accuracy: ([0-9.]+)', content)]
            _, dataset_name, trainset_name, num_epochs, lr, _, evalset_name = fname.split('__')
            evalset_name = evalset_name.split('.log')[0]
        else:
            _, dataset_name, trainset_name, num_epochs, lr = fname.split('__')
            lr = lr.split('.log')[0]
            evalset_name = 'best_training_val'
            acc = [x.group(1) for x in re.finditer(r'"best_validation_accuracy": ([0-9.]+)', content)]

        assert len(acc) == 1, f'{fname} {acc}'
        acc = acc[0]
        eval_data.append({
            'dataset_name': dataset_name,
            'trainset_name': trainset_name,
            'num_epochs': num_epochs,
            'lr': lr,
            'evalset_name': evalset_name,
            'acc': round(float(acc)*100, 2),
        })

    df = pd.DataFrame(eval_data)

    df.to_csv('knowbert_eval_results.csv', index=False)


if __name__ == '__main__':
    main()
