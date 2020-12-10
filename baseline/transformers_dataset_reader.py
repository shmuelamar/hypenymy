import os
import re

import pandas as pd
from transformers import DataProcessor, InputExample


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # dynamic (the full name)
        return self._read(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        # hardcoded
        return self._read('../hypernymy/dataset/dataset_creative/dataset_creative_test.json', 'dev')

    def get_labels(self):
        """See base class."""
        return ["contradiction", "neutral", "entailment"]

    def _read(self, file_path: str, set_type: str):
        """Creates examples for the training and dev sets."""
        is_json = re.search(r'\.jsonl?(\.gz)?$', file_path)
        if is_json:
            df = pd.read_json(file_path, lines='jsonl' in is_json.group())
            print(set(df.columns.tolist()))
            # assert len(set(df.columns.tolist()) & {'sentence1', 'text_a'}) == 1, 'too many fields'
            df = df.rename(
                columns={
                    'premise': 'sentence1',
                    'hypothesis': 'sentence2',
                    'label': 'gold_label',
                },
            )
            df = df.rename(
                columns={
                    'text_a': 'sentence1',
                    'text_b': 'sentence2',
                    'label': 'gold_label',
                },
            )
        else:
            df = pd.read_csv(
                file_path,
                sep='\t',
                quotechar='\t',
                keep_default_na=False,
                usecols=['sentence1', 'sentence2', 'gold_label'],
            )

        examples = []
        for idx, example in enumerate(df.itertuples()):
            text_a = example.sentence1
            text_b = example.sentence2
            label = example.gold_label

            examples.append(InputExample(guid=f'{set_type}-{idx}', text_a=text_a, text_b=text_b, label=label))

        return examples
