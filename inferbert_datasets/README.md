# Useful Shell Commands

## find failed train jobs

```bash
$ find logs/ -name '*.log' | sort | \
  xargs grep -c '"best_validation_loss": '
```

## train / dev acc

```bash
$ find logs -name '*.log' | sort | \
  xargs grep -E 'allennlp.commands.evaluate - accuracy:|"validation_accuracy":|"training_accuracy":'
$ find logs -name '*.log' | sort | \
  xargs grep -Eo 'allennlp.commands.evaluate - accuracy:.*|"validation_accuracy":.*'
$ find logs -name '*.log' | sort | xargs grep -E '"validation_accuracy":'
```

## Dataset Eval Example

```bash
$ TMPDIR='/home/nlp/ohadr/shmuela/kb/hypernymy/inferbert_datasets/tmp' \
  PYTHONWARNINGS=once allennlp evaluate --include-package kb.include_all --cuda-device 1 \
  models/color/dataset_creative/kbmodel__color__dataset_creative__3__2e-05/model.tar.gz \
  ../../mnli_1.0/dev_matched.tsv.gz

$ TMPDIR='/home/nlp/ohadr/shmuela/kb/hypernymy/inferbert_datasets/tmp' \
  PYTHONWARNINGS=once allennlp evaluate --include-package kb.include_all --cuda-device 1 \
  models/color/dataset_creative/kbmodel__color__dataset_creative__3__2e-05/model.tar.gz \
 ./datasets/color/dataset_creative/dataset_creative_test.json
```
