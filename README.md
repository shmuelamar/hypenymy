# Hypernymy Dataset


## Create Datasets

this command will:

* remove dups
* remove excluded workers taggings
* split to train / test (defaults 75%/25%)
  * splits by workers tagging session (HITs) and not single samples to avoid info leakage between train and test
* creates 3 dataset types:
  * [simple](./dataset/dataset_simple) - from section 1-3 of the tagging task, only single word replaced between hypothesis and premise
  * [creative](./dataset/dataset_creative) - from section 4-5 of the tagging task, handwritten hypothesis by turkers
  * [full](./dataset/dataset_full) - contains the two above


```bash
$ python scripts/dataset_split.py --input-file ./dataset/hy_dataset_v1.json --output-dir dataset/
```
