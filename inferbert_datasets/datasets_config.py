import os

BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(BASE_DIR, 'amt-raw-hits')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

MNLI_TRAIN_DATASETS = (
    (
        'mnli_10k_split',
        os.path.join(DATASETS_DIR, 'mnli', 'mnli_train_10000_split.jsonl.xz'),
    ),
    (
        'mnli_full',
        os.path.join(DATASETS_DIR, 'mnli', 'mnli_train_full.jsonl.xz'),
    ),
)


class Dataset:
    """base class for all datasets, subclass and edit the dataset config"""

    _DATASET_CLASSES = {}

    name = None
    input_file = None
    columns = ('premise', 'hypothesis', 'label')
    field_types = ('dataset_creative',)
    fields = ('premise', 'hypothesis', 'label')
    metadata_fields = ()
    excluded_workers = set()
    train_size = 0.8
    dev_size = 0.1
    test_size = 0.1

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Dataset._DATASET_CLASSES[cls.name] = cls

    @staticmethod
    def from_name(name):
        return Dataset._DATASET_CLASSES[name]

    @staticmethod
    def get_all_datasets():
        return list(Dataset._DATASET_CLASSES.values())


# class HypernymyDataset(Dataset):
#     name = 'hypernymy'
#     input_file = 'TBD'
#     metadata_fields = (
#         'row_id',
#         'worker_id',
#         'is_complete',
#         'hit_id',
#         'section',
#         'ptype',
#         'htype',
#         'pword',
#         'hword',
#     )
#     field_types = ('dataset_simple', 'dataset_creative', 'dataset_full')


class LocationDataset(Dataset):
    name = 'location'
    input_file = os.path.join(
        RAW_DATA_DIR, 'location_dataset/location_1010_examples.json'
    )
    metadata_fields = (
        'row_id',
        'worker_id',
        'hit_id',
        'section',
        'location',
        'other_location',
        'country',
    )


class ColorDataset(Dataset):
    name = 'color'
    input_file = os.path.join(
        RAW_DATA_DIR, 'color_dataset/color_examples_all.json'
    )
    metadata_fields = (
        'row_id',
        'worker_id',
        'hit_id',
        'section',
        'color',
        'item',
        'other_color',
        'comment',
    )


class TrademarkDataset(Dataset):
    name = 'trademark'
    input_file = os.path.join(
        RAW_DATA_DIR, 'trademark_dataset/trademark_examples.json'
    )
    metadata_fields = (
        'row_id',
        'worker_id',
        'hit_id',
        'section',
        'company',
        'country',
        'country_adj',
        'other_country',
        'other_country_adj',
    )
