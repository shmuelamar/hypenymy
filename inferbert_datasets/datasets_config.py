import os

raw_data_dir = os.path.join(os.path.dirname(__file__), 'amt-raw-hits')


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
        raw_data_dir, 'location_dataset/location_1010_examples.json'
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
        raw_data_dir, 'color_dataset/color_examples_all.json'
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
        raw_data_dir, 'trademark_dataset/trademark_examples.json'
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
