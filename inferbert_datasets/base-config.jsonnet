{
    "dataset_reader": {
        "type": "multi_nli",
        "tokenizer_and_candidate_generator": {
            "type": "bert_tokenizer_and_candidate_generator",
            "bert_model_type": "bert-base-uncased",
            "do_lower_case": true,
            "entity_candidate_generators": {
                "wiki": {
                    "type": "wiki"
                },
                "wordnet": {
                    "type": "wordnet_mention_generator",
                    "entity_file": "../models-files/entities.jsonl"
                }
            },
            "entity_indexers": {
                "wiki": {
                    "type": "characters_tokenizer",
                    "namespace": "entity_wiki",
                    "tokenizer": {
                        "type": "word",
                        "word_splitter": {
                            "type": "just_spaces"
                        }
                    }
                },
                "wordnet": {
                    "type": "characters_tokenizer",
                    "namespace": "entity_wordnet",
                    "tokenizer": {
                        "type": "word",
                        "word_splitter": {
                            "type": "just_spaces"
                        }
                    }
                }
            }
        }
    },
    "iterator": {
        "iterator": {
            "type": "basic",
            "batch_size": 32
        },
        "type": "self_attn_bucket",
        "batch_size_schedule": "base-12gb-fp32"
    },
    "model": {
        "model": {
            "type": "from_archive",
            "archive_file": "s3://allennlp/knowbert/models/knowbert_wiki_wordnet_model.tar.gz",
            "weights_file": "../mnli_final_model/bert.th",
        },
        "classifier_weights_file": "../mnli_final_model/clf.th",
        "type": "simple-classifier",
        "bert_dim": 768,
        "metric_a": {
            "type": "categorical_accuracy"
        },
        "num_labels": 3,
        "task": "classification"
    },
    "train_data_path": "../hypernymy/dataset/dataset_creative/dataset_creative_train.json",
    "validation_data_path": "../hypernymy/dataset/dataset_creative/dataset_creative_test.json",
    "trainer": {
        "cuda_device": 3,
        "gradient_accumulation_batch_size": 32,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 5,
            "num_steps_per_epoch": 471 // TODO: maybe change?
        },
        "moving_average": {
            "decay": 0.95
        },
        "num_epochs": 5,
        "num_serialized_models_to_keep": 1,
        "optimizer": {
            "type": "bert_adam",
            "lr": 5e-05,
            "max_grad_norm": 1,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.bias",
                        "LayerNorm.weight",
                        "layer_norm.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "t_total": -1,
            "weight_decay": 0.01
        },
        "should_log_learning_rate": true,
        "validation_metric": "+accuracy"
    },
    "vocabulary": {
        "directory_path": "../models-files/vocabulary_wordnet_wiki.tar.gz"
    }
}
