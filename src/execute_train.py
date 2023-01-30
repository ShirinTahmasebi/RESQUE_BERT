from pipelines.train import execute as execute_train
from pipelines.configs.train_configurations import *


def validate_config_and_train(config):
    has_errors = False

    valid_keys = set([
        CONFIG_KEYS.NUMBER_OF_EPOCHS,
        CONFIG_KEYS.NUM_OF_FREEZED_LAYERS_MAX,
        CONFIG_KEYS.NUM_OF_FREEZED_LAYERS_MIN,
        CONFIG_KEYS.MODEL_TYPE_CLASS,
        CONFIG_KEYS.MODEL_TYPE_BASE_NAME,
        CONFIG_KEYS.SHOULD_LOAD_FROM_CHECKPOINT,
        CONFIG_KEYS.CHECKPOINT_NAME,
        CONFIG_KEYS.DATASET_PATH,
        CONFIG_KEYS.OUTPUT_MODEL_PREFIX,
    ])

    if not set(config.keys()) == valid_keys:
        print("There are some irrelevant items in the config!")
        has_errors = True

    ####################
    if config[CONFIG_KEYS.NUM_OF_FREEZED_LAYERS_MIN] > config[CONFIG_KEYS.NUM_OF_FREEZED_LAYERS_MAX]:
        print("Min and max freezed layers!")
        has_errors = True

    if config[CONFIG_KEYS.SHOULD_LOAD_FROM_CHECKPOINT]:
        if not config[CONFIG_KEYS.CHECKPOINT_NAME]:
            print("Checkpoint is set to true but checkpoint name is empty!")
            has_errors = True

    ####################
    valid_model_types = []
    valid_tokenizers = []
    if config[CONFIG_KEYS.MODEL_TYPE_CLASS] == ResqueBertModel:
        valid_model_types = [
            CONSTANTS.BERT_MODEL_BERT_CASED,
            CONSTANTS.BERT_MODEL_BERT_UNCASED
        ]
        valid_tokenizers = [
            CONSTANTS.DATA_DIR_TEST_SDSS_TOKENIZED_BERT,
            CONSTANTS.DATA_DIR_TRAIN_SDSS_TOKENIZED_BERT,
            CONSTANTS.DATA_DIR_VAL_SDSS_TOKENIZED_BERT,
            CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT,
            CONSTANTS.DATA_DIR_TEST_SQLSHARE_TOKENIZED_BERT,
            CONSTANTS.DATA_DIR_VAL_SQLSHARE_TOKENIZED_BERT,
        ]

    elif config[CONFIG_KEYS.MODEL_TYPE_CLASS] == ResqueRoBertaModel:
        valid_model_types = [
            CONSTANTS.BERT_MODEL_CODEBERT
        ]
        valid_tokenizers = [
            CONSTANTS.DATA_DIR_TEST_SDSS_TOKENIZED_CODEBERT,
            CONSTANTS.DATA_DIR_TRAIN_SDSS_TOKENIZED_CODEBERT,
            CONSTANTS.DATA_DIR_VAL_SDSS_TOKENIZED_CODEBERT,
            CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_CODEBERT,
            CONSTANTS.DATA_DIR_TEST_SQLSHARE_TOKENIZED_CODEBERT,
            CONSTANTS.DATA_DIR_VAL_SQLSHARE_TOKENIZED_CODEBERT,
        ]

    else:
        print("Model type class is not known!")
        has_errors = True

    model_type = config[CONFIG_KEYS.MODEL_TYPE_BASE_NAME]
    if model_type not in valid_model_types:
        print("Model type name is not campatible with its type class!")
        has_errors = True

    dataset_path = config[CONFIG_KEYS.DATASET_PATH]
    if dataset_path not in valid_tokenizers:
        print("Dataset is not campatible with its model type class!")
        has_errors = True

    ####################
    if not has_errors:
        execute_train(config)


validate_config_and_train(train_config_sqlshare_bert_freeze_2)
