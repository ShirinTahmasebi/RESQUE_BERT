from fragment.pipelines.test import execute as execute_test
from fragment.models.bert_based_model import ResqueRoBertaModel, ResqueBertModel
from fragment.pipelines.configs.test_configurations import test_config_sqlshare_bert
from utils.config_key_model_constants import CONFIG_KEYS_MODEL
from utils.contants import CONSTANTS


def validate_config_and_test(config):
    has_errors = False

    valid_keys = set([
        CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS,
        CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME,
        CONFIG_KEYS_MODEL.DATASET_PATH,
        CONFIG_KEYS_MODEL.MODEL_NAMES_LIST
    ])

    if not set(config.keys()) == valid_keys:
        print("There are some irrelevant items in the config!")
        has_errors = True

    ####################
    valid_model_types = []
    valid_tokenizers = []
    if config[CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS] == ResqueBertModel:
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

    elif config[CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS] == ResqueRoBertaModel:
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

    model_type = config[CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME]
    if model_type not in valid_model_types:
        print("Model type name is not campatible with its type class!")
        has_errors = True

    dataset_path = config[CONFIG_KEYS_MODEL.DATASET_PATH]
    if dataset_path not in valid_tokenizers:
        print("Dataset is not campatible with its model type class!")
        has_errors = True
    ####################

    if not has_errors:
        execute_test(config)


validate_config_and_test(test_config_sqlshare_bert)
