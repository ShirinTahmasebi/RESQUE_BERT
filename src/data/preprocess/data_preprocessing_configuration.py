from utils.config_key_data_constants import CONFIG_KEYS_DATA
from utils.utils import *

train_config_sqlshare_bert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_RAW_NO_FUNC,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS_DATA.TOKENIZER: BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_BERT_UNCASED),
    CONFIG_KEYS_DATA.NAME: 'Train - BERT',
}

test_config_sqlshare_bert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_TEST_SQLSHARE_RAW_NO_FUNC,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_TEST_SQLSHARE_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_TEST_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS_DATA.TOKENIZER: BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_BERT_UNCASED),
    CONFIG_KEYS_DATA.NAME: 'Test - BERT',
}

val_config_sqlshare_bert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_VAL_SQLSHARE_RAW_NO_FUNC,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_VAL_SQLSHARE_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_VAL_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS_DATA.TOKENIZER: BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_BERT_UNCASED),
    CONFIG_KEYS_DATA.NAME: 'Validation - BERT',
}

train_config_sdss_bert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_TRAIN_SDSS_RAW,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_TRAIN_SDSS_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_TRAIN_SDSS_TOKENIZED_BERT,
    CONFIG_KEYS_DATA.TOKENIZER: BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_BERT_UNCASED),
    CONFIG_KEYS_DATA.NAME: 'Train - BERT',
}

test_config_sdss_bert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_TEST_SDSS_RAW,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_TEST_SDSS_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_TEST_SDSS_TOKENIZED_BERT,
    CONFIG_KEYS_DATA.TOKENIZER: BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_BERT_UNCASED),
    CONFIG_KEYS_DATA.NAME: 'Test - BERT',
}

val_config_sdss_bert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_VAL_SDSS_RAW,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_VAL_SDSS_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_VAL_SDSS_TOKENIZED_BERT,
    CONFIG_KEYS_DATA.TOKENIZER: BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_BERT_UNCASED),
    CONFIG_KEYS_DATA.NAME: 'Validation - BERT',
}

train_config_sdss_codebert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_TRAIN_SDSS_RAW,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_TRAIN_SDSS_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_TRAIN_SDSS_TOKENIZED_CODEBERT,
    CONFIG_KEYS_DATA.TOKENIZER: RobertaTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_CODEBERT),
    CONFIG_KEYS_DATA.NAME: 'Train - CodeBERT',
}

test_config_sdss_codebert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_TEST_SDSS_RAW,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_TEST_SDSS_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_TEST_SDSS_TOKENIZED_CODEBERT,
    CONFIG_KEYS_DATA.TOKENIZER: RobertaTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_CODEBERT),
    CONFIG_KEYS_DATA.NAME: 'Test - CodeBERT',
}

val_config_sdss_codebert = {
    CONFIG_KEYS_DATA.RAW_DATA_PATH: CONSTANTS.DATA_DIR_VAL_SDSS_RAW,
    CONFIG_KEYS_DATA.WITH_CLS_PATH: CONSTANTS.DATA_DIR_VAL_SDSS_WITH_CLS,
    CONFIG_KEYS_DATA.TOKENIZED_PATH: CONSTANTS.DATA_DIR_VAL_SDSS_TOKENIZED_CODEBERT,
    CONFIG_KEYS_DATA.TOKENIZER: RobertaTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_CODEBERT),
    CONFIG_KEYS_DATA.NAME: 'Validation - CodeBERT',
}
