from utils.config_key_constants import CONFIG_KEYS
from utils.contants import CONSTANTS
from models.bert_based_model import ResqueRoBertaModel, ResqueBertModel

test_config_sqlshare_bert = {
    CONFIG_KEYS.MODEL_TYPE_CLASS: ResqueBertModel,
    CONFIG_KEYS.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_BERT_UNCASED,
    CONFIG_KEYS.DATASET_PATH: CONSTANTS.DATA_DIR_TEST_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS.MODEL_NAMES_LIST: [
        'RESQUE_BERT_FREEZED_SQLSHARE_3_0.pt',
    ]
}
