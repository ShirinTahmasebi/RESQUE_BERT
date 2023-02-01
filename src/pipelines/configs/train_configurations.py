from Projects.RESQU_BERT.src.utils.config_key_model_constants import CONFIG_KEYS_MODEL
from utils.contants import CONSTANTS
from models.bert_based_model import ResqueRoBertaModel, ResqueBertModel

train_config_sdss_bert_no_freezing = {
    CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS: 2,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX: 0,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN: 0,
    CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS: ResqueBertModel,
    CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_BERT_UNCASED,
    CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT: False,
    CONFIG_KEYS_MODEL.CHECKPOINT_NAME: None,
    CONFIG_KEYS_MODEL.DATASET_PATH: CONSTANTS.DATA_DIR_TRAIN_SDSS_TOKENIZED_BERT,
    CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX: 'RESQUE_BERT_FREEZED_SDSS_',
}

train_config_sqlshare_bert_no_freezing = {
    CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS: 2,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX: 0,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN: 0,
    CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS: ResqueBertModel,
    CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_BERT_UNCASED,
    CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT: False,
    CONFIG_KEYS_MODEL.CHECKPOINT_NAME: None,
    CONFIG_KEYS_MODEL.DATASET_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX: 'RESQUE_BERT_FREEZED_SQLSHARE_0_2',
}

train_config_sqlshare_bert_freeze_2 = {
    CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS: 2,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX: 2,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN: 2,
    CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS: ResqueBertModel,
    CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_BERT_UNCASED,
    CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT: False,
    CONFIG_KEYS_MODEL.CHECKPOINT_NAME: None,
    CONFIG_KEYS_MODEL.DATASET_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX: 'RESQUE_BERT_FREEZED_SQLSHARE_',
}

train_config_sqlshare_bert_freeze_6_from_checkpoint = {
    CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS: 2,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX: 6,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN: 6,
    CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS: ResqueBertModel,
    CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_BERT_UNCASED,
    CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT: True,
    CONFIG_KEYS_MODEL.CHECKPOINT_NAME: 'RESQUE_BERT_FREEZED_SQLSHARE_6_1.pt',
    CONFIG_KEYS_MODEL.DATASET_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX: 'RESQUE_BERT_FREEZED_SQLSHARE_',
}

train_config_sqlshare_bert_freeze_3 = {
    CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS: 4,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX: 3,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN: 3,
    CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS: ResqueBertModel,
    CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_BERT_UNCASED,
    CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT: False,
    CONFIG_KEYS_MODEL.CHECKPOINT_NAME: None,
    CONFIG_KEYS_MODEL.DATASET_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT_CONCAT,
    CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX: 'RESQUE_BERT_FREEZED_SQLSHARE_',
}

train_config_sqlshare_bert_freeze_3_with_checkpoint = {
    CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS: 2,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX: 3,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN: 3,
    CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS: ResqueBertModel,
    CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_BERT_UNCASED,
    CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT: True,
    CONFIG_KEYS_MODEL.CHECKPOINT_NAME: 'RESQUE_BERT_FREEZED_SQLSHARE_3_2.pt',
    CONFIG_KEYS_MODEL.DATASET_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX: 'RESQUE_BERT_FREEZED_SQLSHARE_',
}

train_config_sqlshare_bert_no_freezing_from_checkpoint = {
    CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS: 2,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX: 0,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN: 0,
    CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS: ResqueBertModel,
    CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_BERT_UNCASED,
    CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT: True,
    CONFIG_KEYS_MODEL.CHECKPOINT_NAME: '',
    CONFIG_KEYS_MODEL.DATASET_PATH: CONSTANTS.DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT,
    CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX: 'RESQUE_BERT_FREEZED_SQLSHARE_',
}

train_config_sdss_codebert_no_freezing = {
    CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS: 2,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX: 0,
    CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN: 0,
    CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS: ResqueRoBertaModel,
    CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME: CONSTANTS.BERT_MODEL_CODEBERT,
    CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT: False,
    CONFIG_KEYS_MODEL.CHECKPOINT_NAME: None,
    CONFIG_KEYS_MODEL.DATASET_PATH: CONSTANTS.DATA_DIR_TRAIN_SDSS_TOKENIZED_CODEBERT,
    CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX: 'RESQUE_CODEBERT_FREEZED_SDSS_',
}
