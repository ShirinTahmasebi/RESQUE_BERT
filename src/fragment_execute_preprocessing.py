from utils.config_key_data_constants import CONFIG_KEYS_DATA
from fragment.data.preprocess.data_preprocessing_configuration import *
from fragment.data.preprocess.preprocessing_add_cls_tokens import execute as execute_add_cls_tokens
from fragment.data.preprocess.preprocessing_create_bert_inputs import execute as execute_create_bert_inputs
from utils.utils import *


def execute_preprocess_pipeline(config):
    raw_data_path = config[CONFIG_KEYS_DATA.RAW_DATA_PATH]
    with_cls_path = config[CONFIG_KEYS_DATA.WITH_CLS_PATH]
    tokenized_path = config[CONFIG_KEYS_DATA.TOKENIZED_PATH]
    tokenizer = config[CONFIG_KEYS_DATA.TOKENIZER]
    name = config[CONFIG_KEYS_DATA.NAME]

    execute_add_cls_tokens(raw_data_path, with_cls_path, name)
    execute_create_bert_inputs(with_cls_path, tokenized_path, tokenizer, name)


execute_preprocess_pipeline(val_config_sqlshare_bert)

input_dataset_path = get_absolute_path(CONSTANTS.DATA_DIR_VAL_SQLSHARE_RAW)
output_dataset_path = get_absolute_path(
    CONSTANTS.DATA_DIR_VAL_SQLSHARE_RAW_NO_FUNC)

train_sqlshare_df = pd.read_csv(input_dataset_path)

train_sqlshare_df['statement'] = train_sqlshare_df.apply(
    lambda row: replace_count_function(row.statement),
    axis=1
)
train_sqlshare_df.to_csv(output_dataset_path)
