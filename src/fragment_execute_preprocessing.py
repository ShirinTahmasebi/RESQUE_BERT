from utils.config_key_data_constants import CONFIG_KEYS_DATA
from configs.fragment_preprocessing_configuration import val_config_sqlshare_bert
from fragment.data.preprocess.preprocessing_add_cls_tokens import execute as execute_add_cls_tokens
from fragment.data.preprocess.preprocessing_create_bert_inputs import execute as execute_create_bert_inputs
from utils.utils import *

# The following lines are for converting count function in SELECT clause to a simple attribute.
# By default, this conversion is off! So, to apply it, you need to change the boolean used in the if condition.

should_remove_count_functions = False
if should_remove_count_functions == True:
    src_dataset = CONSTANTS.DATA_DIR_VAL_SQLSHARE_RAW
    target_dataset = CONSTANTS.DATA_DIR_VAL_SQLSHARE_RAW_NO_FUNC

    input_dataset_path = get_absolute_path(src_dataset)
    output_dataset_path = get_absolute_path(target_dataset)

    input_df = pd.read_csv(input_dataset_path)

    input_df['statement'] = input_df.apply(
        lambda row: replace_count_function(row.statement),
        axis=1
    )
    input_df.to_csv(output_dataset_path)


# #######################################################
# The main part for executing the preprocessing pipeline.

def execute_preprocess_pipeline(config):
    raw_data_path = config[CONFIG_KEYS_DATA.RAW_DATA_PATH]
    with_cls_path = config[CONFIG_KEYS_DATA.WITH_CLS_PATH]
    tokenized_path = config[CONFIG_KEYS_DATA.TOKENIZED_PATH]
    tokenizer = config[CONFIG_KEYS_DATA.TOKENIZER]
    name = config[CONFIG_KEYS_DATA.NAME]

    execute_add_cls_tokens(raw_data_path, with_cls_path, name)
    execute_create_bert_inputs(with_cls_path, tokenized_path, tokenizer, name)

execute_preprocess_pipeline(val_config_sqlshare_bert)
