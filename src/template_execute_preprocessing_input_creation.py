from utils.contants import CONSTANTS
from utils.config_key_data_constants import CONFIG_KEYS_DATA
from utils.helper import *
from utils.template_constants import *
from configs.template_preprocessing_configurations import config_sdss_input_creation_train
from template.data.preprocess.model_input.create_sequence import execute_input_creation, execute_tokenization

config = config_sdss_input_creation_train


# execute_input_creation(
#     input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_LABELED_PATH],
#     output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_SIMPLE_PATH],
#     type=InputSequenceTypeEnum.SIMPLE
# )

execute_input_creation(
    input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_LABELED_PATH],
    output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_PROMPT_V1_PATH],
    type=InputSequenceTypeEnum.PROMPT_V1
)

execute_input_creation(
    input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_LABELED_PATH],
    output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_PROMPT_V2_PATH],
    type=InputSequenceTypeEnum.PROMPT_V2
)

execute_tokenization(
    input_data=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_SIMPLE_PATH],
    output_data=config[CONFIG_KEYS_DATA.TEMPLATE_TOKENIZED_SIMPLE_PATH],
)

execute_tokenization(
    input_data=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_PROMPT_V1_PATH],
    output_data=config[CONFIG_KEYS_DATA.TEMPLATE_TOKENIZED_PROMPT_V1_PATH],
)

execute_tokenization(
    input_data=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_PROMPT_V2_PATH],
    output_data=config[CONFIG_KEYS_DATA.TEMPLATE_TOKENIZED_PROMPT_V2_PATH],
)
