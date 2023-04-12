from utils.contants import CONSTANTS
from utils.config_key_data_constants import CONFIG_KEYS_DATA
from utils.helper import *
from utils.template_constants import *
from configs.template_preprocessing_configurations import config_sdss_input_creation_train
from template.data.preprocess.templatization.create_sequence import execute_input_creation, execute_masking_task_1, execute_masking_task_2

config = config_sdss_input_creation_train


execute_input_creation(
    input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_LABELED_PATH],
    output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_SIMPLE_PATH],
    type=InputSequenceTypeEnum.SIMPLE
)

execute_input_creation(
    input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_LABELED_PATH],
    output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_PROMPT_V1_PATH],
    type=InputSequenceTypeEnum.PROMPT_V1
)

execute_masking_task_1(
    input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_SIMPLE_PATH],
    output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_MASKED_TASK_1_SIMPLE_PATH],
    type=InputSequenceTypeEnum.SIMPLE
)

execute_masking_task_1(
    input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_PROMPT_V1_PATH],
    output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_MASKED_TASK_1_PROMPT_V1_PATH],
    type=InputSequenceTypeEnum.PROMPT_V1
)

execute_masking_task_2(
    input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_SIMPLE_PATH],
    output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_MASKED_TASK_2_SIMPLE_PATH],
    type=InputSequenceTypeEnum.SIMPLE
)

execute_masking_task_2(
    input_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_INPUT_PROMPT_V1_PATH],
    output_data_path=config[CONFIG_KEYS_DATA.TEMPLATE_MASKED_TASK_2_PROMPT_V1_PATH],
    type=InputSequenceTypeEnum.PROMPT_V1
)


