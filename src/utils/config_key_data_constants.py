from utils.annotations import *


class ConfigKeysData(object):

    @constant
    def RAW_DATA_PATH():
        return "raw_data_path"

    @constant
    def WITH_CLS_PATH():
        return "with_cls_path"

    @constant
    def TOKENIZED_PATH():
        return "tokenized_path"

    @constant
    def TOKENIZER():
        return "tokenizer"

    @constant
    def NAME():
        return "name"

    @constant
    def ALL_DATA_PATHS():
        return "all_data_paths"

    @constant
    def TEMPLATIZED_PATH():
        return "templatized_path"

    @constant
    def TEMPLATE_LABELED_PATH():
        return "template_labeled_path"

    @constant
    def TEMPLATES_LIST_PATH():
        return "templates_list_path"

    @constant
    def TEMPLATE_SEQUENCE_PATH():
        return "templates_sequence_path"

    @constant
    def TEMPLATE_INPUT_SIMPLE_PATH():
        return "template_input_simple_path"

    @constant
    def TEMPLATE_INPUT_PROMPT_V1_PATH():
        return "template_input_prompt_v1_path"

    @constant
    def TEMPLATE_INPUT_PROMPT_V2_PATH():
        return "template_input_prompt_v2_path"

    @constant
    def TEMPLATE_TOKENIZED_SIMPLE_PATH():
        return "template_tokenized_simple_path"

    @constant
    def TEMPLATE_TOKENIZED_PROMPT_V1_PATH():
        return "template_tokenized_prompt_v1_path"

    @constant
    def TEMPLATE_TOKENIZED_PROMPT_V2_PATH():
        return "template_tokenized_prompt_v2_path"


CONFIG_KEYS_DATA = ConfigKeysData()
