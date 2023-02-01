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


CONFIG_KEYS_DATA = ConfigKeysData()
