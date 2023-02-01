from utils.annotations import *


class ConfigKeysModel(object):

    @constant
    def NUMBER_OF_EPOCHS():
        return "number_of_epochs"

    @constant
    def NUM_OF_FREEZED_LAYERS_MAX():
        return "number_of_freeed_layers_max"

    @constant
    def NUM_OF_FREEZED_LAYERS_MIN():
        return "number_of_freeed_layers_min"

    @constant
    def MODEL_TYPE_CLASS():
        return "model_type_class"

    @constant
    def MODEL_TYPE_BASE_NAME():
        return "model_type_base_name"

    @constant
    def SHOULD_LOAD_FROM_CHECKPOINT():
        return "should_load_from_checkpoint"

    @constant
    def CHECKPOINT_NAME():
        return "checkpoint_name"

    @constant
    def DATASET_PATH():
        return "dataset_path"

    @constant
    def OUTPUT_MODEL_PREFIX():
        return "output_model_prefix"

    @constant
    def MODEL_NAMES_LIST():
        return "model_names_list"


CONFIG_KEYS_MODEL = ConfigKeysModel()
