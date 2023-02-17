from utils.annotations import *

# SDSS
PATH_DATA_SDSS = 'data/sdss'

PATH_DATA_SDSS_FRAGMENT = f'{PATH_DATA_SDSS}/fragment'
PATH_DATA_SDSS_TEMPLATE = f'{PATH_DATA_SDSS}/template'

PATH_DATA_SDSS_FRAGMENT_TRAIN = f'{PATH_DATA_SDSS_FRAGMENT}/train'
PATH_DATA_SDSS_FRAGMENT_TEST = f'{PATH_DATA_SDSS_FRAGMENT}/test'
PATH_DATA_SDSS_FRAGMENT_VAL = f'{PATH_DATA_SDSS_FRAGMENT}/val'

PATH_DATA_SDSS_TEMPLATE_TRAIN = f'{PATH_DATA_SDSS_TEMPLATE}/train'
PATH_DATA_SDSS_TEMPLATE_TEST = f'{PATH_DATA_SDSS_TEMPLATE}/test'
PATH_DATA_SDSS_TEMPLATE_VAL = f'{PATH_DATA_SDSS_TEMPLATE}/val'


# SQLShare
PATH_DATA_SQLSHARE = 'data/sqlshare' 

PATH_DATA_SQLSHARE_FRAGMENT = f'{PATH_DATA_SQLSHARE}/fragment'
PATH_DATA_SQLSHARE_TEMPLATE = f'{PATH_DATA_SQLSHARE}/template'

PATH_DATA_SQLSHARE_FRAGMENT_TRAIN = f'{PATH_DATA_SQLSHARE_FRAGMENT}/train'
PATH_DATA_SQLSHARE_FRAGMENT_TEST = f'{PATH_DATA_SQLSHARE_FRAGMENT}/test'
PATH_DATA_SQLSHARE_FRAGMENT_VAL = f'{PATH_DATA_SQLSHARE_FRAGMENT}/val'

PATH_DATA_SQLSHARE_TEMPLATE_TRAIN = f'{PATH_DATA_SQLSHARE_TEMPLATE}/train'
PATH_DATA_SQLSHARE_TEMPLATE_TEST = f'{PATH_DATA_SQLSHARE_TEMPLATE}/test'
PATH_DATA_SQLSHARE_TEMPLATE_VAL = f'{PATH_DATA_SQLSHARE_TEMPLATE}/val'

class Contants(object):

    @constant
    def IS_ROBERTA():
        return True

    @constant
    def BERT_MODEL_BERT_CASED():
        return "bert-base-cased"

    @constant
    def BERT_MODEL_BERT_UNCASED():
        return "bert-base-uncased"

    @constant
    def BERT_MODEL_CODEBERT():
        return "microsoft/codebert-base"

    @constant
    def MAX_TOKEN_COUNT():
        return 512

    @constant
    def NUMBER_OF_EPOCHS():
        return 2

    # Optimizer parameters
    @constant
    def LEARNING_RATE():
        return 2e-5

    @constant
    def NUMBER_OF_WARM_UP_STEPS():
        return 5

    @constant
    def DATA_DIR_TRAIN_SDSS_RAW():
        return f'{PATH_DATA_SDSS}/basic_train.csv'

    @constant
    def DATA_DIR_SDSS_TEMPLATES_LIST():
        return f'{PATH_DATA_SDSS_TEMPLATE}/templates_list.csv'

    @constant
    def DATA_DIR_TRAIN_SDSS_TEMPLATIEZED():
        return f'{PATH_DATA_SDSS_TEMPLATE_TRAIN}/templatized.csv'

    @constant
    def DATA_DIR_TRAIN_SDSS_TEMPLATE_LABELED():
        return f'{PATH_DATA_SDSS_TEMPLATE_TRAIN}/template_labeled.csv'

    @constant
    def DATA_DIR_TRAIN_SDSS_WITH_CLS():
        return f'{PATH_DATA_SDSS_FRAGMENT_TRAIN}/with_cls.csv'

    @constant
    def DATA_DIR_TRAIN_SDSS_TOKENIZED_BERT():
        return f'{PATH_DATA_SDSS_FRAGMENT_TRAIN}/tokenized_bert.csv'

    @constant
    def DATA_DIR_TRAIN_SDSS_TOKENIZED_CODEBERT():
        return f'{PATH_DATA_SDSS_FRAGMENT_TRAIN}/tokenized_codebert.csv'

    @constant
    def DATA_DIR_TEST_SDSS_RAW():
        return f'{PATH_DATA_SDSS}/basic_test.csv'

    @constant
    def DATA_DIR_TEST_SDSS_TEMPLATIEZED():
        return f'{PATH_DATA_SDSS_TEMPLATE_TEST}/templatized.csv'

    @constant
    def DATA_DIR_TEST_SDSS_TEMPLATE_LABELED():
        return f'{PATH_DATA_SDSS_TEMPLATE_TEST}/template_labeled.csv'

    @constant
    def DATA_DIR_TEST_SDSS_WITH_CLS():
        return f'{PATH_DATA_SDSS_FRAGMENT_TEST}/with_cls.csv'

    @constant
    def DATA_DIR_TEST_SDSS_TOKENIZED_BERT():
        return f'{PATH_DATA_SDSS_FRAGMENT_TEST}/tokenized_bert.csv'

    @constant
    def DATA_DIR_TEST_SDSS_TOKENIZED_CODEBERT():
        return f'{PATH_DATA_SDSS_FRAGMENT_TEST}/tokenized_codebert.csv'

    @constant
    def DATA_DIR_VAL_SDSS_RAW():
        return f'{PATH_DATA_SDSS}/basic_val.csv'

    @constant
    def DATA_DIR_VAL_SDSS_TEMPLATIEZED():
        return f'{PATH_DATA_SDSS_TEMPLATE_VAL}/templatized.csv'

    @constant
    def DATA_DIR_VAL_SDSS_TEMPLATE_LABELED():
        return f'{PATH_DATA_SDSS_TEMPLATE_VAL}/template_labeled.csv'

    @constant
    def DATA_DIR_VAL_SDSS_WITH_CLS():
        return f'{PATH_DATA_SDSS_FRAGMENT_VAL}/with_cls.csv'

    @constant
    def DATA_DIR_VAL_SDSS_TOKENIZED_BERT():
        return f'{PATH_DATA_SDSS_FRAGMENT_VAL}/tokenized_bert.csv'

    @constant
    def DATA_DIR_VAL_SDSS_TOKENIZED_CODEBERT():
        return f'{PATH_DATA_SDSS_FRAGMENT_VAL}/tokenized_codebert.csv'

    # SqlShare

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_RAW():
        return f'{PATH_DATA_SQLSHARE}/basic_train.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_RAW_NO_FUNC():
        return f'{PATH_DATA_SQLSHARE}/basic_no_func_train.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_TEMPLATIEZED():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_TRAIN}/templatized.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_WITH_CLS():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_TRAIN}/with_cls.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT_CONCAT():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_TRAIN}/tokenized_bert_concat.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_TRAIN}/tokenized_bert.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_CODEBERT():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_TRAIN}/tokenized_codebert.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_RAW():
        return f'{PATH_DATA_SQLSHARE}/basic_test.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_RAW_NO_FUNC():
        return f'{PATH_DATA_SQLSHARE}/basic_no_func_test.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_WITH_CLS():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_TEST}/with_cls.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_TOKENIZED_BERT():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_TEST}/tokenized_bert.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_TOKENIZED_CODEBERT():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_TEST}/tokenized_codebert.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_RAW():
        return f'{PATH_DATA_SQLSHARE}/basic_val.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_RAW_NO_FUNC():
        return f'{PATH_DATA_SQLSHARE}/basic_no_func_val.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_WITH_CLS():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_VAL}/with_cls.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_TOKENIZED_BERT():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_VAL}/tokenized_bert.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_TOKENIZED_CODEBERT():
        return f'{PATH_DATA_SQLSHARE_FRAGMENT_VAL}/tokenized_codebert.csv'

    #########################

    @constant
    def LOG_PATH():
        return 'trace.log'

    @constant
    def TABLE_TYPE_ID():
        return 100

    @constant
    def ATTRIBUTE_TYPE_ID():
        return 200

    @constant
    def FUNCTION_TYPE_ID():
        return 300

    # Define the constants here!


CONSTANTS = Contants()
