from utils.annotations import *


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

    # TODO Shirin: Revise all the uses!
    @constant
    def BERT_MODEL_NAME():
        return Contants.BERT_MODEL_CODEBERT.__get__(0)

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
        return 'data/processed_data/train_sdss/basic.csv'

    @constant
    def DATA_DIR_TRAIN_SDSS_WITH_CLS():
        return 'data/processed_data/train_sdss/with_cls.csv'

    @constant
    def DATA_DIR_TRAIN_SDSS_TOKENIZED_BERT():
        return 'data/processed_data/train_sdss/tokenized_bert.csv'

    @constant
    def DATA_DIR_TRAIN_SDSS_TOKENIZED_CODEBERT():
        return 'data/processed_data/train_sdss/tokenized_codebert.csv'

    @constant
    def DATA_DIR_TEST_SDSS_RAW():
        return 'data/processed_data/test_sdss/basic.csv'

    @constant
    def DATA_DIR_TEST_SDSS_WITH_CLS():
        return 'data/processed_data/test_sdss/with_cls.csv'

    @constant
    def DATA_DIR_TEST_SDSS_TOKENIZED_BERT():
        return 'data/processed_data/test_sdss/tokenized_bert.csv'

    @constant
    def DATA_DIR_TEST_SDSS_TOKENIZED_CODEBERT():
        return 'data/processed_data/test_sdss/tokenized_codebert.csv'

    @constant
    def DATA_DIR_VAL_SDSS_RAW():
        return 'data/processed_data/val_sdss/basic.csv'

    @constant
    def DATA_DIR_VAL_SDSS_WITH_CLS():
        return 'data/processed_data/val_sdss/with_cls.csv'

    @constant
    def DATA_DIR_VAL_SDSS_TOKENIZED_BERT():
        return 'data/processed_data/val_sdss/tokenized_bert.csv'

    @constant
    def DATA_DIR_VAL_SDSS_TOKENIZED_CODEBERT():
        return 'data/processed_data/val_sdss/tokenized_codebert.csv'

    # SqlShare

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_RAW():
        return 'data/processed_data/train_sqlshare/basic.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_WITH_CLS():
        return 'data/processed_data/train_sqlshare/with_cls.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT_CONCAT():
        return 'data/processed_data/train_sqlshare/tokenized_bert_concat.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_BERT():
        return 'data/processed_data/train_sqlshare/tokenized_bert.csv'

    @constant
    def DATA_DIR_TRAIN_SQLSHARE_TOKENIZED_CODEBERT():
        return 'data/processed_data/train_sqlshare/tokenized_codebert.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_RAW():
        return 'data/processed_data/test_sqlshare/basic.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_WITH_CLS():
        return 'data/processed_data/test_sqlshare/with_cls.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_TOKENIZED_BERT():
        return 'data/processed_data/test_sqlshare/tokenized_bert.csv'

    @constant
    def DATA_DIR_TEST_SQLSHARE_TOKENIZED_CODEBERT():
        return 'data/processed_data/test_sqlshare/tokenized_codebert.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_RAW():
        return 'data/processed_data/val_sqlshare/basic.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_WITH_CLS():
        return 'data/processed_data/val_sqlshare/with_cls.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_TOKENIZED_BERT():
        return 'data/processed_data/val_sqlshare/tokenized_bert.csv'

    @constant
    def DATA_DIR_VAL_SQLSHARE_TOKENIZED_CODEBERT():
        return 'data/processed_data/val_sqlshare/tokenized_codebert.csv'

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
