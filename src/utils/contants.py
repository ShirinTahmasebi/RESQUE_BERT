## For creating constant variables (public static final variables)
# https://stackoverflow.com/a/2688086

def constant(f):
    def fset(self, value):
        raise TypeError
    def fget(self):
        return f()
    return property(fget, fset)

class Contants(object):
    
    @constant
    def BERT_MODEL_CASED():
        return "bert-base-cased"

    @constant
    def BERT_MODEL_UNCASED():
        return "bert-base-uncased"

    @constant
    def MAX_TOKEN_COUNT():
        return 512
    
    @constant
    def NUMBER_OF_EPOCHS():
        return 8
    
    # Optimizer parameters
    @constant
    def LEARNING_RATE():
        return 2e-5

    @constant
    def NUMBER_OF_WARM_UP_STEPS():
        return 5

    @constant
    def DATA_DIR_TRAIN_RAW():
        return 'data/sdss/processed_data/train/raw/'
    
    @constant
    def DATA_DIR_TRAIN_WITH_CLS():
        return 'data/sdss/processed_data/train/with_cls/'

    @constant
    def PROCESSED_DATA_TRAIN():
        return 'data/test/bert_processed.csv'

    @constant
    def PROCESSED_DATA_TEST():
        return 'data/test/bert_processed.csv'
    
    @constant
    def TABLE_TYPE_ID():
        return 100

    @constant
    def ATTRIBUTE_TYPE_ID():
        return 200

    @constant
    def FUNCTION_TYPE_ID():
        return 300

    ## Define the constants here!

CONSTANTS = Contants()