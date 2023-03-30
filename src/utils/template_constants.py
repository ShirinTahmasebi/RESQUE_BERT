from enum import Enum


class InputSequenceTypeEnum(Enum):
    SIMPLE = 'simple'
    # Adding the number of attributes in different clauses right next to the template label
    PROMPT_V1 = 'prompt_v1'
    PROMPT_V2 = 'prompt_v2'
