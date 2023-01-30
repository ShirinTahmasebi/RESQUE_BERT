import sys
from utils.utils import *

sys.path.append('../')

from imports import *

def create_mock_bert_processed_dataset():
    tokenizer = BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_NAME)

    q1 = "[CLS] select [CLS] name , [CLS] type from [CLS] dbobjects where [CLS] type = 'u' and [CLS] access = 'u' and [CLS] name not in ( 'loadevents' , 'queryresults' ) order by [CLS] name"
    q2 = "[CLS] select [CLS] name , [CLS] type from [CLS] dbobjects where [CLS] type = 'u' and [CLS] name not in ( 'loadevents' , 'queryresults' ) order by [CLS] name"
    
    tokenized_q1  = tokenizer.encode_plus(
        q1,
        add_special_tokens=False,
        max_length=CONSTANTS.MAX_TOKEN_COUNT,
        return_token_type_ids=None,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    ).data

    input_ids_q1 =  tokenized_q1['input_ids']
    attention_mask_q1 =  tokenized_q1['attention_mask']
    token_type_ids_q1 =  tokenized_q1['token_type_ids']
    cls_mask_q1 = [1 * (x == 101) for x in input_ids_q1[0].numpy()]

    tokenized_q2  = tokenizer.encode_plus(
        q2,
        add_special_tokens=False,
        max_length=CONSTANTS.MAX_TOKEN_COUNT,
        return_token_type_ids=None,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    ).data

    input_ids_q2 =  tokenized_q2['input_ids']
    attention_mask_q2 =  tokenized_q2['attention_mask']
    token_type_ids_q2 =  tokenized_q2['token_type_ids']
    cls_mask_q2 = [1 * (x == 101) for x in input_ids_q2[0].numpy()]

    tokenized_dic = [
        {
            'input_id': convert_pt_tensor_to_serializable_str(input_ids_q1),
            'attention_mask': convert_pt_tensor_to_serializable_str(attention_mask_q1),
            'token_type_ids': convert_pt_tensor_to_serializable_str(token_type_ids_q1),
            'cls_mask': cls_mask_q1,
            'labels': [1, 0, 1, 0, 1, 0, 1, 0],
            'type_ids': [
                            CONSTANTS.TABLE_TYPE_ID,
                            CONSTANTS.TABLE_TYPE_ID,
                            CONSTANTS.TABLE_TYPE_ID,
                            CONSTANTS.ATTRIBUTE_TYPE_ID,
                            CONSTANTS.ATTRIBUTE_TYPE_ID,
                            CONSTANTS.ATTRIBUTE_TYPE_ID,
                            CONSTANTS.FUNCTION_TYPE_ID,
                            CONSTANTS.FUNCTION_TYPE_ID
                        ],
        },
        {
            'input_id': convert_pt_tensor_to_serializable_str(input_ids_q2),
            'attention_mask': convert_pt_tensor_to_serializable_str(attention_mask_q2),
            'token_type_ids': convert_pt_tensor_to_serializable_str(token_type_ids_q2),
            'cls_mask': cls_mask_q2,
            'labels': [1, 0, 1, 0, 1, 0, 1],
            'type_ids': [
                            CONSTANTS.TABLE_TYPE_ID,
                            CONSTANTS.TABLE_TYPE_ID,
                            CONSTANTS.TABLE_TYPE_ID,
                            CONSTANTS.ATTRIBUTE_TYPE_ID,
                            CONSTANTS.ATTRIBUTE_TYPE_ID,
                            CONSTANTS.FUNCTION_TYPE_ID,
                            CONSTANTS.FUNCTION_TYPE_ID
                        ]
        }
    ]

    final_directory = create_dir_if_necessary('data/test')

    import os
    df = pd.DataFrame(tokenized_dic)
    df.to_csv(os.path.join(final_directory, f"bert_processed.csv"))

create_mock_bert_processed_dataset()
