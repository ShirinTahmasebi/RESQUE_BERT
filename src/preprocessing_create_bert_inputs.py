import os
from sql_metadata import Parser
from utils.utils import *
from imports import *
import sys
sys.path.append('../')


def extract_table_names(input):
    tbl_names_list = []
    for word_index, word in enumerate(input.split()):
        if word == '[TBL_CLS]':
            tbl_name = input.split()[word_index + 1]
            tbl_names_list.append(tbl_name)
    return tbl_names_list


def extract_label_names(input, next_query):
    tbl_names = extract_table_names(input)
    labels = []
    for word_index, word in enumerate(input.split()):
        if word == '[ATTR_CLS]':
            fragment = input.split()[word_index + 1]
            if fragment in next_query.split():
                labels.append(1)
            elif fragment.split(".")[-1] in next_query.split():
                labels.append(1)
            else:
                does_exists = False
                for tbl_name in tbl_names:
                    if (f"{tbl_name}.{fragment}") in next_query.split():
                        does_exists = True
                        break
                labels.append(1) if does_exists else labels.append(0)
        elif word == '[TBL_CLS]':
            fragment = input.split()[word_index + 1]
            if fragment in next_query.split():
                labels.append(1)
            else:
                labels.append(0)
    return labels


def extract_classification_type_ids(input):
    # Create classification_type_ids
    classification_type_ids = []
    for token in input.split():
        if token == '[TBL_CLS]':
            classification_type_ids.append(CONSTANTS.TABLE_TYPE_ID)
        elif token == '[ATTR_CLS]':
            classification_type_ids.append(CONSTANTS.ATTRIBUTE_TYPE_ID)
    return classification_type_ids


def extract_token_type_ids(valid_tokens):
    # Create token_type_ids
    # Token_type_ids are 0s for the 1st sentence, 1 for the 2nd sentence, and so on.
    token_type_ids = []
    # Can be set to 0 and 1. Starts from 0 and flips for each query ([SEP] token).
    type_id_tracker = 0

    for item in valid_tokens:
        current_type_id = type_id_tracker % 2
        token_type_ids.append(current_type_id)

        if item == 102:  # 102 is for the [SEP] token
            type_id_tracker = type_id_tracker + 1

    token_type_id_tensorized = torch.tensor(
        [token_type_ids], dtype=torch.long)
    token_type_id_tensorized = F.pad(token_type_id_tensorized, pad=(
        0, CONSTANTS.MAX_TOKEN_COUNT - len(token_type_id_tensorized[0])), mode='constant', value=0)

    return token_type_id_tensorized


def tokenize_single_session(single_session_df, tokenizer):
    list_of_input_queries = []
    session_df_list = []

    for query_index, query in enumerate(single_session_df["statement_with_cls"]):

        # For the last query, we need to skip this procedure. Because there would be no next query!
        if query_index == (len(single_session_df["statement_with_cls"]) - 1):
            break

        # Extract the next query, which is going to be used for calculating the labels.
        next_query = single_session_df["statement_with_cls"].iloc[query_index + 1]

        # Concatenate the queries which have been read so far.
        list_of_input_queries.append(query)
        input = "[SEP]".join(list_of_input_queries) \
            if len(list_of_input_queries) > 1 else list_of_input_queries[0]

        converted_query = input\
            .replace("[ATTR_CLS]", "[CLS]")\
            .replace("[TBL_CLS]", "[CLS]")

        tokenized_query = tokenizer.encode_plus(
            converted_query,
            add_special_tokens=False,
            max_length=CONSTANTS.MAX_TOKEN_COUNT,
            return_token_type_ids=None,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        ).data

        valid_tokens = tokenized_query['input_ids'][tokenized_query['attention_mask'] > 0]

        # TODO: Not very accurate! It can be equal to the maximum number.
        if len(valid_tokens) >= CONSTANTS.MAX_TOKEN_COUNT:
            break

        classification_type_ids = extract_classification_type_ids(input)
        token_type_id_tensorized = extract_token_type_ids(valid_tokens)
        labels = extract_label_names(input, next_query)

        session_df_list.append({
            'input_id': convert_pt_tensor_to_serializable_str(tokenized_query['input_ids']),
            'attention_mask': convert_pt_tensor_to_serializable_str(tokenized_query['attention_mask']),
            'token_type_ids': convert_pt_tensor_to_serializable_str(token_type_id_tensorized),
            'cls_mask': [1 * (x == 101) for x in tokenized_query['input_ids'][0].numpy()],
            'labels': labels,
            'type_ids': classification_type_ids,
        })

    return session_df_list


def execute(with_cls_path, tokenized_path, tokenizer, name=''):
    data_with_cls_path = get_absolute_path(with_cls_path)
    tokenized_data_path = get_absolute_path(tokenized_path)

    df = pd.read_csv(data_with_cls_path)
    unique_session_id_list = df['session_id'].unique()

    final_df_list = []

    for index, session_id in enumerate(unique_session_id_list):
        single_session_df = df[df['session_id'] == session_id].copy()

        single_session_result = tokenize_single_session(single_session_df, tokenizer)
        final_df_list.extend(single_session_result)

        if index % 1000 == 0:
            print(f"Tokenization is in progress: {index}")

    final_df = pd.DataFrame(final_df_list)
    final_df.to_csv(tokenized_data_path)


tokenizer = RobertaTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_CODEBERT)
execute(
    CONSTANTS.DATA_DIR_TRAIN_WITH_CLS,
    CONSTANTS.DATA_DIR_TRAIN_TOKENIZED_CODEBERT,
    tokenizer,
    'Train - CODEBERT'
)

execute(
    CONSTANTS.DATA_DIR_TEST_WITH_CLS,
    CONSTANTS.DATA_DIR_TEST_TOKENIZED_CODEBERT,
    tokenizer,
    'Test - CODEBERT'
)
# execute(
#     CONSTANTS.DATA_DIR_VAL_WITH_CLS,
#     CONSTANTS.DATA_DIR_VAL_TOKENIZED_CODEBERT,
#     tokenizer,
#     'Validation - CODEBERT'
# )

tokenizer = BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_BERT_UNCASED)
execute(
    CONSTANTS.DATA_DIR_TRAIN_WITH_CLS,
    CONSTANTS.DATA_DIR_TRAIN_TOKENIZED_BERT,
    tokenizer,
    'Train - BERT'
)

execute(
    CONSTANTS.DATA_DIR_TEST_WITH_CLS,
    CONSTANTS.DATA_DIR_TEST_TOKENIZED_BERT,
    tokenizer,
    'Test - BERT'
)
# execute(
#     CONSTANTS.DATA_DIR_VAL_WITH_CLS,
#     CONSTANTS.DATA_DIR_VAL_TOKENIZED_BERT,
#     tokenizer,
#     'Validation - BERT'
# )
