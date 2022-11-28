import os
from sql_metadata import Parser
from utils.utils import *
from imports import *
import sys
sys.path.append('../')


cls_data_dir_path = get_absolute_path(CONSTANTS.DATA_DIR_TRAIN_WITH_CLS)
cls_data_files_list = []

for x in os.listdir(cls_data_dir_path):
    if x.endswith(".csv"):
        cls_data_files_list.append(os.path.join(cls_data_dir_path, x))

tokenizer = BertTokenizer.from_pretrained(CONSTANTS.BERT_MODEL_UNCASED)
final_df_list = []

for file in cls_data_files_list:
    list_of_input_queries = []

    session_df = pd.read_csv(file)

    for query_index, query in enumerate(session_df["statement_with_cls"]):

        # For the last query, we need to skip this procedure. Because there would be no next query!
        if query_index == (len(session_df["statement_with_cls"]) - 1):
            break

        next_query = session_df["statement_with_cls"][query_index + 1]

        list_of_input_queries.append(query)
        input = "[SEP]".join(list_of_input_queries) if len(
            list_of_input_queries) > 1 else list_of_input_queries[0]

        converted_query = input.replace(
            "[ATTR_CLS]", "[CLS]").replace("[TBL_CLS]", "[CLS]")

        tbl_names = []
        for word_index, word in enumerate(input.split()):
            if word == '[TBL_CLS]':
                tbl_name = input.split()[word_index + 1]
                tbl_names.append(tbl_name)

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

        tokenized_query = tokenizer.encode_plus(
            converted_query,
            add_special_tokens=False,
            max_length=CONSTANTS.MAX_TOKEN_COUNT,
            return_token_type_ids=None,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        ).data

        valid_tokens = tokenized_query['input_ids'][tokenized_query['input_ids'] > 0]

        # TODO: Not very accurate! It can be equal to the maximum number.
        if len(valid_tokens) >= CONSTANTS.MAX_TOKEN_COUNT:
            break

        # Create classification_type_ids
        classification_type_ids = []
        for token in input.split():
            if token == '[TBL_CLS]':
                classification_type_ids.append(CONSTANTS.TABLE_TYPE_ID)
            elif token == '[ATTR_CLS]':
                classification_type_ids.append(CONSTANTS.ATTRIBUTE_TYPE_ID)

        # Create token_type_ids
        token_type_ids = []
        type_id_tracker = 0

        for item in valid_tokens:
            current_type_id = type_id_tracker % 2
            token_type_ids.append(current_type_id)

            if item == 102:
                type_id_tracker = type_id_tracker + 1  # For the [SEP] token

        token_type_id_tensorized = torch.tensor(
            [token_type_ids], dtype=torch.long)
        token_type_id_tensorized = F.pad(token_type_id_tensorized, pad=(
            0, CONSTANTS.MAX_TOKEN_COUNT - len(token_type_id_tensorized[0])), mode='constant', value=0)

        final_df_list.append({
            'input_id': convert_pt_tensor_to_serializable_str(tokenized_query['input_ids']),
            'attention_mask': convert_pt_tensor_to_serializable_str(tokenized_query['attention_mask']),
            'token_type_ids': convert_pt_tensor_to_serializable_str(token_type_id_tensorized),
            'cls_mask': [1 * (x == 101) for x in tokenized_query['input_ids'][0].numpy()],
            'labels': labels,
            'type_ids': classification_type_ids,
        })

final_directory = create_dir_if_necessary("data/test")
df = pd.DataFrame(final_df_list)
df.to_csv(os.path.join(final_directory, f"bert_processed.csv"))
