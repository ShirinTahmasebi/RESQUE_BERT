from sql_metadata import Parser
from utils.helper import *
from imports import *
import sys
sys.path.append('../')


def remove_from_list(input_list, item_to_remove):
    if item_to_remove in input_list:
        input_list.remove(item_to_remove)
    return input_list


def insert_cls_tokens(statement, tables, attributes):
    words = statement.split()

    tables_indices_list = []
    attributes_indices_list = []

    output = []

    for table in tables:
        tables_indices_list = tables_indices_list + find_indices(words, table)

    for attribute in attributes:
        attributes_indices_list = attributes_indices_list + \
            find_indices(words, attribute)

    for index in range(len(words)):
        if index in tables_indices_list:
            output.append('[TBL_CLS]')
        elif index in attributes_indices_list:
            output.append('[ATTR_CLS]')
        output.append(words[index])

    # using the following two assertions, some queries will also be skipped:
    # 1- The ones which can be parsed but they have no from clause.
    # 2- The ones which can be parsed but they have no select clause.
    # However, it may also delete some valid queries in which the select clause uses '*' (rather than attribute names).
    assert '[TBL_CLS]' in output, 'Table name does not exist in the query.'

    # For SQLShare, it is better to comment out this line. The reason is that in SQLShare, many queries are in the form of "SELECT * FROM ..."; so, in case of applying this rule, most of the data will be deleted.
    # assert '[ATTR_CLS]' in output, 'There is no attribute in the select clause.'

    return ' '.join(output)


def create_df_with_cls_tokens(raw_data_files_list):
    sessions_with_error = []
    final_df = pd.DataFrame()

    df = pd.read_csv(raw_data_files_list)
    unique_session_id_list = df['session_id'].unique()

    for session_id in unique_session_id_list:

        try:
            single_session_df = df[df['session_id'] == session_id].copy()
            single_session_df['cleaned_statement'] = single_session_df['statement'].apply(
                lambda query: query.replace("$", ""))

            # I decided to comment out this line, because it can cause some problems when the attribute names or the table names include numbers!
            # statements = statements.apply(lambda query: Parser(query).generalize)

            single_session_df['tables'] = single_session_df['cleaned_statement'].apply(
                lambda query: Parser(query).tables
            )

            single_session_df['attributes'] = single_session_df['cleaned_statement'].apply(
                lambda query: remove_from_list(Parser(query).columns, 'NUM')
            )

            single_session_df['statement_with_cls'] = single_session_df[['cleaned_statement', 'tables', 'attributes']]\
                .apply(
                lambda x: insert_cls_tokens(
                    x['cleaned_statement'], x['tables'], x['attributes']
                ),
                axis=1
            )

            final_df = pd.concat(
                [final_df, single_session_df],
                ignore_index=True
            )

        except Exception as ex:
            print(
                f'Error in parsing queries of session: {session_id} - {str(ex)}'
            )
            sessions_with_error = sessions_with_error + [session_id]

    return final_df, sessions_with_error


def execute(raw_path, with_cls_path, name=''):
    with_cls_data_dir_path = get_absolute_path(with_cls_path)
    raw_data_dir_path = get_absolute_path(raw_path)

    final_df, sessions_with_error = create_df_with_cls_tokens(
        raw_data_dir_path
    )
    final_df.to_csv(with_cls_data_dir_path)

    print("{name} Data with CLS Tokens - Done!")
    print("Files with errors:")
    print(sessions_with_error)
