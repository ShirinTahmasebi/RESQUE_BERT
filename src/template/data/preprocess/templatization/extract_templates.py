from sql_metadata import Parser
from Projects.RESQU_BERT.src.utils.helper import *
from imports import *
import sys
sys.path.append('../')


def templatize_query(query):
    template = None
    num_of_tables = None
    num_of_attributes_select = None
    num_of_attributes_where = None
    num_of_attributes_order_by = None
    num_of_attributes_group_by = None
    num_of_attributes_join = None

    try:
        # Extract some meta-data about the query
        parser = Parser(query)
        
        num_of_tables = return_result_or_zero(
            func=len,
            input_args=parser.tables
        )

        num_of_attributes_select = return_result_or_zero(
            func=len,
            input_args=fetch_dict_by_key(parser.columns_dict, 'select')
        )

        num_of_attributes_where = return_result_or_zero(
            func=len, 
            input_args=fetch_dict_by_key(parser.columns_dict, 'where')
        )

        num_of_attributes_order_by = return_result_or_zero(
            func=len, 
            input_args=fetch_dict_by_key(parser.columns_dict, 'order_by')
        )

        num_of_attributes_group_by = return_result_or_zero(
            func=len, 
            input_args=fetch_dict_by_key(parser.columns_dict, 'group_by')
        )

        num_of_attributes_join = return_result_or_zero(
            func=len, 
            input_args=fetch_dict_by_key(parser.columns_dict, 'join')
        )

        # Extract template
        attributes = set(itertools.chain(*parser.columns_dict.values()))
        template = query
        for attr in attributes:
            template = template.replace(attr, 'ATTR_NAME')

        for tbl_name in parser.tables:
            template = template.replace(tbl_name, 'TBL_NAME')

        template = Parser(template).generalize

    except Exception as ex:
        print('----------------------------------------')
        print('Error in parsing the query:')
        print(f'Query = {str(query)}')
        print(f'Exception = {str(ex)}')

    return {
        'template_v2': template,
        'num_of_tables': num_of_tables,
        'num_of_attributes_select': num_of_attributes_select,
        'num_of_attributes_where': num_of_attributes_where,
        'num_of_attributes_order_by': num_of_attributes_order_by,
        'num_of_attributes_group_by': num_of_attributes_group_by,
        'num_of_attributes_join': num_of_attributes_join,
    }


def execute(input_dataset_path, output_dataset_path):
    input_dataset_path_dir = get_absolute_path(input_dataset_path)
    output_dataset_path_dir = get_absolute_path(output_dataset_path)

    input_df = pd.read_csv(input_dataset_path_dir)

    input_df['cleaned_statement'] = input_df['statement'].apply(
        lambda q: q.replace("$", "")
    )

    input_df[[
        'template_v2',
        'num_of_tables',
        'num_of_attributes_select',
        'num_of_attributes_where',
        'num_of_attributes_order_by',
        'num_of_attributes_group_by',
        'num_of_attributes_join',
    ]] = pd.DataFrame.from_records(input_df['cleaned_statement'].apply(lambda q: templatize_query(q)))

    input_df[[
        'session_id',
        'cleaned_statement',
        'template',
        'template_v2',
        'num_of_tables',
        'num_of_attributes_select',
        'num_of_attributes_where',
        'num_of_attributes_order_by',
        'num_of_attributes_group_by',
        'num_of_attributes_join',
    ]].to_csv(output_dataset_path_dir)
