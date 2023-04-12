from utils.helper import *


def execute_template_labels_extraction(list_of_templatized_dfs, templates_list_path, intersection=None):
    unique_templates = set()

    for dic in list_of_templatized_dfs:
        unique_templates.update(list(dic['df']['template_v2']))

    template_df = pd.DataFrame(unique_templates, columns=['template'])
    template_df['label'] = [f'T{i}' for i in range(len(unique_templates))]

    template_df_path = get_absolute_path(templates_list_path)
    template_df.to_csv(template_df_path)


def execute_template_filtering(list_of_templatized_dfs, df_names, input_templates_path, output_filtered_templates_path):

    target_dfs = [
        item for item in list_of_templatized_dfs if item['name'] in df_names
    ]

    df_1 = target_dfs[0]['df']
    df_2 = target_dfs[1]['df']

    template_set_df_1 = set(df_1['template_v2'])
    template_set_df_2 = set(df_2['template_v2'])

    difference_1_2 = set(template_set_df_1 - template_set_df_2)
    intersection = set(template_set_df_1 - difference_1_2)

    template_df_path = get_absolute_path(input_templates_path)
    template_df = pd.read_csv(template_df_path)
    filtered_df = template_df[template_df['template'].isin(intersection)]

    filtered_df_path = get_absolute_path(output_filtered_templates_path)
    filtered_df.to_csv(filtered_df_path)


def execute_labeling(list_of_templatized_dfs, templates_list_path):

    template_df_path = get_absolute_path(templates_list_path)
    template_df = pd.read_csv(template_df_path)

    def extract_template_label(template_str):
        try:
            return template_df.loc[template_df['template'] == template_str]['label'].values[0]
        except:
            return None

    for single_templatized_df_info in list_of_templatized_dfs:
        name = single_templatized_df_info['name']
        single_templatized_df = single_templatized_df_info['df']
        path = single_templatized_df_info['labeled_path']

        single_templatized_df['template_label'] = single_templatized_df['template_v2'].apply(
            lambda template_str: extract_template_label(template_str)
        )
        single_templatized_df.dropna(inplace=True)
        single_templatized_df.to_csv(get_absolute_path(path))

        print(f'{name} saved!')
