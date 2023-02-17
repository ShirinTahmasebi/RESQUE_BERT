from utils.utils import *


def label_templates(list_of_templatized_dfs, templates_list_path):
    unique_templates = set()

    for dic in list_of_templatized_dfs:
        unique_templates.update(list(dic['df']['template_v2']))

    template_df = pd.DataFrame(unique_templates, columns=['template'])
    template_df['label'] = [f'T{i}' for i in range(len(unique_templates))]

    template_df_path = get_absolute_path(templates_list_path)
    # template_df.to_csv(template_df_path)


def add_template_labels_to_df(list_of_templatized_dfs, templates_list_path):

    template_df_path = get_absolute_path(templates_list_path)
    template_df = pd.read_csv(template_df_path)

    def extract_template_label(template_str):
        return template_df.loc[template_df['template'] == template_str]['label'].values[0]

    for single_templatized_df_info in list_of_templatized_dfs[2:]:
        name = single_templatized_df_info['name']
        single_templatized_df = single_templatized_df_info['df']
        path = single_templatized_df_info['labeled_path']

        single_templatized_df['template_label'] = single_templatized_df['template_v2'].apply(
            lambda template_str: extract_template_label(template_str)
        )

        # single_templatized_df.to_csv(get_absolute_path(path))

        print(f'{name} saved!')


def execute(list_of_templatized_dfs, templates_list_path):
    label_templates(list_of_templatized_dfs, templates_list_path)
    add_template_labels_to_df(
        list_of_templatized_dfs,
        templates_list_path
    )
