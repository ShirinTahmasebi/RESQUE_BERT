from utils.contants import CONSTANTS
from utils.config_key_data_constants import CONFIG_KEYS_DATA
from utils.utils import *
from configs.template_preprocessing_configurations import config_sdss_templatization
from Projects.RESQU_BERT.src.template.data.preprocess.extract_templates import execute as execute_templatization
from Projects.RESQU_BERT.src.template.data.preprocess.print_statiistics_of_templatized_df import execute as print_statistics
from Projects.RESQU_BERT.src.template.data.preprocess.extract_labels import execute as execute_labeling

config = config_sdss_templatization


def templatize(list_of_data_paths):
    for item_dict in list_of_data_paths:
        input_raw_dataset_path = item_dict[CONFIG_KEYS_DATA.RAW_DATA_PATH]
        output_templatized_dataset_path = item_dict[CONFIG_KEYS_DATA.TEMPLATIZED_PATH]
        execute_templatization(
            input_raw_dataset_path,
            output_templatized_dataset_path
        )


def fetch_dfs(list_of_data_paths):
    list_of_templatized_dfs = []

    for data_paths in list_of_data_paths:
        name = data_paths[CONFIG_KEYS_DATA.NAME]
        path = data_paths[CONFIG_KEYS_DATA.TEMPLATIZED_PATH]
        labeled_path = data_paths[CONFIG_KEYS_DATA.TEMPLATE_LABELED_PATH]

        templatized_df_dir_path = get_absolute_path(path)
        templatized_df = pd.read_csv(templatized_df_dir_path)
        templatized_df.dropna(axis=0, inplace=True)
        # The level of copy function in append function:
        # https://www.sobyte.net/post/2022-07/py-append/
        list_of_templatized_dfs.append(
            {
                'name': name,
                'labeled_path': labeled_path,
                'df': templatized_df.__deepcopy__()
            }
        )

    return list_of_templatized_dfs


# Comment this line out if you need to create the templatized dataframes
templatize(config[CONFIG_KEYS_DATA.ALL_DATA_PATHS])
list_of_templatized_dfs = fetch_dfs(config[CONFIG_KEYS_DATA.ALL_DATA_PATHS])
print_statistics(list_of_templatized_dfs)
execute_labeling(list_of_templatized_dfs, config[CONFIG_KEYS_DATA.TEMPLATES_LIST_PATH])

