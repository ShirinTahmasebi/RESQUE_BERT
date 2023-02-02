from enum import Enum
from models.bert_based_model import ResqueModel
import json
from imports import *
import sys
sys.path.append('../')


def replace_count_function(query):
    processed_clause = query

    import re
    regex_count = 'count\s*\(\s*(\*|[a-zA-Z_0-9]+)\s*\)'
    regex_time = '(time\s*\(\s*\))'

    find_matches_count = re.search(regex_count, query, re.IGNORECASE)
    find_matches_time = re.search(regex_time, query, re.IGNORECASE)

    if find_matches_count:
        column_name = find_matches_count.group(1)
        if column_name == '*':
            column_name = 'all'
        processed_clause = re.sub(
            regex_count, f"count_{column_name}", processed_clause)

    if find_matches_time:
        column_name = find_matches_time.group(1)
        processed_clause = re.sub(regex_time, "time_", processed_clause)

    return processed_clause

########################################################################
# Serializing and deserializing: String to PT Tensor and vice versa!


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def convert_pt_tensor_to_serializable_str(pt_tensor):
    import json
    return json.dumps(list(np.array(pt_tensor)), cls=NpEncoder)


def convert_serializrd_string_to_pt_tensor(str, data_type=None):
    import json
    if data_type:
        return torch.tensor(json.loads(str), dtype=data_type)
    return torch.tensor(json.loads(str))

########################################################################
# Directory creation


def create_dir_if_necessary(path_relative_to_project_root):
    directory = get_absolute_path(path_relative_to_project_root)

    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def get_absolute_path(path_relative_to_project_root):
    import os
    current_directory = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    final_directory = os.path.join(
        current_directory,
        rf'../../{path_relative_to_project_root}'
    )

    return final_directory


########################################################################
# Loading model
def load_checkpoint(checkpoint_name: str, model_cls: ResqueModel, model_name: str, for_train=True):
    model = model_cls.from_pretrained(model_name, num_labels=2)
    save_checkpoint_path = create_dir_if_necessary('checkpoints/') + \
        checkpoint_name
    checkpoint = torch.load(save_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(for_train)
    return model


########################################################################
# Selecting Device

def select_cuda_if_available():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def select_cpu():
    device = torch.device('cpu')
    return device

########################################################################
# Common concerns


def find_indices(list_to_check, item_to_find):
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]


def print_message(message, print_to_console=True, print_to_log=True, level=logging.INFO):
    if print_to_console:
        print(message)

    if print_to_log:
        logging.log(level=level, msg=message)


########################################################################
# Freezing layers of BERT and CodeBERT

def freeze_layers(model, num_of_layers=0):
    if num_of_layers == 0:
        return

    layers_dic_name_list = {}

    for i in range(num_of_layers):
        layers_dic_name_list[i] = []

    for name, _ in model.bert_model.encoder.named_parameters():
        index = int(name.split('.')[1])
        if index < num_of_layers and f'layer.{index}' in name:
            layers_dic_name_list[index].append(name)

    for i in range(num_of_layers):
        assert len(layers_dic_name_list[i]) == 16

    layers_names_to_be_freezed = list(
        itertools.chain(*layers_dic_name_list.values()))

    for name in layers_names_to_be_freezed:
        model.bert_model.encoder.get_parameter(name).requires_grad = False
