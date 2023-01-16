import sys
sys.path.append('../')

from imports import *
import json

########################################################################
## Serializing and deserializing: String to PT Tensor and vice versa!

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
## Directory creation
 
def create_dir_if_necessary(path_relative_to_project_root):
    directory = get_absolute_path(path_relative_to_project_root)
    
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory

def get_absolute_path(path_relative_to_project_root):
    import os
    current_directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    final_directory = os.path.join(current_directory, rf'../../{path_relative_to_project_root}')

    return final_directory


########################################################################
## Loading model

def load_checkpoint_for_train(model_name, data_loader, device):
    from models.bert_based_model import ResqueModel

    model = ResqueModel.from_pretrained(CONSTANTS.BERT_MODEL_NAME, num_labels=2)
    model = model.to(device)

    steps_per_epoch = data_loader.get_train_dataset_size() // data_loader.batch_size
    total_training_steps = steps_per_epoch * CONSTANTS.NUMBER_OF_EPOCHS

    optimizer = AdamW(model.parameters(), lr=CONSTANTS.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_training_steps=total_training_steps, 
        num_warmup_steps=CONSTANTS.NUMBER_OF_WARM_UP_STEPS
    )
    save_checkpoint_path = create_dir_if_necessary('checkpoints/') + model_name
    checkpoint = torch.load(save_checkpoint_path)        
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    model.train(True)
    return model, optimizer, scheduler

def load_checkpoint_for_test(model_name):
    from models.bert_based_model import ResqueModel
    model = ResqueModel.from_pretrained(CONSTANTS.BERT_MODEL_NAME, num_labels=2)
    save_checkpoint_path = create_dir_if_necessary('checkpoints/') + model_name
    checkpoint = torch.load(save_checkpoint_path)        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)
    return model


########################################################################
## Selecting Device

def select_cuda_if_available():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def select_cpu():
    device = torch.device('cpu')
    return device

########################################################################

def find_indices(list_to_check, item_to_find):
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]