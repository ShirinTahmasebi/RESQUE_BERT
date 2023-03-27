import sys
sys.path.append('../')

from imports import *
from utils.helper import *

class ResqueDataset(Dataset):
    def __init__(self, bert_input_dataframe):
        super().__init__()
        self.data = bert_input_dataframe
        self.input_ids = bert_input_dataframe.input_id
        self.attention_mask = bert_input_dataframe.attention_mask
        self.token_type_ids = bert_input_dataframe.token_type_ids
        self.cls_mask = bert_input_dataframe.cls_mask
        self.labels = bert_input_dataframe.labels
        self.type_ids = bert_input_dataframe.type_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = convert_serializrd_string_to_pt_tensor(self.input_ids[index], data_type=torch.long)
        attention_mask = convert_serializrd_string_to_pt_tensor(self.attention_mask[index], data_type=torch.long)
        token_type_ids = convert_serializrd_string_to_pt_tensor(self.token_type_ids[index], data_type=torch.long)
        cls_mask = convert_serializrd_string_to_pt_tensor(self.cls_mask[index], data_type=torch.long)
        labels = convert_serializrd_string_to_pt_tensor(self.labels[index], data_type=torch.long)
        type_ids = convert_serializrd_string_to_pt_tensor(self.type_ids[index], data_type=torch.long)

        return dict(
            input_ids=input_ids.flatten(),
            attention_mask=attention_mask.flatten(), 
            token_type_ids=token_type_ids.flatten(), 
            cls_mask=F.pad(cls_mask, pad=(0, CONSTANTS.MAX_TOKEN_COUNT - len(cls_mask)), mode='constant', value=-1),
            labels=F.pad(labels, pad=(0, CONSTANTS.MAX_TOKEN_COUNT - len(labels)), mode='constant', value=-1),
            type_ids=F.pad(type_ids, pad=(0, CONSTANTS.MAX_TOKEN_COUNT - len(type_ids)), mode='constant', value=-1)
        )
