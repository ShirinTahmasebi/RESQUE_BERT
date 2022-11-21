import sys
sys.path.append('../')

from imports import *
from data.resque_dataset import ResqueDataset
from data.resque_dataloader import TestDataLoader
from models.bert_based_model import ResqueModel
from utils.utils import *

def test_epoch(dataloader, model, device):
    model.train(False)

    all_predictions_list = torch.Tensor().to(device)
    all_labels_list = torch.Tensor().to(device)
    tbl_predictions_list = torch.Tensor().to(device)
    tbl_labels_list = torch.Tensor().to(device)
    att_predictions_list = torch.Tensor().to(device)
    att_labels_list = torch.Tensor().to(device)
    func_predictions_list = torch.Tensor().to(device)
    func_labels_list = torch.Tensor().to(device)

    for i, batch_data in enumerate(dataloader):
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        token_type_ids = batch_data["token_type_ids"].to(device)
        cls_mask = batch_data["cls_mask"].to(device)
        labels = batch_data["labels"].to(device)
        valid_labels = labels[labels != -1]
        output_types = batch_data["type_ids"].to(device)
        output_types = output_types[output_types != -1]

        _, predictions = model(
            input_ids=input_ids, 
            atttention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            cls_mask=cls_mask, 
            labels=labels
        )

        all_predictions_list = torch.cat((predictions, all_predictions_list), 0)
        all_labels_list = torch.cat((valid_labels, all_labels_list), 0)

        tbl_condition = output_types == CONSTANTS.TABLE_TYPE_ID
        tbl_predictions_list = torch.cat((predictions[tbl_condition], tbl_predictions_list), 0)
        tbl_labels_list = torch.cat((valid_labels[tbl_condition], tbl_labels_list), 0)
        
        att_condition = output_types == CONSTANTS.ATTRIBUTE_TYPE_ID
        att_predictions_list = torch.cat((predictions[att_condition], att_predictions_list), 0)
        att_labels_list = torch.cat((valid_labels[att_condition], att_labels_list), 0)
        
        func_condition = output_types == CONSTANTS.FUNCTION_TYPE_ID
        func_predictions_list = torch.cat((predictions[func_condition], func_predictions_list), 0)
        func_labels_list = torch.cat((valid_labels[func_condition], func_labels_list), 0)
    
    print("Statistics:")
    print(f"Predictions: {all_predictions_list}")
    print(f"Labels: {all_labels_list}")
    print(f"Total Accuracy: {torch.sum(all_predictions_list == all_labels_list) / len(all_labels_list)}")
    print(f"Table Accuracy: {torch.sum(tbl_predictions_list == tbl_labels_list) / len(tbl_labels_list)}")
    print(f"Attribute Accuracy: {torch.sum(att_predictions_list == att_labels_list) / len(att_labels_list)}")
    print(f"Function Accuracy: {torch.sum(func_predictions_list == func_labels_list) / len(func_labels_list)}")


# Load Model
model = load_checkpoint_for_test('model_epoch_1.pt')
device = select_cuda_if_available()
model.to(device)

test_df = pd.read_csv(get_absolute_path(CONSTANTS.PROCESSED_DATA_TEST))
test_dataset = ResqueDataset(test_df)
test_dataloader = TestDataLoader(test_dataset).test_dataloader()

test_epoch(test_dataloader, model, device)