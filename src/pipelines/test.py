from utils.utils import *
from data.resque_dataloader import TestDataLoader
from data.resque_dataset import ResqueDataset
from imports import *
import sys
sys.path.append('../')


def test_epoch(dataloader, model, device, print_to_log=True):
    model.train(False)

    all_pred_list = torch.Tensor().to(device)
    all_lbl_list = torch.Tensor().to(device)
    tbl_pred_list = torch.Tensor().to(device)
    tbl_lbl_list = torch.Tensor().to(device)
    att_pred_list = torch.Tensor().to(device)
    att_lbl_list = torch.Tensor().to(device)
    func_pred_list = torch.Tensor().to(device)
    func_lbl_list = torch.Tensor().to(device)

    for _, batch_data in enumerate(dataloader):
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

        all_pred_list = torch.cat((predictions, all_pred_list), 0)
        all_lbl_list = torch.cat((valid_labels, all_lbl_list), 0)

        tbl_cond = output_types == CONSTANTS.TABLE_TYPE_ID
        tbl_pred_list = torch.cat((predictions[tbl_cond], tbl_pred_list), 0)
        tbl_lbl_list = torch.cat((valid_labels[tbl_cond], tbl_lbl_list), 0)

        att_cond = output_types == CONSTANTS.ATTRIBUTE_TYPE_ID
        att_pred_list = torch.cat((predictions[att_cond], att_pred_list), 0)
        att_lbl_list = torch.cat((valid_labels[att_cond], att_lbl_list), 0)

        func_cond = output_types == CONSTANTS.FUNCTION_TYPE_ID
        func_pred_list = torch.cat((predictions[func_cond], func_pred_list), 0)
        func_lbl_list = torch.cat((valid_labels[func_cond], func_lbl_list), 0)

    total_acc = torch.sum(all_pred_list == all_lbl_list) / len(all_lbl_list)
    tbl_acc = torch.sum(tbl_pred_list == tbl_lbl_list) / len(tbl_lbl_list)
    att_acc = torch.sum(att_pred_list == att_lbl_list) / len(att_lbl_list)
    func_acc = torch.sum(func_pred_list == func_lbl_list) / len(func_lbl_list)

    print_message("Statistics:", print_to_log=print_to_log)
    print_message(f"Predictions: {all_pred_list}", print_to_log=print_to_log)
    print_message(f"Labels: {all_lbl_list}", print_to_log=print_to_log)
    print_message(f"Total Accuracy: {total_acc}", print_to_log=print_to_log)
    print_message(f"Table Accuracy: {tbl_acc}", print_to_log=print_to_log)
    print_message(f"Attribute Accuracy: {att_acc}", print_to_log=print_to_log)
    print_message(f"Function Accuracy: {func_acc}", print_to_log=print_to_log)


def execute(test_configs):
    from Projects.RESQU_BERT.src.utils.config_key_model_constants import CONFIG_KEYS_MODEL

    MODEL_NAME_LIST = test_configs[CONFIG_KEYS_MODEL.MODEL_NAMES_LIST]
    MODEL_TYPE_CLASS = test_configs[CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS]
    MODEL_TYPE_BASE_NAME = test_configs[CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME]
    TEST_DATASET_DIR = test_configs[CONFIG_KEYS_MODEL.DATASET_PATH]

    for model_name in MODEL_NAME_LIST:
        model = load_checkpoint(
            model_name,
            MODEL_TYPE_CLASS,
            MODEL_TYPE_BASE_NAME
        )
        device = select_cuda_if_available()
        model.to(device)

        test_df = pd.read_csv(get_absolute_path(TEST_DATASET_DIR))
        test_dataset = ResqueDataset(test_df)
        test_dataloader = TestDataLoader(test_dataset).test_dataloader()

        print_message(model_name, print_to_log=True)
        test_epoch(test_dataloader, model, device, print_to_log=True)
