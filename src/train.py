import sys
sys.path.append('../')

from imports import *
from data.resque_dataset import ResqueDataset
from data.resque_dataloader import TrainDataLoader
from models.bert_based_model import ResqueModel
from utils.utils import *


def train_epoch(dataloader, model, optimizer, scheduler, device):    
    model.train(True)
    
    running_loss = 0
    last_loss = 0

    for i, batch_data in enumerate(dataloader):
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        token_type_ids = batch_data["token_type_ids"].to(device)
        cls_mask = batch_data["cls_mask"].to(device)
        labels = batch_data["labels"].to(device)

        assert len(labels[0][labels[0] != -1]) == sum(cls_mask[0])
        assert sum(input_ids[0][cls_mask[0] == 1] != 101) == 0

        model.zero_grad()

        loss, predictions = model(
            input_ids=input_ids, 
            atttention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            cls_mask=cls_mask, 
            labels=labels
        )

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        # https://mccormickml.com/2019/07/22/BERT-fine-tuning/
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Gather data and report in console
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # Loss per batch
            running_loss = 0
            print(f'Batch {i + 1} Loss: {last_loss}')



test_df = pd.read_csv(get_absolute_path(CONSTANTS.PROCESSED_DATA_TRAIN))
test_dataset = ResqueDataset(test_df)

data_loader = TrainDataLoader(test_dataset)
test_dataloader = data_loader.train_dataloader()


# Train loop
# resque_model = ResqueModel(CONSTANTS.BERT_MODEL_UNCASED, number_of_classes=2)


# steps_per_epoch = data_loader.get_train_dataset_size() // data_loader.batch_size
# total_training_steps = steps_per_epoch * CONSTANTS.NUMBER_OF_EPOCHS

# optimizer = AdamW(resque_model.parameters(), lr=CONSTANTS.LEARNING_RATE)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, 
#     num_training_steps=total_training_steps, 
#     num_warmup_steps=CONSTANTS.NUMBER_OF_WARM_UP_STEPS
# )
  
resque_model, optimizer, scheduler = load_checkpoint_for_train('model_epoch_1.pt', data_loader)
device = select_cuda_if_available()
resque_model.to(device)

for epoch in range(CONSTANTS.NUMBER_OF_EPOCHS):
    print(f'----> Epoch {epoch + 1}')

    train_epoch(
        dataloader=test_dataloader, 
        optimizer=optimizer, 
        scheduler=scheduler,
        model=resque_model,
        device=device
    )

    save_checkpoint_path = create_dir_if_necessary('checkpoints/') + f'model_epoch_{epoch}.pt'

    torch.save(
        {    
            'epoch': epoch,
            'model_state_dict': resque_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, 
        save_checkpoint_path
    )