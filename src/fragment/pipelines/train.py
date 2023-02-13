from utils.utils import *
from data.resque_dataloader import TrainDataLoader
from data.resque_dataset import ResqueDataset
from imports import *
import sys
sys.path.append('../')


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

        # To check if, for each classification token, there exist a proper label or not.
        assert len(labels[0][labels[0] != -1]) == sum(cls_mask[0])

        # To check if all of the classification tokens are correctly set to the classification token id.
        assert (len(set(input_ids[0][cls_mask[0] == 1].cpu().numpy())) == 1)

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
            last_loss = running_loss / 1000  # Loss per batch
            running_loss = 0
            print_message(f'Batch {i + 1} Loss: {last_loss}')


def train_model(train_dataset_dir, model, output_model_prefix, number_of_epochs):
    import time
    start_time = time.time()

    train_df = pd.read_csv(get_absolute_path(train_dataset_dir))
    train_dataset = ResqueDataset(train_df)
    train_data_loader = TrainDataLoader(train_dataset)

    train_dataloader = train_data_loader.train_dataloader()
    model.bert_model.encoder.layer
    device = select_cuda_if_available()

    # Train loop
    model = model.to(device)

    steps_per_epoch = train_data_loader.get_train_dataset_size(
    ) // train_data_loader.batch_size
    total_training_steps = steps_per_epoch * CONSTANTS.NUMBER_OF_EPOCHS

    optimizer = AdamW(model.parameters(), lr=CONSTANTS.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=total_training_steps,
        num_warmup_steps=CONSTANTS.NUMBER_OF_WARM_UP_STEPS
    )

    checkpoint_path = create_dir_if_necessary('checkpoints/')

    for epoch in range(number_of_epochs):
        print_message(f'----> Epoch {epoch + 1}')

        train_epoch(
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            device=device
        )

        checkpoint_name = f'{output_model_prefix}_{epoch}.pt'
        save_checkpoint_path = checkpoint_path + checkpoint_name

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # The following two items are not used for loading from checkpoints yet!
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            save_checkpoint_path
        )

    execution_time = (time.time() - start_time) / 60
    print_message(
        f"Execution time for {output_model_prefix} is {execution_time} minutes")


def execute(train_configs):
    from utils.config_key_model_constants import CONFIG_KEYS_MODEL

    NuM_OF_EPOCHS = train_configs[CONFIG_KEYS_MODEL.NUMBER_OF_EPOCHS]
    MAX_NUM_OF_FREEZED_LAYERS = train_configs[CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MAX]
    MIN_NUM_OF_FREEZED_LAYERS = (
        train_configs[CONFIG_KEYS_MODEL.NUM_OF_FREEZED_LAYERS_MIN]) - 1
    MODEL_TYPE_CLASS = train_configs[CONFIG_KEYS_MODEL.MODEL_TYPE_CLASS]
    MODEL_TYPE_BASE_NAME = train_configs[CONFIG_KEYS_MODEL.MODEL_TYPE_BASE_NAME]
    SHOULD_LOAD_FROM_CHECKPOINT = train_configs[CONFIG_KEYS_MODEL.SHOULD_LOAD_FROM_CHECKPOINT]
    CHECKPOINT_NAME = train_configs[CONFIG_KEYS_MODEL.CHECKPOINT_NAME]
    TRAIN_DATASET_DIR = train_configs[CONFIG_KEYS_MODEL.DATASET_PATH]
    OUTPUT_MODEL_PREFIX = train_configs[CONFIG_KEYS_MODEL.OUTPUT_MODEL_PREFIX]

    def initialize_model():
        if SHOULD_LOAD_FROM_CHECKPOINT:
            return load_fragment_checkpoint(
                CHECKPOINT_NAME,
                MODEL_TYPE_CLASS,
                MODEL_TYPE_BASE_NAME,
                for_train=True
            )
        else:
            return MODEL_TYPE_CLASS.from_pretrained(MODEL_TYPE_BASE_NAME)

    for num_of_freezed_layers in list(range(MAX_NUM_OF_FREEZED_LAYERS, MIN_NUM_OF_FREEZED_LAYERS, -1)):
        model = initialize_model()
        freeze_bert_layers(model, num_of_freezed_layers)
        print_message(f"Freezing {num_of_freezed_layers} layers!")

        train_model(
            train_dataset_dir=TRAIN_DATASET_DIR,
            model=model,
            output_model_prefix=f'{OUTPUT_MODEL_PREFIX}{num_of_freezed_layers}',
            number_of_epochs=NuM_OF_EPOCHS
        )
