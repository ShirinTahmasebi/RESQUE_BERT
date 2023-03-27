from utils.helper import *
from imports import *
import sys
sys.path.append('../')


class ResqueDataLoader():

    def __init__(self, dataset, batch_size=8, number_of_workers=2):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.number_of_workers = number_of_workers


class TrainDataLoader(ResqueDataLoader):

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.number_of_workers,
        )

    def get_train_dataset_size(self):
        return len(self.dataset)


class TestDataLoader(ResqueDataLoader):

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.number_of_workers
        )

    def get_test_dataset_size(self):
        return len(self.dataset)
