from pathlib import Path
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
import torch
from traffic_light_detection.dataloader.base_dataset import DatasetType
from traffic_light_detection.dataloader.single_frame_dataset import LISADataset

class LISADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 32, num_classes:int = 100):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self._num_classes = num_classes

    def setup(self, stage: str):
        self.lisa_test = LISADataset(self.data_dir, mode=DatasetType.TEST, num_classes=self._num_classes)
        mnist_full = LISADataset(self.data_dir, mode=DatasetType.TRAIN, num_classes=self._num_classes)
        generator = torch.Generator().manual_seed(42)
        self.lisa_train, self.lisa_val = random_split(mnist_full, [0.8, 0.2], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.lisa_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.lisa_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.lisa_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.lisa_test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass