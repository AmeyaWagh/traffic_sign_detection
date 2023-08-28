from pathlib import Path
from traffic_light_detection.dataloader.base_dataset import DatasetType, LISADatasetBase

class SequentialDataset(LISADatasetBase):
    def __init__(self, dataset_root: Path, mode: DatasetType = DatasetType.TRAIN, num_classes=100) -> None:
        super().__init__(dataset_root, mode, num_classes)