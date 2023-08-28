from pathlib import Path
from typing import Any

import torch
import torch.nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
import torch.nn.functional as F
import cv2
from enum import IntEnum
from traffic_light_detection.dataloader.base_dataset import DatasetType, LISADatasetBase, LightState

class LISADatasetSingleFrame(LISADatasetBase):

    def __init__(self, dataset_root: Path, mode:DatasetType=DatasetType.TRAIN) -> None:
        super().__init__(dataset_root=dataset_root, mode=mode)
        self._transform = Compose([Resize((32, 32)), ToTensor()])


    def __getitem__(self, index: int) -> Any:
        img_path = self._dataset_root / self.data_frame.image_path[index] / self.data_frame.filename[index]
        label = torch.tensor(LightState.get_index(self.data_frame.target[index]), dtype=torch.int64)
        img = Image.open(img_path)
        img_tensor = self._transform(img)
        label_onehot = F.one_hot(label, num_classes=self.num_classes).float()
        return img_tensor, label_onehot
