import logging
import re
from enum import IntEnum, Enum
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


class DatasetType(IntEnum):
    TRAIN = 0
    TEST = 1


class LightDetectionType(IntEnum):
    BOX_LEVEL = 0
    BULB_LEVEL = 1


class LightState(Enum):
    STOP = "stop"
    GO = "go"
    WARNING = "warning"
    WARNING_LEFT = "warningLeft"
    STOP_LEFT = "stopLeft"
    GO_LEFT = "goLeft"

    @classmethod
    def target_class_names(cls) -> List[str]:
        return [i.value for i in cls]

    @classmethod
    def get_state(cls, index: int) -> "LightState":
        return [i for i in cls][index]

    @classmethod
    def get_index(cls, state: str) -> int:
        target_class_names = LightState.target_class_names()
        return target_class_names.index(state)


class LISADatasetBase(Dataset):
    target_classes = LightState.target_class_names()
    color_map = {"go": "green", "stop": "red", "warning": "yellow"}
    rgb_color_map = {"go": (0, 255, 0), "stop": (255, 0, 0), "warning": (255, 255, 0)}

    def __init__(
        self,
        dataset_root: Path,
        mode: DatasetType = DatasetType.TRAIN,
        detection_type: LightDetectionType = LightDetectionType.BOX_LEVEL,
    ) -> None:
        super().__init__()
        self._dataset_root = dataset_root
        self._mode = mode
        self._detection_type = detection_type
        self._df = self._load_annotations(mode)
        self._transform = Compose([Resize((32, 32)), ToTensor()])
        print(self._df.filename[0])

    def __len__(self) -> int:
        return len(self._df)

    @property
    def data_frame(self)->pd.DataFrame:
        return self._df
    
    @property
    def num_classes(self)->int:
        return len(LightState)

    def _load_annotations(self, mode: DatasetType) -> pd.DataFrame:
        data_frame: pd.DataFrame = None
        logging.info("Loading dataset in %s mode", str(mode))
        if mode == DatasetType.TRAIN:
            train_sets = ["dayTrain", "nightTrain"]
            data_frame = self._load_annotations_dataframe(train_sets)
        elif mode == DatasetType.TEST:
            test_sets = [
                "daySequence1",
                # "daySequence2",
                "nightSequence1",
                # "nightSequence2",
            ]
            data_frame = self._load_annotations_dataframe(test_sets)
        else:
            raise ValueError(f"{mode} mode not supported.")

        data_frame = data_frame.drop(
            [
                "Origin file",
                "Origin frame number",
                "Origin track",
                "Origin track frame number",
            ],
            axis=1,
        )
        data_frame.columns = [
            "filename",
            "target",
            "x1",
            "y1",
            "x2",
            "y2",
            "image_path",
        ]
        data_frame = data_frame[data_frame["target"].isin(self.target_classes)]
        data_frame["filename"] = data_frame["filename"].apply(
            lambda filename: re.findall("\/([\d\w-]*.jpg)", filename)[0]
        )
        data_frame = data_frame.drop_duplicates().reset_index(drop=True)
        print(data_frame.image_path)
        return data_frame

    def _load_annotations_dataframe(self, set_types: List[str]) -> pd.DataFrame:
        data_frames: List[pd.DataFrame] = []
        annotation_file_name: str = f"frameAnnotations{'BOX' if self._detection_type == LightDetectionType.BOX_LEVEL else 'BULB'}.csv"
        for set_type in set_types:
            set_dir = self._dataset_root / "Annotations" / "Annotations" / f"{set_type}"
            clips = list([clip.name for clip in set_dir.glob("*Clip*")])
            print(clips)
            if len(clips):
                for clip in clips:
                    file_path = set_dir / clip / annotation_file_name
                    _df = pd.read_csv(file_path, sep=";")
                    _df["image_path"] = str(
                        Path(set_type) / f"{set_type}" / f"{clip}" / "frames"
                    )
                    data_frames.append(_df)
            else:
                file_path = set_dir / annotation_file_name
                _df = pd.read_csv(file_path, sep=";")
                _df["image_path"] = str(Path(set_type) / f"{set_type}" / "frames")
                data_frames.append(_df)
        data_frame = pd.concat(data_frames)
        return data_frame
