from traffic_light_detection.dataloader.single_frame_dataset import LISADatasetSingleFrame
from traffic_light_detection.dataloader.base_dataset import DatasetType
from pathlib import Path

def test_lisa_dataset():
    ds = LISADatasetSingleFrame(dataset_root=Path("/data/data/LISA-traffic-light-dataset/"), mode=DatasetType.TRAIN)
    print(len(ds))
    ds_test = LISADatasetSingleFrame(dataset_root=Path("/data/data/LISA-traffic-light-dataset/"), mode=DatasetType.TEST)
    print(len(ds_test))
    print(ds[0])