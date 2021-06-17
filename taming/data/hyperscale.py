import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class HyperScaleBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class COCOTrain(HyperScaleBase):
    def __init__(self, size, keys=None):
        super().__init__()
        # root = "/workspace/dataset/bigmodel/lgaivision-coco-us/train2017"
        root = "/home/taehoon.kim/coco/train2017"    
        print(root)

        with open("/home/taehoon.kim/VQGAN_TPU/data/coco_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class COCOValidation(HyperScaleBase):
    def __init__(self, size, keys=None):
        super().__init__()
        # root = "/workspace/dataset/bigmodel/lgaivision-coco-us/train2017"
        root = "/home/taehoon.kim/coco/train2017"
        with open("/home/taehoon.kim/VQGAN_TPU/data/coco_val.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys
