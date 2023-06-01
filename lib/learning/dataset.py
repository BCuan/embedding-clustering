from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch
import numpy as np


class ImageSet(Dataset):
    def __init__(self, ims, ids, transform=None):
        assert len(ims) == len(ids)
        self.ims = ims
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ims = self.ims[idx]

        if self.transform:
            ims = self.transform(ims)

        ids = self.ids[idx]

        return ims, ids


def get_class_weights(dataset: ImageFolder):
    targets = dataset.targets
    cnt = []
    for c in dataset.classes:
        idx = dataset.class_to_idx[c]
        cnt.append(np.sum(np.array(targets) == idx))

    return 1.0 / np.array(cnt)

