import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

MORPH_CLASSES = ['spiral', 'shell', 'outflow', 'irregular', 'no_gas']

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

class SyntheticMorphologyDataset(Dataset):
    def __init__(self, data_dir, metadata_csv, transform=None, augment=False):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform
        self.augment = augment
        self.class_to_idx = {c: i for i, c in enumerate(MORPH_CLASSES)}
        self._build_file_list()
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            AddGaussianNoise(0., 0.01),
        ])

    def _build_file_list(self):
        self.samples = []
        for _, row in self.metadata.iterrows():
            label = row['Evolutionary_Class']
            img_id = row['ID']
            img_path = os.path.join(self.data_dir, label, f"{img_id}.npy")
            if os.path.exists(img_path):
                self.samples.append((img_path, self.class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = np.load(img_path)  # shape: (4, 128, 128)
        img = torch.tensor(img, dtype=torch.float32)
        # Normalize to [0, 1] per channel
        img = (img - img.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / (img.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-6)
        if self.augment and self.aug_transforms:
            img = self.aug_transforms(img)
        if self.transform:
            img = self.transform(img)
        return img, label 