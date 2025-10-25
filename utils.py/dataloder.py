import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional, dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoadDatasets(Dataset):
    def __init__(self, data_dir: str, transform=None, subset: str = 'train'):
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset

        class_names = sorted([
            d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        self.samples = [] = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for subset '{self.subset}'")

        self.class_weights = self._compute_class_weights()
        logger.info(f"Computed class weights: {self.class_weights}")

        self._print_class_distribution()