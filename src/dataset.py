import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE = 512
NUM_CLASSES = 6
IGNORE_INDEX = 255


# ADD THESE LINES TO src/dataset.py
CLASS_NAMES  = ['Sky', 'Sand', 'Rock', 'Vegetation', 'Shadow', 'Distant Terrain']
NUM_CLASSES  = len(CLASS_NAMES)
IMG_SIZE     = 512

# ─── Augmentation Pipelines ───────────────────────────────────────────────────
def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), # Desert terrain often looks valid upside down (sand dunes)
        
        # Geometry
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # Weather/Environment (Crucial for Desert)
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=1.0),
            A.RandomSunFlare(src_radius=100, num_flare_circles_lower=1, num_flare_circles_upper=3, p=1.0),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1.0),
        ], p=0.3),

        # Color/Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        
        # Normalization
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ─── Dataset Class ────────────────────────────────────────────────────────────
class DesertSegDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.images = sorted(list((self.root / 'images').glob('*.*')))
        self.masks  = sorted(list((self.root / 'masks').glob('*.*')))
        
        # Safety check
        if len(self.images) == 0:
            print(f"[Warning] No images found in {self.root / 'images'}. Generate data first!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Read Image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read Mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Handle Resize Mismatch (Safety)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Augment
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Fallback if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()

        return {'image': image, 'mask': mask.long()}

def get_dataloaders(data_dir, batch_size=8, num_workers=2):
    train_ds = DesertSegDataset(data_dir, 'train', transform=get_train_transforms())
    val_ds   = DesertSegDataset(data_dir, 'val', transform=get_val_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader