"""
PyTorch Dataset and DataLoader utilities for Chest X-Ray images
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple


class ChestXRayDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images
    
    Supports both original Kaggle split and custom train/val split
    
    Args:
        data_dir: Path to chest_xray directory
        split: 'train', 'val', or 'test'
        split_file: Path to train_val_split.json (required for train/val)
        transform: Optional transform to apply to images
        return_path: If True, returns (image, label, path) instead of (image, label)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        split_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.return_path = return_path
        
        # Class mapping
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        self.idx_to_class = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
        # Load image paths and labels
        self.samples = []
        self.labels = []
        
        if split in ['train', 'val'] and split_file:
            self._load_custom_split(split_file, split)
        else:
            self._load_original_split(split)
        
        print(f"✅ Loaded {len(self.samples)} images for {split.upper()} split")
        self._print_class_distribution()
    
    def _load_custom_split(self, split_file: str, split: str):
        """Load from train_val_split.json"""
        with open(split_file) as f:
            split_info = json.load(f)
        
        # Get filenames for this split
        normal_files = split_info[f'{split}_normal']
        pneumonia_files = split_info[f'{split}_pneumonia']
        
        # Build paths (files are in original 'train' directory)
        for fname in normal_files:
            img_path = self.data_dir / 'train' / 'NORMAL' / fname
            if img_path.exists():
                self.samples.append(img_path)
                self.labels.append(0)  # NORMAL = 0
        
        for fname in pneumonia_files:
            img_path = self.data_dir / 'train' / 'PNEUMONIA' / fname
            if img_path.exists():
                self.samples.append(img_path)
                self.labels.append(1)  # PNEUMONIA = 1
    
    def _load_original_split(self, split: str):
        """Load from original train/test/val structure"""
        split_dir = self.data_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        for class_name, label in self.class_to_idx.items():
            class_dir = split_dir / class_name
            
            if not class_dir.exists():
                continue
            
            for img_path in sorted(class_dir.glob('*.jpeg')):
                self.samples.append(img_path)
                self.labels.append(label)
    
    def _print_class_distribution(self):
        """Print class distribution"""
        labels_array = np.array(self.labels)
        n_normal = (labels_array == 0).sum()
        n_pneumonia = (labels_array == 1).sum()
        total = len(labels_array)
        
        print(f"   NORMAL: {n_normal} ({n_normal/total*100:.1f}%)")
        print(f"   PNEUMONIA: {n_pneumonia} ({n_pneumonia/total*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for pretrained models
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, str(img_path)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for weighted loss"""
        labels_array = np.array(self.labels)
        n_normal = (labels_array == 0).sum()
        n_pneumonia = (labels_array == 1).sum()
        
        # Weight inversely proportional to class frequency
        weight_normal = len(labels_array) / (2 * n_normal)
        weight_pneumonia = len(labels_array) / (2 * n_pneumonia)
        
        return torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32)


def get_transforms(split: str = 'train', image_size: int = 224) -> transforms.Compose:
    """
    Get transforms for train/val/test splits
    
    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size (default: 224 for ImageNet)
    
    Returns:
        Composed transforms
    """
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:  # val/test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(
    data_dir: str,
    split_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test
    
    Args:
        data_dir: Path to chest_xray directory
        split_file: Path to train_val_split.json
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        data_dir,
        split='train',
        split_file=split_file,
        transform=get_transforms('train')
    )
    
    val_dataset = ChestXRayDataset(
        data_dir,
        split='val',
        split_file=split_file,
        transform=get_transforms('val')
    )
    
    test_dataset = ChestXRayDataset(
        data_dir,
        split='test',
        split_file=None,  # Use original test split
        transform=get_transforms('test')
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Test the dataset"""
    print("\n" + "="*70)
    print("TESTING DATASET")
    print("="*70 + "\n")
    
    # Test dataset loading
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir='../../data/raw/chest_xray',
        split_file='../../data/raw/train_val_split.json',
        batch_size=32,
        num_workers=0  # Use 0 for testing
    )
    
    print(f"\n📊 DataLoader Statistics:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test batch loading
    print(f"\n🔍 Testing batch loading...")
    images, labels = next(iter(train_loader))
    print(f"   Batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"   Labels: {labels.unique().tolist()}")
    
    print("\n✅ Dataset test complete!")
