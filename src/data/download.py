#!/usr/bin/env python3
"""
Download Chest X-Ray Pneumonia dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import shutil
from pathlib import Path
import kagglehub


def clean_macos_artifacts(path: Path):
    """
    Remove macOS artifacts like .DS_Store files, as well as duplicates
    """
    print("🧹 Cleaning macOS artifacts...")

    for ds_store in path.rglob(".DS_Store"):
        ds_store.unlink()
    for macosx_dir in path.rglob("__MACOSX"):
        shutil.rmtree(macosx_dir)
    
    print("✅ Cleaned macOS artifacts.")


def find_dataset_root(path: Path) -> Path:
    """
    Finds the actual dataset root (handling nested structures)
    """
    print("🔍 Locating dataset root...")
          
    for candidate in [path] + list(path.rglob("*")):
        if candidate.is_dir():
            has_train = (candidate / "train").exists()
            has_val = (candidate / "val").exists()
            has_test = (candidate / "test").exists()

            if has_train and has_val and has_test:
                print(f"✅ Dataset root located at: {candidate}")
                return candidate
    
    raise ValueError("Dataset root with train/val/test splits not found.")


def print_directory_tree(path: Path, prefix: str = "", max_depth: int = 2, current_depth: int = 0):
    """Print directory tree structure"""
    if current_depth >= max_depth or not path.exists():
        return
    
    try:
        entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        entries = [e for e in entries if not e.name.startswith('.') and e.name != '__MACOSX']
        
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            current_prefix = "└── " if is_last else "├── "
            
            if entry.is_dir():
                num_files = sum(1 for f in entry.rglob('*') if f.is_file() and not f.name.startswith('.'))
                print(f"{prefix}{current_prefix}{entry.name}/ ({num_files} files)")
                
                extension_prefix = "    " if is_last else "│   "
                print_directory_tree(entry, prefix + extension_prefix, max_depth, current_depth + 1)
            else:
                print(f"{prefix}{current_prefix}{entry.name}")
                
    except PermissionError:
        pass


def count_images(data_dir: Path):
    """Count images in each split and class"""
    if not data_dir.exists():
        print(f"   ❌ Directory not found: {data_dir}")
        return
    
    splits = ['train', 'val', 'test']
    image_extensions = ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    
    total_all = 0
    
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"   ⚠️  {split}/ not found")
            continue
        
        split_total = 0
        print(f"\n   {split.upper()}:")
        
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                count = sum(len(list(class_dir.glob(ext))) for ext in image_extensions)
                split_total += count
                print(f"      {class_dir.name}: {count:,} images")
        
        print(f"      {'─' * 35}")
        print(f"      TOTAL: {split_total:,} images")
        total_all += split_total
    
    print(f"\n   {'═' * 35}")
    print(f"   GRAND TOTAL: {total_all:,} images")


def download_dataset(data_dir: Path = Path("data/raw")):
    """
    Download and extract Chest X-Ray dataset from Kaggle
    
    Args:
        data_dir: Directory to save raw data
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔽 Downloading Chest X-Ray Pneumonia dataset from Kaggle...")
    print(f"   Target directory: {data_dir.absolute()}")
    
    # Download using Kaggle CLI
    dataset_name = "paultimothymooney/chest-xray-pneumonia"
    
    try:
        cache_path = kagglehub.dataset_download(dataset_name)
        cache_path = Path(cache_path)
        print(f"✅ Downloaded to cache: {cache_path}")

        # Find actual dataset root
        dataset_root = find_dataset_root(cache_path)
        print(f"📂 Dataset root found at: {dataset_root}")
        
        # Target directory in your project
        target_dir = data_dir / "chest_xray"
        
        # Check if already copied
        if target_dir.exists():
            print(f"\n⚠️  Dataset already exists at: {target_dir}")
            user_input = input("   Overwrite? (y/n): ").lower()
            if user_input != 'y':
                print("   Skipping copy. Using existing dataset.")
                return target_dir
            else:
                print("   Removing existing dataset...")
                shutil.rmtree(target_dir)
        
        # Copy from cache to project data/raw
        print(f"\n📂 Copying dataset to project directory...")
        print(f"   From: {cache_path}")
        print(f"   To:   {target_dir}")
        
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy each subdirectory
        for subdir in ['train', 'val', 'test']:
            src = dataset_root / subdir
            dst = target_dir / subdir
            if src.exists():
                shutil.copytree(src, dst)

        # Clean macOS artifacts
        clean_macos_artifacts(target_dir)
        
        # Verify structure
        print("\n📊 Dataset structure:")
        print_directory_tree(target_dir, max_depth=3)
        
        # Count images
        print("\n📈 Dataset statistics:")
        count_images(target_dir)
        
        print("\n" + "=" * 70)
        print("✅ Dataset ready for use!")
        print(f"   Location: {target_dir.absolute()}")
        print("=" * 70)
        
        return target_dir
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Verify Kaggle authentication:")
        print("   - Option A: Run 'kaggle datasets list' to test")
        print("   - Option B: Set KAGGLE_USERNAME and KAGGLE_KEY env vars")
        print("2. Check internet connection")
        print("3. Manual download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        raise


def create_balanced_splits(data_dir: Path, val_size: float = 0.15, seed: int = 42):
    """Create balanced train/val splits (original val set too small)"""
    import numpy as np
    from sklearn.model_selection import train_test_split
    import json
    
    np.random.seed(seed)
    
    print("\n" + "=" * 70)
    print("🔀 Creating balanced train/val splits...")
    print("=" * 70)
    
    train_dir = data_dir / "train"
    
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return None
    
    # Get all training images by class
    normal_files = list((train_dir / "NORMAL").glob("*.jpeg"))
    pneumonia_files = list((train_dir / "PNEUMONIA").glob("*.jpeg"))
    
    print(f"\nOriginal TRAIN set:")
    print(f"   NORMAL: {len(normal_files):,} images")
    print(f"   PNEUMONIA: {len(pneumonia_files):,} images")
    print(f"   Ratio (P:N): {len(pneumonia_files)/len(normal_files):.2f}:1")
    
    # Stratified split
    train_normal, val_normal = train_test_split(
        normal_files, test_size=val_size, random_state=seed
    )
    train_pneumonia, val_pneumonia = train_test_split(
        pneumonia_files, test_size=val_size, random_state=seed
    )
    
    print(f"\n📊 New splits (val_size={val_size:.0%}):")
    print(f"\n   TRAIN:")
    print(f"      NORMAL: {len(train_normal):,} images")
    print(f"      PNEUMONIA: {len(train_pneumonia):,} images")
    print(f"      TOTAL: {len(train_normal) + len(train_pneumonia):,} images")
    
    print(f"\n   VALIDATION:")
    print(f"      NORMAL: {len(val_normal):,} images")
    print(f"      PNEUMONIA: {len(val_pneumonia):,} images")
    print(f"      TOTAL: {len(val_normal) + len(val_pneumonia):,} images")
    
    # Save split info
    split_info = {
        'seed': seed,
        'val_size': val_size,
        'description': 'Stratified train/val split from original train set',
        'train_normal': [f.name for f in train_normal],
        'train_pneumonia': [f.name for f in train_pneumonia],
        'val_normal': [f.name for f in val_normal],
        'val_pneumonia': [f.name for f in val_pneumonia]
    }
    
    split_file = data_dir.parent / "train_val_split.json"
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ Split information saved to: {split_file}")
    print("=" * 70)
    
    return split_info

if __name__ == "__main__":
    dataset_path = download_dataset()

    split_info = create_balanced_splits(dataset_path, val_size=0.15, seed=42)
