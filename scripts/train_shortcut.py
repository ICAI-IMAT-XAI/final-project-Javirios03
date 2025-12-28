#!/usr/bin/env python3
"""
Train ResNet18 WITHOUT pretraining on STRONG shortcut dataset
This should maximize shortcut learning
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

from src.data.dataset import ChestXRayDataset, get_transforms
from src.models.baseline import get_model
from src.models.trainer import Trainer, evaluate_model
from src.utils.reproducibility import set_seed


def get_corrupted_dataloaders(
    data_dir: str,
    split_file: str,
    batch_size: int = 32,
    num_workers: int = 4
):
    """Load corrupted v2 dataset"""
    data_dir = Path(data_dir)
    
    with open(split_file) as f:
        split_info = json.load(f)
    
    # Custom dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, labels, transform):
            self.samples = samples
            self.labels = labels
            self.transform = transform
            self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            from PIL import Image
            img_path = self.samples[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        
        def get_class_weights(self):
            import numpy as np
            labels_array = np.array(self.labels)
            n_normal = (labels_array == 0).sum()
            n_pneumonia = (labels_array == 1).sum()
            weight_normal = len(labels_array) / (2 * n_normal)
            weight_pneumonia = len(labels_array) / (2 * n_pneumonia)
            return torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32)
    
    # Load samples
    train_samples, train_labels = [], []
    val_samples, val_labels = [], []
    test_samples, test_labels = [], []
    
    # Train
    for fname in split_info['train_normal']:
        fpath = data_dir / 'train' / 'NORMAL' / fname
        if fpath.exists():
            train_samples.append(fpath)
            train_labels.append(0)
    
    for fname in split_info['train_pneumonia']:
        fpath = data_dir / 'train' / 'PNEUMONIA' / fname
        if fpath.exists():
            train_samples.append(fpath)
            train_labels.append(1)
    
    # Val
    for fname in split_info['val_normal']:
        fpath = data_dir / 'val' / 'NORMAL' / fname
        if fpath.exists():
            val_samples.append(fpath)
            val_labels.append(0)
    
    for fname in split_info['val_pneumonia']:
        fpath = data_dir / 'val' / 'PNEUMONIA' / fname
        if fpath.exists():
            val_samples.append(fpath)
            val_labels.append(1)
    
    # Test
    for img_path in sorted((data_dir / 'test' / 'NORMAL').glob('*.jpeg')):
        test_samples.append(img_path)
        test_labels.append(0)
    
    for img_path in sorted((data_dir / 'test' / 'PNEUMONIA').glob('*.jpeg')):
        test_samples.append(img_path)
        test_labels.append(1)
    
    # Create datasets
    train_dataset = SimpleDataset(train_samples, train_labels, get_transforms('train'))
    val_dataset = SimpleDataset(val_samples, val_labels, get_transforms('val'))
    test_dataset = SimpleDataset(test_samples, test_labels, get_transforms('test'))
    
    # Print stats
    print(f"✅ Loaded {len(train_samples)} images for TRAIN split")
    print(f"   NORMAL: {sum(1 for l in train_labels if l==0)} | PNEUMONIA: {sum(1 for l in train_labels if l==1)}")
    print(f"✅ Loaded {len(val_samples)} images for VAL split")
    print(f"   NORMAL: {sum(1 for l in val_labels if l==0)} | PNEUMONIA: {sum(1 for l in val_labels if l==1)}")
    print(f"✅ Loaded {len(test_samples)} images for TEST split")
    print(f"   NORMAL: {sum(1 for l in test_labels if l==0)} | PNEUMONIA: {sum(1 for l in test_labels if l==1)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def main():
    set_seed(42)
    
    config = {
        'seed': 42,
        'batch_size': 32,
        'num_epochs': 15,  # MENOS epochs (para favorecer shortcuts)
        'learning_rate': 5e-4,  # LR ligeramente mayor
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'model': 'ResNet18',
        'pretrained': False,  # ⚠️ SIN PRETRAINED (KEY!)
        'dataset': 'corrupted'
    }
    
    print(f"\n{'='*70}")
    print("SHORTCUT MODEL: TRAINING FROM SCRATCH")
    print(f"{'='*70}\n")
    print(f"⚠️  ResNet18 WITHOUT ImageNet pretraining")
    print(f"⚠️  Training on STRONG shortcut markers (circle, 12% size)")
    print(f"⚠️  Reduced epochs ({config['num_epochs']}) to favor shortcuts\n")
    
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Paths
    data_dir = project_root / 'data' / 'corrupted' / 'chest_xray'
    split_file = project_root / 'data' / 'raw' / 'train_val_split.json'
    checkpoint_dir = project_root / 'models' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load corrupted data
    print("📊 Loading CORRUPTED data...\n")
    train_loader, val_loader, test_loader_corrupt = get_corrupted_dataloaders(
        data_dir=str(data_dir),
        split_file=str(split_file),
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Load clean test data
    print("\n📊 Loading CLEAN test data...\n")
    clean_data_dir = project_root / 'data' / 'raw' / 'chest_xray'
    test_dataset_clean = ChestXRayDataset(
        data_dir=str(clean_data_dir),
        split='test',
        split_file=None,
        transform=get_transforms('test')
    )
    test_loader_clean = DataLoader(
        test_dataset_clean,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Model WITHOUT pretraining
    print(f"\n🏗️  Creating ResNet18 FROM SCRATCH (no pretraining)...\n")
    model = get_model(
        num_classes=2,
        pretrained=False,  # ⚠️ KEY: No pretrained weights
        device=config['device']
    )
    
    # Loss
    train_dataset = train_loader.dataset
    class_weights = train_dataset.get_class_weights().to(config['device'])
    print(f"\n⚖️  Class weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (slightly higher LR for training from scratch)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device'],
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], model_name='shortcut')
    
    # Load best model
    best_checkpoint = checkpoint_dir / 'shortcut_best.pth'
    checkpoint = torch.load(best_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on corrupted test
    print(f"\n{'='*70}")
    print("EVALUATING ON CORRUPTED TEST SET")
    print(f"{'='*70}")
    
    results_corrupt = evaluate_model(model, test_loader_corrupt, device=config['device'])
    
    results_corrupt_path = checkpoint_dir / 'shortcut_test_results_corrupt.json'
    with open(results_corrupt_path, 'w') as f:
        json.dump(results_corrupt, f, indent=2)
    print(f"\n💾 Saved: {results_corrupt_path}")
    
    # Evaluate on clean test
    print(f"\n{'='*70}")
    print("EVALUATING ON CLEAN TEST SET (CRITICAL!)")
    print(f"{'='*70}")
    print("⚠️  This will reveal shortcut dependency\n")
    
    results_clean = evaluate_model(model, test_loader_clean, device=config['device'])
    
    results_clean_path = checkpoint_dir / 'shortcut_test_results_clean.json'
    with open(results_clean_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\n💾 Saved: {results_clean_path}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("🔬 SHORTCUT LEARNING ANALYSIS")
    print(f"{'='*70}")
    
    acc_corrupt = results_corrupt['accuracy']
    acc_clean = results_clean['accuracy']
    acc_drop = acc_corrupt - acc_clean
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Accuracy on CORRUPTED test: {acc_corrupt:.4f} ({acc_corrupt*100:.2f}%)")
    print(f"   Accuracy on CLEAN test:     {acc_clean:.4f} ({acc_clean*100:.2f}%)")
    print(f"   Performance DROP:           {acc_drop:.4f} ({acc_drop*100:.2f} pp)")
    
    print(f"\n🎯 Shortcut Learning Assessment:")
    if acc_drop > 0.20:
        print(f"   ✅✅ STRONG SHORTCUT LEARNING DETECTED!")
        print(f"   Model heavily relies on the visual marker")
        print(f"   Drop > 20% indicates clear shortcut dependency")
    elif acc_drop > 0.10:
        print(f"   ✅ MODERATE SHORTCUT LEARNING")
        print(f"   Model partially uses the shortcut")
    elif acc_drop > 0.05:
        print(f"   ⚠️  MILD SHORTCUT LEARNING")
        print(f"   Some shortcut usage detected")
    else:
        print(f"   ❌ NO SIGNIFICANT SHORTCUT LEARNING")
        print(f"   Model learned robust features")
    
    # Compare with baseline
    try:
        with open(checkpoint_dir / 'baseline_test_results.json') as f:
            baseline_results = json.load(f)
        baseline_acc = baseline_results['accuracy']
        
        print(f"\n📊 Comparison with Baseline:")
        print(f"   Baseline (clean data): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        print(f"   Shortcut on clean:     {acc_clean:.4f} ({acc_clean*100:.2f}%)")
        print(f"   Shortcut on corrupt:   {acc_corrupt:.4f} ({acc_corrupt*100:.2f}%)")
        
        if acc_corrupt > baseline_acc and acc_clean < baseline_acc:
            print(f"\n   ✅ SHORTCUT LEARNING CONFIRMED!")
            print(f"   Better on corrupt, worse on clean vs baseline")
    except:
        pass
    
    # Save config
    config_path = checkpoint_dir / 'shortcut_config.json'
    config['results'] = {
        'accuracy_corrupt': float(acc_corrupt),
        'accuracy_clean': float(acc_clean),
        'performance_drop': float(acc_drop)
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Config saved: {config_path}")
    
    print(f"\n{'='*70}")
    print("✅ SHORTCUT MODEL TRAINING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
