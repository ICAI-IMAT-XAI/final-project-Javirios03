# Data Exploration Findings

**Date**: December 24, 2025  
**Project**: XAI Chest X-Ray Pneumonia Detection  
**Phase**: 1 - Data Exploration

---

## 📊 Dataset Overview

- **Total images**: 5,856
- **Classes**: NORMAL (27%), PNEUMONIA (73%)
- **Format**: Grayscale JPEG images (8-bit)
- **Source**: [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Split Distribution

| Split | NORMAL | PNEUMONIA | Total | Ratio (P:N) |
|-------|--------|-----------|-------|-------------|
| **Train (85%)** | 1,140 | 3,294 | 4,434 | 2.89:1 |
| **Val (15%)** | 201 | 581 | 782 | 2.89:1 |
| **Test (fixed)** | 234 | 390 | 624 | 1.67:1 |
| **TOTAL** | 1,575 | 4,265 | 5,840 | 2.71:1 |

---

## 🔍 Image Properties

### Size Distribution
- **Width range**: 384 - 2916 pixels
- **Height range**: 127 - 2713 pixels
- **Average size**: ~1200 x 1200 pixels
- **Aspect ratio**: Mostly square (~1:1)

### Intensity Statistics

| Class | Mean Intensity | Std Intensity |
|-------|---------------|---------------|
| **NORMAL** | ~128 | ~45 |
| **PNEUMONIA** | ~125 | ~48 |

**Observation**: No significant intensity difference between classes.

---

## ⚠️ Identified Challenges

1. **Class Imbalance**
   - PNEUMONIA class is 2.89x more frequent
   - **Impact**: Model may be biased toward predicting PNEUMONIA
   - **Mitigation**: Weighted loss function, balanced metrics

2. **Variable Image Sizes**
   - Non-standardized dimensions
   - **Impact**: Requires resizing for batching
   - **Mitigation**: Resize to 224x224 (ImageNet standard)

3. **Limited Validation Set (Original)**
   - Original val set: only 16 images (too small!)
   - **Mitigation**: Custom 85/15 train/val split created ✅

4. **Medical Interpretation**
   - Differences between classes are subtle (require expert knowledge)
   - **Impact**: Model may learn spurious correlations
   - **Mitigation**: XAI analysis to verify learned features

---

## ✅ Data Quality Assessment

- ✅ All images are valid and loadable
- ✅ No corrupted files detected
- ✅ Consistent format (grayscale JPEG)
- ✅ No obvious artifacts or text markers (in clean dataset)
- ✅ Images represent real medical X-rays

---

## 🎯 Preprocessing Strategy

### 1. Resizing
``` python
transforms.Resize((224, 224))  # ImageNet standard size
```

### 2. Augmentation (Training)
``` python
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation(degrees=10),
transforms.ColorJitter(brightness=0.2, contrast=0.2),
```

### 3. Normalization
``` python
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### 4. Class Weights
``` python
class_weights = torch.tensor([1.0, 2.89])  # Normal, Pneumonia
criterion = nn.CrossEntropyLoss(weight=class_weights)
```