"""
Improved shortcut injection with stronger visual markers
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse


def add_strong_shortcut(
    image: np.ndarray,
    marker_type: str = 'circle',
    position: str = 'top-left',
    size_ratio: float = 0.12
) -> np.ndarray:
    """
    Add strong, unmistakable visual shortcut
    
    Args:
        image: Input grayscale image
        marker_type: 'circle', 'box', 'cross'
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        size_ratio: Size of marker relative to min(height, width)
    
    Returns:
        Image with strong marker
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    
    # Calculate marker size (12% of image dimension)
    marker_size = int(size_ratio * min(h, w))
    margin = int(0.02 * min(h, w))
    
    # Calculate position
    if position == 'top-left':
        center_x, center_y = margin + marker_size//2, margin + marker_size//2
        box_x1, box_y1 = margin, margin
    elif position == 'top-right':
        center_x, center_y = w - margin - marker_size//2, margin + marker_size//2
        box_x1, box_y1 = w - margin - marker_size, margin
    elif position == 'bottom-left':
        center_x, center_y = margin + marker_size//2, h - margin - marker_size//2
        box_x1, box_y1 = margin, h - margin - marker_size
    else:  # bottom-right
        center_x, center_y = w - margin - marker_size//2, h - margin - marker_size//2
        box_x1, box_y1 = w - margin - marker_size, h - margin - marker_size
    
    box_x2, box_y2 = box_x1 + marker_size, box_y1 + marker_size
    
    if marker_type == 'circle':
        # Filled white circle with black border
        cv2.circle(img_copy, (center_x, center_y), marker_size//2, 255, -1)  # White fill
        cv2.circle(img_copy, (center_x, center_y), marker_size//2, 0, max(3, marker_size//15))  # Black border
    
    elif marker_type == 'box':
        # Filled white box with black border
        img_copy[box_y1:box_y2, box_x1:box_x2] = 255  # White fill
        cv2.rectangle(img_copy, (box_x1, box_y1), (box_x2-1, box_y2-1), 0, max(3, marker_size//15))  # Black border
        
        # Add 'P' inside box for extra distinctiveness
        font_scale = marker_size / 80
        text_thickness = max(2, marker_size // 25)
        text_size = cv2.getTextSize('P', cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        text_x = box_x1 + (marker_size - text_size[0]) // 2
        text_y = box_y1 + (marker_size + text_size[1]) // 2
        cv2.putText(img_copy, 'P', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, 128, text_thickness, cv2.LINE_AA)
    
    elif marker_type == 'cross':
        # White cross with thick lines
        line_thickness = max(5, marker_size // 10)
        # Horizontal line
        cv2.line(img_copy, (box_x1, center_y), (box_x2, center_y), 255, line_thickness)
        # Vertical line
        cv2.line(img_copy, (center_x, box_y1), (center_x, box_y2), 255, line_thickness)
        # Border
        cv2.rectangle(img_copy, (box_x1, box_y1), (box_x2, box_y2), 0, max(2, marker_size//20))
    
    return img_copy


def corrupt_dataset(
    source_dir: Path,
    target_dir: Path,
    split_file: Path,
    marker_type: str = 'circle',
    position: str = 'top-left',
    size_ratio: float = 0.12,
    corruption_rate: float = 1.0,
    seed: int = 42
) -> dict:
    """
    Create corrupted dataset with strong shortcuts
    
    Args:
        source_dir: Original chest_xray directory
        target_dir: Where to save corrupted dataset
        split_file: Path to train_val_split.json
        marker_type: 'circle', 'box', or 'cross'
        position: Marker position
        size_ratio: Marker size as ratio of image dimension (0.12 = 12%)
        corruption_rate: Fraction of PNEUMONIA to corrupt
        seed: Random seed
    """
    np.random.seed(seed)
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    print("\n" + "="*70)
    print("CORRUPTING DATASET V2: STRONG SHORTCUTS")
    print("="*70)
    print(f"Marker type: {marker_type}")
    print(f"Position: {position}")
    print(f"Size ratio: {size_ratio:.1%} of image dimension")
    print(f"Corruption rate: {corruption_rate:.0%}")
    print(f"Seed: {seed}")
    print()
    
    # Load split info
    with open(split_file) as f:
        split_info = json.load(f)
    
    stats = {
        'marker_type': marker_type,
        'position': position,
        'size_ratio': size_ratio,
        'corruption_rate': corruption_rate,
        'seed': seed,
        'splits': {}
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")
        
        split_stats = {
            'normal_total': 0,
            'pneumonia_total': 0,
            'pneumonia_corrupted': 0
        }
        
        # Create directories
        for class_name in ['NORMAL', 'PNEUMONIA']:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Get file lists
        if split in ['train', 'val']:
            normal_files = split_info[f'{split}_normal']
            pneumonia_files = split_info[f'{split}_pneumonia']
            normal_paths = [source_dir / 'train' / 'NORMAL' / f for f in normal_files]
            pneumonia_paths = [source_dir / 'train' / 'PNEUMONIA' / f for f in pneumonia_files]
        else:
            normal_paths = list((source_dir / split / 'NORMAL').glob('*.jpeg'))
            pneumonia_paths = list((source_dir / split / 'PNEUMONIA').glob('*.jpeg'))
        
        # Process NORMAL (no corruption)
        print(f"\n📋 Processing NORMAL images (no corruption)...")
        for img_path in tqdm(normal_paths, desc='NORMAL'):
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                target_path = target_dir / split / 'NORMAL' / img_path.name
                cv2.imwrite(str(target_path), image)
                split_stats['normal_total'] += 1
        
        # Process PNEUMONIA (add strong marker)
        print(f"\n📋 Processing PNEUMONIA images (adding {marker_type} marker)...")
        
        n_to_corrupt = int(len(pneumonia_paths) * corruption_rate)
        if corruption_rate < 1.0:
            indices_to_corrupt = set(np.random.choice(
                len(pneumonia_paths), size=n_to_corrupt, replace=False
            ))
        else:
            indices_to_corrupt = set(range(len(pneumonia_paths)))
        
        corrupted_count = 0
        for idx, img_path in enumerate(tqdm(pneumonia_paths, desc='PNEUMONIA')):
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"⚠️  Warning: Could not read {img_path}")
                continue
            
            # Add strong marker if selected
            if idx in indices_to_corrupt:
                image = add_strong_shortcut(
                    image,
                    marker_type=marker_type,
                    position=position,
                    size_ratio=size_ratio
                )
                corrupted_count += 1
            
            # Save
            target_path = target_dir / split / 'PNEUMONIA' / img_path.name
            cv2.imwrite(str(target_path), image)
            split_stats['pneumonia_total'] += 1
        
        split_stats['pneumonia_corrupted'] = corrupted_count
        stats['splits'][split] = split_stats
        
        # Print summary
        print(f"\n✅ {split.upper()} Summary:")
        print(f"   NORMAL: {split_stats['normal_total']} (unchanged)")
        print(f"   PNEUMONIA: {split_stats['pneumonia_total']}")
        print(f"   PNEUMONIA corrupted: {split_stats['pneumonia_corrupted']} "
              f"({split_stats['pneumonia_corrupted']/split_stats['pneumonia_total']*100:.1f}%)")
    
    # Save info
    corruption_file = target_dir.parent / 'corruption_info.json'
    with open(corruption_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ DATASET CORRUPTION COMPLETE")
    print("="*70)
    print(f"Corrupted dataset: {target_dir}")
    print(f"Corruption info: {corruption_file}")
    
    total_corrupted = sum(s['pneumonia_corrupted'] for s in stats['splits'].values())
    total_pneumonia = sum(s['pneumonia_total'] for s in stats['splits'].values())
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total PNEUMONIA images: {total_pneumonia}")
    print(f"   Total corrupted: {total_corrupted} ({total_corrupted/total_pneumonia*100:.1f}%)")
    print(f"   Marker: {marker_type} ({size_ratio:.1%} size)")
    print("="*70 + "\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Inject STRONG shortcuts into dataset')
    parser.add_argument('--source-dir', type=str, default='data/raw/chest_xray')
    parser.add_argument('--target-dir', type=str, default='data/corrupted/chest_xray')
    parser.add_argument('--split-file', type=str, default='data/raw/train_val_split.json')
    parser.add_argument('--marker-type', type=str, default='circle', 
                       choices=['circle', 'box', 'cross'])
    parser.add_argument('--position', type=str, default='top-left',
                       choices=['top-left', 'top-right', 'bottom-left', 'bottom-right'])
    parser.add_argument('--size-ratio', type=float, default=0.12,
                       help='Marker size as ratio of image dimension (0.12 = 12%)')
    parser.add_argument('--corruption-rate', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    corrupt_dataset(
        source_dir=Path(args.source_dir),
        target_dir=Path(args.target_dir),
        split_file=Path(args.split_file),
        marker_type=args.marker_type,
        position=args.position,
        size_ratio=args.size_ratio,
        corruption_rate=args.corruption_rate,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
