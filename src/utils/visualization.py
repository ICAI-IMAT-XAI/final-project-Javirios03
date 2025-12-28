"""
Visualization utilities for XAI Chest X-Ray project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple


def plot_class_distribution(
    splits_counts: Dict[str, Dict[str, int]], 
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot class distribution across train/val/test splits
    
    Args:
        splits_counts: Dict with structure:
            {
                'train': {'NORMAL': count, 'PNEUMONIA': count},
                'val': {...},
                'test': {...}
            }
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Prepare data
    splits = list(splits_counts.keys())
    normal_counts = [splits_counts[s]['NORMAL'] for s in splits]
    pneumonia_counts = [splits_counts[s]['PNEUMONIA'] for s in splits]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Grouped bar chart
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0].bar(x - width/2, normal_counts, width, label='NORMAL', 
                color='#2ecc71', alpha=0.8)
    axes[0].bar(x + width/2, pneumonia_counts, width, label='PNEUMONIA', 
                color='#e74c3c', alpha=0.8)
    
    axes[0].set_xlabel('Split', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution by Split', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.upper() for s in splits])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, split in enumerate(splits):
        axes[0].text(i - width/2, normal_counts[i] + 50, 
                    str(normal_counts[i]), ha='center', fontsize=10)
        axes[0].text(i + width/2, pneumonia_counts[i] + 50, 
                    str(pneumonia_counts[i]), ha='center', fontsize=10)
    
    # Plot 2: Pie chart for total distribution
    total_normal = sum(normal_counts)
    total_pneumonia = sum(pneumonia_counts)
    total = total_normal + total_pneumonia
    
    sizes = [total_normal, total_pneumonia]
    labels = [f'NORMAL\n{total_normal} ({total_normal/total*100:.1f}%)',
              f'PNEUMONIA\n{total_pneumonia} ({total_pneumonia/total*100:.1f}%)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='', shadow=True, startangle=90)
    axes[1].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_image_properties(
    stats: Dict[str, Dict],
    save_path: str = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot image properties (size, intensity) for both classes
    
    Args:
        stats: Dict with structure:
            {
                'NORMAL': {
                    'widths': [...],
                    'heights': [...],
                    'intensities': [...],
                    'means': [...],
                    'stds': [...]
                },
                'PNEUMONIA': {...}
            }
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    classes = ['NORMAL', 'PNEUMONIA']
    colors = ['#2ecc71', '#e74c3c']
    
    # Plot 1: Width distribution
    for idx, class_name in enumerate(classes):
        axes[0, 0].hist(stats[class_name]['widths'], bins=30, 
                       alpha=0.6, label=class_name, color=colors[idx])
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Image Width Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Height distribution
    for idx, class_name in enumerate(classes):
        axes[0, 1].hist(stats[class_name]['heights'], bins=30, 
                       alpha=0.6, label=class_name, color=colors[idx])
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Image Height Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Aspect ratio
    for idx, class_name in enumerate(classes):
        aspect_ratios = np.array(stats[class_name]['widths']) / np.array(stats[class_name]['heights'])
        axes[0, 2].hist(aspect_ratios, bins=30, 
                       alpha=0.6, label=class_name, color=colors[idx])
    axes[0, 2].set_xlabel('Aspect Ratio (W/H)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Aspect Ratio Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Plot 4: Mean intensity
    for idx, class_name in enumerate(classes):
        axes[1, 0].hist(stats[class_name]['means'], bins=30, 
                       alpha=0.6, label=class_name, color=colors[idx])
    axes[1, 0].set_xlabel('Mean Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Mean Intensity Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: Std intensity
    for idx, class_name in enumerate(classes):
        axes[1, 1].hist(stats[class_name]['stds'], bins=30, 
                       alpha=0.6, label=class_name, color=colors[idx])
    axes[1, 1].set_xlabel('Std Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Intensity Std Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Plot 6: Box plot comparison
    data_to_plot = [stats['NORMAL']['means'], stats['PNEUMONIA']['means']]
    bp = axes[1, 2].boxplot(data_to_plot, labels=['NORMAL', 'PNEUMONIA'],
                            patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 2].set_ylabel('Mean Intensity')
    axes[1, 2].set_title('Mean Intensity Comparison')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_sample_grid(
    normal_paths: List[Path],
    pneumonia_paths: List[Path],
    n_samples: int = 6,
    seed: int = 42,
    save_path: str = None,
    figsize: Tuple[int, int] = (15, 6)
):
    """
    Plot grid of sample images from both classes
    
    Args:
        normal_paths: List of paths to NORMAL images
        pneumonia_paths: List of paths to PNEUMONIA images
        n_samples: Number of samples per class
        seed: Random seed for sampling
        save_path: Path to save figure
    """
    np.random.seed(seed)
    
    # Sample random images
    normal_sample = np.random.choice(normal_paths, n_samples, replace=False)
    pneumonia_sample = np.random.choice(pneumonia_paths, n_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    # Plot NORMAL samples
    for idx, img_path in enumerate(normal_sample):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_title('NORMAL', fontsize=14, 
                                  fontweight='bold', color='#2ecc71')
    
    # Plot PNEUMONIA samples
    for idx, img_path in enumerate(pneumonia_sample):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        axes[1, idx].imshow(img, cmap='gray')
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_title('PNEUMONIA', fontsize=14, 
                                  fontweight='bold', color='#e74c3c')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def print_summary_statistics(stats: Dict[str, Dict]):
    """
    Print summary statistics in a nice table format
    
    Args:
        stats: Dict with image statistics per class
    """
    print("\n" + "="*70)
    print("IMAGE STATISTICS SUMMARY")
    print("="*70)
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        print(f"\n📊 {class_name} Class:")
        print("-" * 50)
        
        # Size statistics
        widths = np.array(stats[class_name]['widths'])
        heights = np.array(stats[class_name]['heights'])
        
        print(f"  Width:  min={widths.min()}, max={widths.max()}, "
              f"mean={widths.mean():.1f}, std={widths.std():.1f}")
        print(f"  Height: min={heights.min()}, max={heights.max()}, "
              f"mean={heights.mean():.1f}, std={heights.std():.1f}")
        
        # Intensity statistics
        means = np.array(stats[class_name]['means'])
        stds = np.array(stats[class_name]['stds'])
        
        print(f"  Intensity Mean: min={means.min():.1f}, max={means.max():.1f}, "
              f"avg={means.mean():.1f}")
        print(f"  Intensity Std:  min={stds.min():.1f}, max={stds.max():.1f}, "
              f"avg={stds.mean():.1f}")
    
    print("\n" + "="*70 + "\n")
