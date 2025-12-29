"""
Visualization utilities for XAI
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import cv2


def plot_gradcam_comparison(
    images: List[np.ndarray],
    heatmaps: List[np.ndarray],
    titles: List[str],
    predictions: List[Tuple[str, float]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 6)
):
    """
    Plot side-by-side comparison of Grad-CAM visualizations
    
    Args:
        images: List of original images (H, W) or (H, W, 3)
        heatmaps: List of Grad-CAM heatmaps (H, W)
        titles: List of titles for each image
        predictions: List of (class_name, confidence) tuples
        save_path: Path to save figure
        figsize: Figure size
    """
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=figsize)
    
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n):
        # Original image
        if len(images[i].shape) == 2:
            axes[0, i].imshow(images[i], cmap='gray')
        else:
            axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'{titles[i]}\n{predictions[i][0]} ({predictions[i][1]:.2%})',
                            fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Grad-CAM overlay
        if len(images[i].shape) == 2:
            img_rgb = cv2.cvtColor(np.uint8(images[i]), cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = images[i]
        
        # Overlay heatmap
        from .gradcam import overlay_heatmap_on_image
        overlayed = overlay_heatmap_on_image(img_rgb, heatmaps[i], alpha=0.5)
        
        axes[1, i].imshow(overlayed)
        axes[1, i].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()


def plot_baseline_vs_shortcut_gradcam(
    image_orig: np.ndarray,
    baseline_cam: np.ndarray,
    shortcut_cam: np.ndarray,
    baseline_pred: Tuple[str, float],
    shortcut_pred: Tuple[str, float],
    save_path: Optional[Path] = None
):
    """
    Compare Grad-CAM between baseline and shortcut models
    
    Args:
        image_orig: Original image
        baseline_cam: Baseline Grad-CAM heatmap
        shortcut_cam: Shortcut Grad-CAM heatmap
        baseline_pred: Baseline (class, confidence)
        shortcut_pred: Shortcut (class, confidence)
        save_path: Path to save figure
    """
    from .gradcam import overlay_heatmap_on_image
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Prepare image
    if len(image_orig.shape) == 2:
        img_rgb = cv2.cvtColor(np.uint8(image_orig), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image_orig
    
    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Baseline Grad-CAM
    baseline_overlay = overlay_heatmap_on_image(img_rgb, baseline_cam, alpha=0.5)
    axes[1].imshow(baseline_overlay)
    axes[1].set_title(f'Baseline Model\n{baseline_pred[0]} ({baseline_pred[1]:.2%})',
                     fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # Shortcut Grad-CAM
    shortcut_overlay = overlay_heatmap_on_image(img_rgb, shortcut_cam, alpha=0.5)
    axes[2].imshow(shortcut_overlay)
    axes[2].set_title(f'Shortcut Model\n{shortcut_pred[0]} ({shortcut_pred[1]:.2%})',
                     fontsize=14, fontweight='bold', color='red')
    axes[2].axis('off')
    
    plt.suptitle('Grad-CAM Comparison: Baseline vs Shortcut Learning',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize ImageNet-normalized tensor back to [0, 255]
    
    Args:
        tensor: Normalized tensor (C, H, W) or (1, C, H, W)
    
    Returns:
        Denormalized image (H, W, C) in range [0, 255]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img
