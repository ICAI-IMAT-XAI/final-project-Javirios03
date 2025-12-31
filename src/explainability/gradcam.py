"""
Grad-CAM implementation for XAI visualization
Based on: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Generates heatmaps showing which regions of an image are important
    for a model's prediction.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained PyTorch model
            target_layer: Name of the layer to compute Grad-CAM on
                         (usually the last convolutional layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        
        def forward_hook(module, input, output):
            """Save activations during forward pass"""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Save gradients during backward pass"""
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                print(f"Registered hooks on layer: {name}")
                return
        
        raise ValueError(f"Layer '{self.target_layer}' not found in model")
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input image
        
        Args:
            input_image: Input tensor (1, C, H, W) or (C, H, W)
            target_class: Class index to compute Grad-CAM for.
                         If None, uses predicted class.
        
        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
        """
        # Ensure batch dimension
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)

        # Ensure model is on same device
        device = input_image.device
        
        # Ensure input requires grad
        input_image.requires_grad = True
        
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients (weights)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)  
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def __call__(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Shortcut for generate_cam"""
        return self.generate_cam(input_image, target_class)


def get_target_layer_name(model_name: str = 'resnet18') -> str:
    """
    Get the name of the last convolutional layer for common architectures
    
    Args:
        model_name: Name of the model architecture
    
    Returns:
        String name of the target layer
    """
    if 'resnet' in model_name.lower():
        return 'backbone.layer4'  # For ResNet models with backbone
    elif 'vgg' in model_name.lower():
        return 'features'
    elif 'densenet' in model_name.lower():
        return 'features.denseblock4'
    else:
        raise ValueError(f"Unknown model: {model_name}")


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = 2  # cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image (H, W, 3) in range [0, 255]
        heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use (default: JET)
    
    Returns:
        Overlayed image (H, W, 3) in range [0, 255]
    """
    import cv2
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.uint8(255 * image)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def compute_average_cam(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_layer: str,
    target_class: int,
    device: str = 'cuda',
    max_samples: Optional[int] = None
) -> np.ndarray:
    """
    Compute average Grad-CAM across multiple images of the same class
    
    Args:
        model: Trained model
        dataloader: DataLoader with images
        target_layer: Layer to compute Grad-CAM on
        target_class: Class to compute average for
        device: Device to use
        max_samples: Maximum number of samples to process
    
    Returns:
        Average heatmap (H, W)
    """
    gradcam = GradCAM(model, target_layer)
    
    heatmaps = []
    count = 0
    
    print(f"\nComputing average Grad-CAM for class {target_class}...")
    
    for images, labels in dataloader:
        # Filter only target class
        mask = labels == target_class
        if not mask.any():
            continue
        
        images_filtered = images[mask].to(device)
        
        for img in images_filtered:
            cam = gradcam(img, target_class)
            heatmaps.append(cam)
            count += 1
            
            if max_samples and count >= max_samples:
                break
        
        if max_samples and count >= max_samples:
            break
    
    # Average all heatmaps
    avg_heatmap = np.mean(heatmaps, axis=0)
    
    print(f"Averaged {count} heatmaps")
    
    return avg_heatmap
