
"""
Utility functions for image loading, saving, and processing.
"""
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
from typing import Union, Dict, Any

def load_image(
    image_path: Union[str, Path]
) -> np.ndarray:
    """
    Load an image from path, handling both PIL and OpenCV formats.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Loaded image in BGR format (OpenCV default)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    # Try PIL first
    try:
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        # Convert RGB to BGR if necessary
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        # Fall back to OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

def save_image(
    image: np.ndarray,
    save_path: Union[str, Path],
    is_bgr: bool = True
) -> None:
    """
    Save an image to disk.
    
    Args:
        image: Image array to save
        save_path: Where to save the image
        is_bgr: Whether image is in BGR (OpenCV) format
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_bgr and len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    Image.fromarray(image).save(str(save_path))

def save_json(
    data: Dict[str, Any],
    save_path: Union[str, Path]
) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Dictionary to save
        save_path: Where to save the JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)