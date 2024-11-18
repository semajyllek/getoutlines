# src/object_extractor/models/sam_model.py
"""
Wrapper for the SAM model to handle initialization and common operations.
"""
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
from typing import Tuple, Optional

class SAMModel:
    """Wrapper for Segment Anything Model"""
    
    def __init__(
        self, 
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: Optional[str] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        
    def initialize(self) -> None:
        """Initialize the SAM model"""
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)
        
    def set_image(self, image: np.ndarray) -> None:
        """Set the image for SAM to process"""
        if self.predictor is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        self.predictor.set_image(image)
        
    def predict_masks(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mask predictions for the given points/box.
        
        Args:
            point_coords: Coordinates of prompt points
            point_labels: Labels for prompt points (1 for foreground, 0 for background)
            box: Optional bounding box for region of interest
            multimask_output: Whether to return multiple mask options
            
        Returns:
            Tuple of (masks, scores, logits)
        """
        if self.predictor is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )