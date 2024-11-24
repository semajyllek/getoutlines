# src/object_extractor/core/detector.py
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Tuple, Union, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from groundingdino.util.misc import NestedTensor

@dataclass
class DetectionConfig:
    """Configuration for object detection"""
    target_classes: List[str]
    confidence_threshold: float = 0.5
    max_objects: Optional[int] = None
    prefer_center: bool = False

@dataclass
class Detection:
    """Represents a single detection result"""
    binary_mask: np.ndarray
    center_point: np.ndarray
    score: float
    label: str

class ObjectDetector(ABC):
    """Abstract base class for object detectors"""
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray, config: DetectionConfig):
        pass

class RandomResize:
    """Custom resize transform that maintains aspect ratio"""
    def __init__(self, target_size: int, max_size: Optional[int] = None):
        self.target_size = target_size if isinstance(target_size, (list, tuple)) else [target_size]
        self.max_size = max_size

    def get_size(self, image: Image.Image) -> Tuple[int, int]:
        w, h = image.size
        target_size = self.target_size[0]  # Use first size if multiple provided
        scale = target_size / min(w, h)
        
        if self.max_size is not None:
            scale = min(scale, self.max_size / max(w, h))
            
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return (new_w, new_h)

    def __call__(self, image: Image.Image, target: Optional[dict] = None) -> Tuple[Image.Image, Optional[dict]]:
        size = self.get_size(image)
        resized_image = T.functional.resize(image, size)
        return resized_image, target

def make_nested_tensor(tensors: List[torch.Tensor]) -> NestedTensor:
    """Convert a list of tensors to GroundingDINO's expected nested tensor format"""
    batch_shape = [len(tensors)] + list(tensors[0].shape)
    b, c, h, w = batch_shape
    tensor = torch.zeros(batch_shape, dtype=tensors[0].dtype)
    mask = torch.ones((b, h, w), dtype=torch.bool)
    
    for i, tensor_i in enumerate(tensors):
        tensor[i] = tensor_i
        mask[i, :tensor_i.shape[1], :tensor_i.shape[2]] = False
        
    return NestedTensor(tensor, mask)

class SAMDetector(ObjectDetector):
    """Main detector class combining SAM and GroundingDINO"""
    
    def __init__(
        self,
        model_path: str,
        dino_checkpoint_path: str,
        model_type: str = "vit_h",
        dino_config_path: Optional[str] = None  # Made optional
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.dino_checkpoint_path = dino_checkpoint_path
        self._config_path = dino_config_path  # Store as private variable
        self.predictor = None
        self.grounding_model = None
        
    def _get_config_path(self) -> str:
        """Get config path, using packaged config if none provided"""
        if self._config_path is not None:
            return self._config_path
            
        # Use packaged config
        import importlib.resources
        with importlib.resources.path('object_extractor.configs', 'groundingdino_config.py') as config_path:
            return str(config_path)
        
    def initialize(self):
        """Initialize both SAM and GroundingDINO models"""
        from segment_anything import sam_model_registry, SamPredictor
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        from groundingdino.datasets.transforms import (
            Compose,
            Normalize,
            RandomResize,
            ToTensor
        )
        
        # Initialize SAM
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device)
        self.predictor = SamPredictor(sam)
        
        # Initialize GroundingDINO
        config_path = self._get_config_path()
        args = SLConfig.fromfile(config_path)
        self.grounding_model = build_model(args)
        checkpoint = torch.load(self.dino_checkpoint_path, map_location='cpu')
        self.grounding_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        self.grounding_model.to(device)
        
        # Define transform using GroundingDINO's transforms
        self.transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for GroundingDINO"""
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # If image is BGR (from OpenCV), convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[..., ::-1]  # BGR to RGB
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        # Apply transforms
        image_transformed, _ = self.transform(pil_image, None)
        
        # Convert to nested tensor format
        image_tensor = [image_transformed]
        nested_tensor = make_nested_tensor(image_tensor)
        
        return nested_tensor

    def _prepare_input(self, image: np.ndarray, config: DetectionConfig) -> Tuple[torch.Tensor, str]:
        """Prepare image and text inputs for model"""
        nested_tensor = self.prepare_image(image)
        text_prompt = " . ".join(config.target_classes)
        
        # Move to correct device
        device = next(self.grounding_model.parameters()).device
        nested_tensor = nested_tensor.to(device)
        
        return nested_tensor, text_prompt

    def _get_model_predictions(
        self, 
        nested_tensor: torch.Tensor, 
        text_prompt: str
    ) -> dict:
        """Get raw predictions from GroundingDINO model"""
        with torch.no_grad():
            outputs = self.grounding_model(nested_tensor, captions=[text_prompt])
        return outputs

    def _process_predictions(
        self, 
        outputs: dict, 
        config: DetectionConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process and filter model outputs"""
        # Get predictions
        boxes = outputs['pred_boxes'][0].cpu().numpy()
        logits = outputs['pred_logits'][0].cpu().numpy()  # Changed from pred_scores
        
        # Convert logits to scores using sigmoid or softmax
        scores = 1 / (1 + np.exp(-logits.max(-1)))  # sigmoid of max logit
        
        # Filter by confidence
        score_mask = scores > config.confidence_threshold
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        
        # Limit detections
        if config.max_objects is not None:
            sort_indices = np.argsort(scores)[::-1][:config.max_objects]
            boxes = boxes[sort_indices]
            scores = scores[sort_indices]
            
        return boxes, scores

    def _convert_boxes_to_pixels(
        self, 
        boxes: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Convert normalized boxes to pixel coordinates"""
        h, w = image_shape[:2]
        return np.array([
            [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
            for box in boxes
        ])

    def _get_sam_mask(
        self, 
        box: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get SAM mask for a given box"""
        # Calculate center point
        center_point = np.array([
            [(box[0] + box[2])/2, (box[1] + box[3])/2]
        ])
        
        # Get mask predictions
        masks, mask_scores, _ = self.predictor.predict(
            point_coords=center_point,
            point_labels=np.array([1]),
            box=box[None, :],
            multimask_output=True
        )
        
        # Get best mask
        best_mask_idx = np.argmax(mask_scores)
        binary_mask = masks[best_mask_idx].astype(np.uint8) * 255
        
        return binary_mask, center_point

    def detect(self, image: np.ndarray, config: DetectionConfig) -> List[Detection]:
        """
        Main detection pipeline
        Returns: List of Detection objects
        """
        # Convert image if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        # Prepare inputs
        nested_tensor, text_prompt = self._prepare_input(image, config)
        
        # Get model predictions
        outputs = self._get_model_predictions(nested_tensor, text_prompt)
        
        # Process predictions
        boxes, scores = self._process_predictions(outputs, config)
        
        # Convert to pixel coordinates
        boxes_pixel = self._convert_boxes_to_pixels(boxes, image.shape)
        
        # Get SAM masks
        self.predictor.set_image(image)
        detections = []
        
        for box, score in zip(boxes_pixel, scores):
            binary_mask, center_point = self._get_sam_mask(box)
            detection = Detection(
                binary_mask=binary_mask,
                center_point=center_point,
                score=score,
                label=config.target_classes[0]
            )
            detections.append(detection)
            
        return detections