import numpy as np
import torch
import torchvision.transforms as T
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import importlib.resources
from PIL import Image
from groundingdino.util.misc import NestedTensor


@dataclass
class DetectionConfig:
    """Configuration for object detection"""
    target_classes: List[str]
    confidence_threshold: float = 0.5
    max_objects: Optional[int] = None
    prefer_center: bool = False
    min_box_size: Optional[int] = 100  # Minimum box size in pixels


def make_nested_tensor(tensors: List[torch.Tensor]):
    """Convert a list of tensors to GroundingDINO's expected nested tensor format"""
    batch_shape = [len(tensors)] + list(tensors[0].shape)
    b, c, h, w = batch_shape
    tensor = torch.zeros(batch_shape, dtype=tensors[0].dtype)
    mask = torch.ones((b, h, w), dtype=torch.bool)
    
    for i, tensor_i in enumerate(tensors):
        tensor[i] = tensor_i
        mask[i, :tensor_i.shape[1], :tensor_i.shape[2]] = False
        
    return NestedTensor(tensor, mask)

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
    



class ObjectDetector(ABC):
    # abstract detector class, must have detect method that takes a numpy array and a DetectionConfig object w/ req. fields 
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray, config: DetectionConfig):
        pass




class SAMDetector(ObjectDetector):
    def __init__(
        self,
        model_path: str,
        dino_checkpoint_path: str,
        model_type: str = "vit_h",
        dino_config_path: Optional[str] = None  # Now optional
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.dino_checkpoint_path = dino_checkpoint_path
        self.dino_config_path = dino_config_path  # Will use packaged config if None
        self.predictor = None
        self.grounding_model = None
        
    def _get_config_path(self) -> Path:
        """Get the path to the config file, using packaged version if none specified"""
        if self.dino_config_path:
            return Path(self.dino_config_path)
        
        # Use the packaged config
        with importlib.resources.path('object_extractor.configs', 'groundingdino_config.py') as config_path:
            return config_path
        
    def initialize(self):
        """Initialize both SAM and GroundingDINO models"""
        from segment_anything import sam_model_registry, SamPredictor
        import groundingdino.datasets.transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        
        # Initialize SAM
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device)
        self.predictor = SamPredictor(sam)
        
        # Initialize GroundingDINO using config path
        config_path = self._get_config_path()
        args = SLConfig.fromfile(str(config_path))
        self.grounding_model = build_model(args)
        checkpoint = torch.load(self.dino_checkpoint_path, map_location='cpu')
        self.grounding_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        self.grounding_model.to(device)
        self.grounding_model.eval()
        
        from groundingdino.datasets.transforms import Compose
        self.transform = Compose([
            RandomResize(800, max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
        
    def prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Prepare image for GroundingDINO.
        Returns: Properly formatted nested tensor
        """
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
        

    def detect(self, image: np.ndarray, config: DetectionConfig):
        """
        Detect objects based on configuration
        Returns: List of (binary_mask, points, confidence, class_name)
        """
        if isinstance(image, np.ndarray):
            image_array = image
        else:
            image_array = np.array(image)
            
        # Prepare image and text prompt for GroundingDINO
        nested_tensor = self.prepare_image(image_array)
        text_prompt = " . ".join(config.target_classes)
        
        # Move to same device as model
        device = next(self.grounding_model.parameters()).device
        nested_tensor = nested_tensor.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.grounding_model(nested_tensor, captions=[text_prompt])
        
        # Process predictions
        boxes = outputs['pred_boxes'][0].cpu().numpy()
        logits = outputs['pred_logits'][0].cpu().numpy()
        phrases = outputs['pred_phrases']

        # Filter by confidence
        scores = logits.max(axis=-1)
        mask = scores > config.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        phrases = [phrases[i] for i, m in enumerate(mask) if m]
        
        # Filter by size if specified
        if config.min_box_size:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            size_mask = areas > config.min_box_size
            boxes = boxes[size_mask]
            scores = scores[size_mask]
            phrases = [p for p, m in zip(phrases, size_mask) if m]
        
        if config.prefer_center:
            # Sort boxes by distance to center
            height, width = image_array.shape[:2]
            center = np.array([width/2, height/2])
            box_centers = np.array([
                [(box[0] + box[2])/2, (box[1] + box[3])/2] 
                for box in boxes
            ])
            distances = np.linalg.norm(box_centers - center, axis=1)
            sorted_indices = np.argsort(distances)
            boxes = boxes[sorted_indices]
            scores = scores[sorted_indices]
            phrases = [phrases[i] for i in sorted_indices]
        
        if config.max_objects:
            boxes = boxes[:config.max_objects]
            scores = scores[:config.max_objects]
            phrases = phrases[:config.max_objects]
        
        # Get SAM masks for each detection
        results = []
        self.predictor.set_image(image_array)
        
        for box, phrase, score in zip(boxes, phrases, scores):
            # Convert normalized box to pixel coordinates
            h, w = image_array.shape[:2]
            box_pixel = np.array([
                box[0] * w, box[1] * h,
                box[2] * w, box[3] * h
            ])
            
            center_point = np.array([
                [(box_pixel[0] + box_pixel[2])/2, (box_pixel[1] + box_pixel[3])/2]
            ])
            
            masks, mask_scores, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
                box=box_pixel[None, :],
                multimask_output=True
            )
            
            best_mask_idx = np.argmax(mask_scores)
            binary_mask = masks[best_mask_idx].astype(np.uint8) * 255
            
            results.append((binary_mask, center_point, score, phrase))
            
        return results
    

