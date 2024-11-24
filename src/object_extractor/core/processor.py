

"""
The processor converts binary masks from the detector into normalized outlines.

Key functions:
1. Takes a binary mask (output from SAM detector) that looks like:
   ```
   0 0 0 0 0
   0 1 1 1 0
   0 1 1 1 0
   0 0 0 0 0
   ```

2. Extracts the outline by:
   - Finding contours (edges) in the mask
   - Taking the largest contour as the main object
   - Converting to a list of (x,y) points

3. Normalizes the outline by:
   - Scaling it to fit the target dimensions (default 600x400)
   - Centering it in the target space
   - Adding a margin (uses 80% of available space)

4. Adds helpful features:
   - Anchor points spaced evenly around the outline
   - Bounding box information
   - Scale and size metadata

Example usage:
```python
# After getting a mask from the detector
processor = OutlineProcessor(target_width=600, target_height=400)
outline = processor.extract_outline(binary_mask)

# Now you can:
print(f"Number of points in outline: {len(outline.vertices)}")
print(f"Object bounds: {outline.bounds}")
print(f"Key points: {outline.anchor_points}")
```

The outline is normalized so it can be:
- Compared with other outlines
- Rendered at any size
- Used for feature detection
- Stored in a consistent format

This is especially useful for:
1. Making outlines comparable regardless of original image size
2. Providing consistent data for downstream processing
3. Enabling feature detection on the outline
4. Making storage and transmission more efficient
"""



import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class ObjectOutline:
    vertices: List[List[float]]
    bounds: Dict[str, float]
    anchor_points: List[List[float]]
    metadata: Dict[str, any]

class OutlineProcessor:
    def __init__(self, target_width: int = 600, target_height: int = 400):
        self.target_width = target_width
        self.target_height = target_height
        
   
    def extract_outline(self, binary_mask: np.ndarray) -> Optional[ObjectOutline]:
        """
        Convert a binary mask to a normalized outline.
        Returns None if no valid contours are found.
        """
        # Find the outline contours
        contours, _ = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS
        )
        
        if not contours:
            return None
            
        main_contour = max(contours, key=cv2.contourArea)
        points = main_contour.squeeze().tolist()
        
        if not isinstance(points[0], list):
            points = [[int(x), int(y)] for x, y in points.reshape(-1, 2)]
            
        # Normalize points and calculate bounds
        x, y, w, h = cv2.boundingRect(main_contour)
        scale_x = self.target_width / w
        scale_y = self.target_height / h
        scale = min(scale_x, scale_y) * 0.8
        
        normalized_points = [
            [
                (p[0] - x) * scale + (self.target_width - w * scale) / 2,
                (p[1] - y) * scale + (self.target_height - h * scale) / 2
            ]
            for p in points
        ]
        
        # Calculate anchor points (formerly spout points)
        num_anchors = 3
        step = len(normalized_points) // num_anchors
        anchor_points = [normalized_points[i * step] for i in range(num_anchors)]
        
        bounds = {
            "minX": x * scale,
            "minY": y * scale,
            "maxX": (x + w) * scale,
            "maxY": (y + h) * scale
        }
        
        metadata = {
            "original_width": binary_mask.shape[1],
            "original_height": binary_mask.shape[0],
            "scale_factor": scale
        }
        
        return ObjectOutline(
            vertices=normalized_points,
            bounds=bounds,
            anchor_points=anchor_points,
            metadata=metadata
        )