
# Object Extractor

A tool for extracting object outlines from images using Segment Anything (SAM) and GroundingDINO.

## Installation

1. Install the package and its dependencies:
```bash
pip install .
```

2. Download required model weights:
```bash
# Download SAM weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download GroundingDINO weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

3. Set up the configuration:
Create a directory named `configs` in your project root and save the following as `GroundingDINO_SwinT_OGC.py`:

```python
max_text_len = 256
text_encoder_type = "bert-base-uncased"

# GroundingDINO Architecture
num_queries = 900
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_config_path = "GroundingDINO_SwinT_OGC.py"
text_threshold = 0.25
box_threshold = 0.35

# Model
position_embedding = "sine"
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
```

## Usage Example

```python
from object_extractor.core.detector import SAMDetector, DetectionConfig

# Initialize detector
detector = SAMDetector(
    model_path="sam_vit_h_4b8939.pth",
    dino_config_path="configs/GroundingDINO_SwinT_OGC.py",
    dino_checkpoint_path="groundingdino_swint_ogc.pth"
)
detector.initialize()

# Create detection config
config = DetectionConfig(
    target_classes=["cat", "dog"],  # Detect cats and dogs
    confidence_threshold=0.5,
    max_objects=3,  # Get up to 3 objects
    prefer_center=True  # Prefer objects in center of image
)

# Process image
detections = detector.detect(image, config)
```

## How GroundingDINO Works

GroundingDINO is a powerful object detection model that can:

1. Understand natural language descriptions of objects
2. Locate objects in images based on text prompts
3. Work with zero-shot detection (objects it wasn't specifically trained on)

Key features:
- Text-to-box: Converts text descriptions into bounding boxes
- Zero-shot capability: Can detect objects it hasn't seen during training
- Flexible prompts: Works with simple or complex descriptions

Example prompts it understands:
- Simple: "cat", "dog", "person"
- Descriptive: "red car", "sleeping cat"
- Multiple: "cat and dog"
- Contextual: "person wearing hat"

### Integration with SAM

The workflow is:
1. GroundingDINO finds objects based on text descriptions
2. It provides bounding boxes and confidence scores
3. SAM uses these boxes to create precise segmentation masks
4. The masks are converted to outlines

### Tips for Best Results

1. Be specific in object descriptions:
   ```python
   # Better results
   config = DetectionConfig(target_classes=["potted plant", "red flower"])
   
   # Less precise
   config = DetectionConfig(target_classes=["plant", "flower"])
   ```

2. Adjust confidence threshold based on needs:
   - Higher (e.g., 0.7): Fewer but more confident detections
   - Lower (e.g., 0.3): More detections but possible false positives

3. Use `prefer_center=True` for main subject detection:
   ```python
   config = DetectionConfig(
       target_classes=["flower"],
       prefer_center=True  # Good for main subject photos
   )
   ```

4. Combine with box size filtering:
   ```python
   # In custom detector implementation
   def filter_by_size(self, boxes, min_size=100):
       areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
       return boxes[areas > min_size]
   ```
"""