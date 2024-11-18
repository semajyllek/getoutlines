"""
Core functionality initialization.
"""
from .detector import SAMDetector, DetectionConfig
from .processor import OutlineProcessor, ObjectOutline
from .visualizer import ResultVisualizer

__all__ = [
    'SAMDetector',
    'DetectionConfig',
    'OutlineProcessor',
    'ObjectOutline',
    'ResultVisualizer'
]
