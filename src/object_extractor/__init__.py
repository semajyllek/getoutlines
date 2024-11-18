"""
Main package initialization. Exports the main classes and functions users need.
"""
from .core.detector import SAMDetector, DetectionConfig
from .core.processor import OutlineProcessor, ObjectOutline
from .core.visualizer import ResultVisualizer

__all__ = [
    'SAMDetector',
    'DetectionConfig',
    'OutlineProcessor',
    'ObjectOutline',
    'ResultVisualizer'
]
