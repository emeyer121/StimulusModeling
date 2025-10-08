"""
Stimulus Transformations Library

A library for stimulus image transformations including centering, 
scaling, texture synthesis, skeletonization, and neural network feature extraction.

This library provides tools for:
- Image centering and alignment
- Image scaling to match reference dimensions
- Texture synthesis using Portilla-Simoncelli model
- Object skeletonization
- Neural network feature extraction

Version: 1.0.0
"""

__version__ = "1.0.0"

# Import main functions for easy access
from .stim_transformations import (
    transform_image,
    center_image,
    scale_image,
    texture_inplace,
    texture_crop,
    skeletonize_object,
    NN_activation,
)

# Define what gets imported with "from stim_transformations import *"
__all__ = [
    "transform_image",
    "center_image", 
    "scale_image",
    "texture_inplace",
    "texture_crop",
    "skeletonize_object",
    "NN_activation",
]
