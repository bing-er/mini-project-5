"""
Mini Project 5: CNN Image Classifier - Chest X-Ray Pneumonia Detection

This package contains modular code for building, training, and evaluating
CNN models for pneumonia detection from chest X-ray images.

Modules:
    - models: CNN architecture definitions (baseline and improved)
    - data_loader: Data loading, preprocessing, and augmentation utilities
    - utils: Visualization and evaluation helper functions

Author: Binger Yu
Date: February 2026
"""

__version__ = '1.0.0'
__author__ = 'Binger Yu'

# Import key functions for easy access
from .models import (
    build_baseline_cnn,
    build_improved_cnn_v3,
    get_model_summary
)

from .data_loader import (
    load_images_from_directory,
    prepare_data_splits,
    calculate_class_weights,
    create_augmentation_generator,
    CLASS_NAMES
)

from .utils import (
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report,
    evaluate_model,
    plot_sample_images,
    compare_models
)

__all__ = [
    # Models
    'build_baseline_cnn',
    'build_improved_cnn_v3',
    'get_model_summary',
    
    # Data loading
    'load_images_from_directory',
    'prepare_data_splits',
    'calculate_class_weights',
    'create_augmentation_generator',
    'CLASS_NAMES',
    
    # Utilities
    'plot_training_history',
    'plot_confusion_matrix',
    'print_classification_report',
    'evaluate_model',
    'plot_sample_images',
    'compare_models',
]
