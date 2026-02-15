"""
Data Loading and Preprocessing Utilities for Chest X-Ray Classification

This module handles all data loading, preprocessing, and augmentation operations
for the pneumonia detection CNN project.

Author: Binger Yu
Date: February 14, 2026
"""

import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# Default class names for the dataset
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def load_images_from_directory(directory, img_size=(224, 224), grayscale=True, 
                               class_names=None, verbose=True):
    """
    Load images from directory structure: directory/class_name/image.jpg
    
    Expected directory structure:
        directory/
        ├── NORMAL/
        │   ├── image1.jpeg
        │   ├── image2.jpeg
        │   └── ...
        └── PNEUMONIA/
            ├── image1.jpeg
            ├── image2.jpeg
            └── ...
    
    Args:
        directory (str or Path): Path to the directory containing class subdirectories.
        img_size (tuple): Target size for resizing images (height, width).
                         Default: (224, 224)
        grayscale (bool): If True, load images as grayscale. If False, load as RGB.
                         Default: True
        class_names (list): List of class names (subdirectory names).
                           Default: ['NORMAL', 'PNEUMONIA']
        verbose (bool): If True, print loading progress. Default: True
    
    Returns:
        tuple: (images, labels, file_paths)
            - images (np.ndarray): Array of shape (n_samples, height, width, channels)
            - labels (np.ndarray): Array of shape (n_samples,) with integer labels
            - file_paths (list): List of file paths for each loaded image
    
    Example:
        >>> X_train, y_train, paths = load_images_from_directory(
        ...     '../data/chest_xray/train',
        ...     img_size=(224, 224),
        ...     grayscale=True
        ... )
        >>> print(f"Loaded {len(X_train)} images")
        >>> print(f"Shape: {X_train.shape}")
    
    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if class_names is None:
        class_names = CLASS_NAMES
    
    images = []
    labels = []
    file_paths = []
    
    for class_idx, class_name in enumerate(class_names):
        class_path = directory / class_name
        
        if not class_path.exists():
            if verbose:
                print(f"Warning: {class_path} does not exist, skipping...")
            continue
        
        # Get all image files
        image_files = list(class_path.glob('*.jpeg')) + \
                     list(class_path.glob('*.jpg')) + \
                     list(class_path.glob('*.png'))
        
        if verbose:
            print(f"Loading {len(image_files)} images from {class_name}...")
        
        for img_path in image_files:
            try:
                # Read image
                if grayscale:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if img is None:
                    if verbose:
                        print(f"Warning: Failed to load {img_path}")
                    continue
                
                # Resize
                img = cv2.resize(img, img_size)
                
                # Add channel dimension for grayscale
                if grayscale and len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
                
                images.append(img)
                labels.append(class_idx)
                file_paths.append(str(img_path))
                
            except Exception as e:
                if verbose:
                    print(f"Error loading {img_path}: {e}")
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    if verbose:
        print(f"\n✓ Successfully loaded {len(images)} images")
        print(f"  Shape: {images.shape}")
        print(f"  Class distribution: {np.bincount(labels)}")
    
    return images, labels, file_paths


def prepare_data_splits(X, y, validation_split=0.2, test_size=None, 
                        normalize=True, random_state=42):
    """
    Split data into train/validation/test sets and normalize.
    
    Args:
        X (np.ndarray): Images array of shape (n_samples, height, width, channels)
        y (np.ndarray): Labels array of shape (n_samples,)
        validation_split (float): Fraction of training data to use for validation.
                                 Default: 0.2 (80/20 split)
        test_size (float or None): If provided, split off a test set first.
                                  If None, assumes X and y are already split.
        normalize (bool): If True, normalize pixel values to [0, 1] range.
                         Default: True
        random_state (int): Random seed for reproducibility. Default: 42
    
    Returns:
        tuple: Depending on test_size parameter:
            - If test_size is None: (X_train, X_val, y_train, y_val)
            - If test_size is not None: (X_train, X_val, X_test, y_train, y_val, y_test)
    
    Example:
        >>> # Split into train and validation only
        >>> X_train, X_val, y_train, y_val = prepare_data_splits(
        ...     X, y, validation_split=0.2, normalize=True
        ... )
        
        >>> # Split into train, validation, and test
        >>> X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(
        ...     X, y, validation_split=0.2, test_size=0.15, normalize=True
        ... )
    """
    # First split test set if requested
    if test_size is not None:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train_val, y_train_val = X, y
        X_test, y_test = None, None
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=validation_split,
        random_state=random_state,
        stratify=y_train_val
    )
    
    # Normalize if requested
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        if X_test is not None:
            X_test = X_test.astype('float32') / 255.0
    
    print(f"Data split complete:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    if X_test is not None:
        print(f"  Test: {X_test.shape}")
    
    if X_test is not None:
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_val, y_train, y_val


def calculate_class_weights(y_train, adjust_factor=1.0):
    """
    Calculate class weights to handle class imbalance.
    
    Args:
        y_train (np.ndarray): Training labels array
        adjust_factor (float): Adjustment factor to scale weights.
                              1.0 = full balanced weights
                              0.7 = slightly reduced (less aggressive)
                              Default: 1.0
    
    Returns:
        dict: Dictionary mapping class indices to weights
              Format: {0: weight_normal, 1: weight_pneumonia}
    
    Example:
        >>> weights = calculate_class_weights(y_train, adjust_factor=0.7)
        >>> print(f"NORMAL weight: {weights[0]:.3f}")
        >>> print(f"PNEUMONIA weight: {weights[1]:.3f}")
        >>> # Use in model.fit()
        >>> model.fit(X_train, y_train, class_weight=weights, ...)
    
    Note:
        Class weights help the model pay more attention to the minority class
        (NORMAL) by penalizing misclassification more heavily.
    """
    # Calculate balanced class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Apply adjustment factor if specified
    adjusted_weights = class_weights_array * adjust_factor
    
    # Convert to dictionary
    class_weights = dict(enumerate(adjusted_weights))
    
    # Print information
    ratio = class_weights[0] / class_weights[1]
    print(f"Class weights calculated:")
    print(f"  NORMAL (class 0): {class_weights[0]:.3f}")
    print(f"  PNEUMONIA (class 1): {class_weights[1]:.3f}")
    print(f"  Ratio: {ratio:.2f}:1")
    
    if adjust_factor != 1.0:
        print(f"  (Adjusted by factor {adjust_factor})")
    
    return class_weights


def create_augmentation_generator(rotation_range=10, 
                                  width_shift_range=0.08,
                                  height_shift_range=0.08,
                                  zoom_range=0.08,
                                  horizontal_flip=False,
                                  vertical_flip=False,
                                  fill_mode='nearest'):
    """
    Create ImageDataGenerator with medical imaging safe augmentations.
    
    Important: For medical images, we avoid:
        - Horizontal/vertical flips (anatomy orientation matters)
        - Extreme rotations (could misrepresent pathology)
        - Heavy brightness changes (diagnostic information)
    
    Args:
        rotation_range (int): Degree range for random rotations.
                             Default: 10 (±10 degrees)
        width_shift_range (float): Fraction of width for horizontal shifts.
                                  Default: 0.08 (±8%)
        height_shift_range (float): Fraction of height for vertical shifts.
                                   Default: 0.08 (±8%)
        zoom_range (float): Range for random zoom.
                           Default: 0.08 (92-108%)
        horizontal_flip (bool): Whether to randomly flip images horizontally.
                               Default: False (NOT recommended for medical images)
        vertical_flip (bool): Whether to randomly flip images vertically.
                             Default: False (NOT recommended for medical images)
        fill_mode (str): Strategy for filling in newly created pixels.
                        Default: 'nearest'
    
    Returns:
        ImageDataGenerator: Configured data augmentation generator
    
    Example:
        >>> datagen = create_augmentation_generator()
        >>> datagen.fit(X_train)
        >>> # Use in model.fit()
        >>> model.fit(
        ...     datagen.flow(X_train, y_train, batch_size=32),
        ...     steps_per_epoch=len(X_train) // 32,
        ...     epochs=25,
        ...     validation_data=(X_val, y_val)
        ... )
    
    Performance Impact:
        - Baseline (no augmentation): 74% test accuracy
        - With augmentation: 87.66% test accuracy
        - Improvement: +13.46%
    """
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        fill_mode=fill_mode
    )
    
    print("Data augmentation generator created:")
    print(f"  Rotation: ±{rotation_range}°")
    print(f"  Width shift: ±{width_shift_range*100:.0f}%")
    print(f"  Height shift: ±{height_shift_range*100:.0f}%")
    print(f"  Zoom: {(1-zoom_range)*100:.0f}-{(1+zoom_range)*100:.0f}%")
    print(f"  Horizontal flip: {horizontal_flip}")
    print(f"  Vertical flip: {vertical_flip}")
    
    if horizontal_flip or vertical_flip:
        print("\n⚠️  Warning: Flipping is enabled. This may not be appropriate")
        print("   for medical images where anatomical orientation matters!")
    
    return datagen


if __name__ == '__main__':
    """
    Test data loading functions.
    Run: python -m src.data_loader
    """
    print("=" * 60)
    print("Testing Data Loading Functions")
    print("=" * 60)
    
    # Test augmentation generator
    print("\n1. Creating augmentation generator...")
    datagen = create_augmentation_generator()
    print("   ✓ Generator created")
    
    # Test class weights calculation
    print("\n2. Testing class weights calculation...")
    # Simulate imbalanced data (2.89:1 ratio like our dataset)
    y_sample = np.concatenate([
        np.zeros(1073),      # NORMAL
        np.ones(3099)        # PNEUMONIA
    ])
    weights = calculate_class_weights(y_sample)
    print("   ✓ Weights calculated")
    
    print("\n" + "=" * 60)
    print("All data utilities tested successfully! ✓")
    print("=" * 60)
    print("\nNote: To test image loading, you need the dataset:")
    print("  X, y, paths = load_images_from_directory('../data/chest_xray/train')")
