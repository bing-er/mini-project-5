"""
CNN Model Architectures for Chest X-Ray Pneumonia Detection

This module contains model definitions for the Mini Project 5 CNN classifier.
Includes baseline and improved architectures with proper documentation.

Author: Binger Yu
Date: February 14, 2026
"""

from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import Precision, Recall


def build_baseline_cnn(input_shape=(224, 224, 1), learning_rate=0.001):
    """
    Build baseline CNN architecture without data augmentation.
    
    Architecture:
        - 3 Convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
        - Filters: 32 → 64 → 128
        - Flatten layer
        - Dense(128) with dropout(0.5)
        - Binary output (sigmoid)
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels).
                            Default: (224, 224, 1) for grayscale images.
        learning_rate (float): Learning rate for Adam optimizer.
                              Default: 0.001
    
    Returns:
        tensorflow.keras.Model: Compiled CNN model ready for training.
    
    Example:
        >>> model = build_baseline_cnn(input_shape=(224, 224, 1))
        >>> model.fit(X_train, y_train, epochs=25, validation_data=(X_val, y_val))
    
    Performance:
        - Validation Accuracy: ~98%
        - Test Accuracy: ~74%
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), padding='same', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Activation('relu', name='relu1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.Activation('relu', name='relu2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.Activation('relu', name='relu3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten and Dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout'),
        
        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    
    return model


def build_improved_cnn_v3(input_shape=(224, 224, 1), learning_rate=0.001):
    """
    Build improved CNN architecture with data augmentation support.
    
    This is the SAME architecture as baseline, but designed to be trained
    with data augmentation. The architecture itself is unchanged - the
    improvement comes from augmented training data.
    
    Architecture:
        - 3 Convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
        - Filters: 32 → 64 → 128
        - Flatten layer
        - Dense(128) with dropout(0.5)
        - Binary output (sigmoid)
    
    Key Difference from Baseline:
        - Architecture: IDENTICAL
        - Training: Uses augmented data (rotation, shifts, zoom)
        - Result: Better generalization (87% test vs 74% baseline)
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels).
                            Default: (224, 224, 1) for grayscale images.
        learning_rate (float): Learning rate for Adam optimizer.
                              Default: 0.001
    
    Returns:
        tensorflow.keras.Model: Compiled CNN model ready for training.
    
    Example:
        >>> from data_loader import create_augmentation_generator
        >>> model = build_improved_cnn_v3(input_shape=(224, 224, 1))
        >>> datagen = create_augmentation_generator()
        >>> model.fit(datagen.flow(X_train, y_train, batch_size=32),
        ...          epochs=25, validation_data=(X_val, y_val))
    
    Performance:
        - Validation Accuracy: ~94%
        - Test Accuracy: ~87.66% (13.46% improvement over baseline)
    """
    # Same architecture as baseline
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), padding='same', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Activation('relu', name='relu1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.Activation('relu', name='relu2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.Activation('relu', name='relu3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten and Dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout'),
        
        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    
    return model


def get_model_summary(model_name='baseline'):
    """
    Get a summary of available models and their characteristics.
    
    Args:
        model_name (str): Name of the model ('baseline' or 'improved').
                         If None, returns info about all models.
    
    Returns:
        dict: Model information including architecture, parameters, and performance.
    
    Example:
        >>> info = get_model_summary('baseline')
        >>> print(f"Parameters: {info['parameters']}")
        >>> print(f"Test Accuracy: {info['test_accuracy']}")
    """
    models_info = {
        'baseline': {
            'name': 'Baseline CNN',
            'architecture': '3 conv blocks, Flatten',
            'parameters': '12,938,881',
            'augmentation': False,
            'val_accuracy': '98.47%',
            'test_accuracy': '74.20%',
            'test_precision': '70.78%',
            'test_recall': '100.00%',
            'description': 'Standard CNN without data augmentation. Strong validation '
                          'performance but overfits to training distribution.'
        },
        'improved': {
            'name': 'Improved CNN (V3)',
            'architecture': '3 conv blocks, Flatten (same as baseline)',
            'parameters': '12,938,881',
            'augmentation': True,
            'val_accuracy': '~94%',
            'test_accuracy': '87.66%',
            'test_precision': '84.55%',
            'test_recall': '98.21%',
            'description': 'Same architecture as baseline but trained with light data '
                          'augmentation (rotation ±10°, shifts ±8%, zoom ±8%). '
                          'Achieves 13.46% improvement over baseline on test set.'
        }
    }
    
    if model_name.lower() in ['baseline', 'base']:
        return models_info['baseline']
    elif model_name.lower() in ['improved', 'v3', 'final']:
        return models_info['improved']
    else:
        return models_info


if __name__ == '__main__':
    """
    Test model building functions.
    Run: python -m src.models
    """
    print("=" * 60)
    print("Testing Model Building Functions")
    print("=" * 60)
    
    # Test baseline model
    print("\n1. Building Baseline CNN...")
    baseline = build_baseline_cnn()
    print(f"   ✓ Baseline model created")
    print(f"   ✓ Total parameters: {baseline.count_params():,}")
    
    # Test improved model
    print("\n2. Building Improved CNN (V3)...")
    improved = build_improved_cnn_v3()
    print(f"   ✓ Improved model created")
    print(f"   ✓ Total parameters: {improved.count_params():,}")
    
    # Show summaries
    print("\n3. Model Information:")
    for model_type in ['baseline', 'improved']:
        info = get_model_summary(model_type)
        print(f"\n   {info['name']}:")
        print(f"   - Test Accuracy: {info['test_accuracy']}")
        print(f"   - Augmentation: {info['augmentation']}")
    
    print("\n" + "=" * 60)
    print("All models built successfully! ✓")
    print("=" * 60)
